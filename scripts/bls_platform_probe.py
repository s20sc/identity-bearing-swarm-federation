#!/usr/bin/env python3
"""Standalone BLS12-381 cost probe, for running on other hardware.

Self-contained on purpose: it imports nothing from this repository and
needs no numpy, so it can be copied to a bare ARM64 host, a Jetson, or
any machine with Python and `blspy`, and run without checking out the
project. Copy the single file across, `pip install blspy`, run it, and
bring the JSON back.

    pip install blspy
    python3 bls_platform_probe.py --label "arm64-server" > probe.json

Why this exists. The paper's end-to-end latencies were measured on a
desktop-class machine, and a robotics reviewer is entitled to ask what
they become on a robot. The answer is not a single rescaling factor,
because the protocol's cost has two parts that scale differently: each
member pays one hash-to-curve signature, while the operator pays a
fixed number of pairings plus N cheap group additions. Measuring the
primitives separately is what lets a reader carry the numbers to a
platform neither we nor they have benchmarked.

The verification sweep over participant counts is the load-bearing one.
If verification cost were linear in N, the construction would not scale;
if it is close to flat, a slower platform shifts the curve without
changing its shape.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time

try:
    # AugSchemeMPL/G1Element are imported to prove the whole surface this
    # probe reports on is present, not merely the two calls it times.
    from blspy import (  # noqa: F401
        AugSchemeMPL,
        G1Element,
        PopSchemeMPL,
        PrivateKey,
    )
except ImportError:                                     # pragma: no cover
    sys.exit("blspy is required:  pip install blspy")

import hashlib

PARTICIPANTS = (10, 100, 500)
REPS_CHEAP = 200          # sub-millisecond operations
REPS_DEAR = 30            # operations that build an N-key aggregate
WARMUP = 10


def cpu_model() -> str:
    """Best-effort CPU identification across macOS, Linux, and Jetson."""
    probes = (
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        ["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"],
        ["sh", "-c", "tr -d '\\0' < /proc/device-tree/model 2>/dev/null"],
        ["sh", "-c", "lscpu | grep -m1 'Model name' | cut -d: -f2"],
    )
    for cmd in probes:
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5
            ).stdout.strip()
            if out:
                return " ".join(out.split())
        except Exception:
            continue
    return "unknown"


def time_ms(fn, reps: int) -> dict:
    for _ in range(WARMUP):
        fn()
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000.0)
    xs.sort()
    return {
        "median_ms": round(statistics.median(xs), 4),
        "p95_ms": round(xs[min(int(0.95 * len(xs)), len(xs) - 1)], 4),
        "min_ms": round(xs[0], 4),
        "reps": reps,
    }


def keygen(seed: bytes) -> PrivateKey:
    return PopSchemeMPL.key_gen(hashlib.sha256(seed).digest())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="platform name for the paper, e.g. 'jetson-orin'")
    ap.add_argument("--max-n", type=int, default=500,
                    help="largest participant count to sweep")
    ap.add_argument("--out", default=None,
                    help="write JSON here instead of stdout; used by "
                         "run_all.py so the local row of the cross-platform "
                         "table is refreshed by the same single pass that "
                         "regenerates everything else")
    args = ap.parse_args()

    host = {
        "label": args.label,
        "cpu": cpu_model(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cores": None,
    }
    try:
        import os as _os
        host["loadavg_1min"] = round(_os.getloadavg()[0], 2)
    except Exception:
        host["loadavg_1min"] = None
    try:
        import os
        host["cores"] = os.cpu_count()
    except Exception:
        pass

    msg = b"ibsf/platform-probe/v1"
    sk = keygen(b"probe/operator")

    # Self-reported scheduling jitter, on the same instrument the main
    # suite gates on. The suite runs every experiment in one pass under
    # that gate; this probe cannot, because its whole point is to run on
    # a machine the suite is not running on. Recording the ratio is what
    # keeps the cross-platform table's conditions inspectable instead of
    # merely asserted, and a reader can discard a row taken under a
    # ratio they consider too high.
    _probe_sk = keygen(b"probe/jitter")
    _samples = []
    for _ in range(60):
        _t = time.perf_counter()
        PopSchemeMPL.sign(_probe_sk, msg)
        _samples.append(time.perf_counter() - _t)
    _samples.sort()
    _med = _samples[len(_samples) // 2]
    _p95 = _samples[min(int(0.95 * len(_samples)), len(_samples) - 1)]
    host["scheduling_jitter_p95_over_median"] = round(_p95 / _med, 3)

    pk = sk.get_g1()
    sig = PopSchemeMPL.sign(sk, msg)
    proof = PopSchemeMPL.pop_prove(sk)

    results = [
        {"operation": "member_sign", "n": None,
         **time_ms(lambda: PopSchemeMPL.sign(sk, msg), REPS_CHEAP)},
        {"operation": "single_verify", "n": None,
         **time_ms(lambda: PopSchemeMPL.verify(pk, msg, sig), REPS_CHEAP)},
        {"operation": "pop_prove", "n": None,
         **time_ms(lambda: PopSchemeMPL.pop_prove(sk), REPS_CHEAP)},
        {"operation": "pop_verify", "n": None,
         **time_ms(lambda: PopSchemeMPL.pop_verify(pk, proof), REPS_CHEAP)},
    ]

    for n in [x for x in PARTICIPANTS if x <= args.max_n]:
        sks = [keygen(f"probe/member/{n}/{i}".encode()) for i in range(n)]
        pks = [s.get_g1() for s in sks]
        sigs = [PopSchemeMPL.sign(s, msg) for s in sks]
        agg = PopSchemeMPL.aggregate(sigs)
        results.append({
            "operation": "aggregate", "n": n,
            **time_ms(lambda: PopSchemeMPL.aggregate(sigs), REPS_DEAR),
        })
        results.append({
            "operation": "fast_aggregate_verify", "n": n,
            **time_ms(
                lambda: PopSchemeMPL.fast_aggregate_verify(pks, msg, agg),
                REPS_DEAR,
            ),
        })
        print(f"# n={n} done", file=sys.stderr)

    doc = {"host": host, "results": results}
    if args.out:
        import pathlib
        target = pathlib.Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"# wrote {target}", file=sys.stderr)
    else:
        json.dump(doc, sys.stdout, indent=2)
        print(file=sys.stdout)

    by = {(r["operation"], r["n"]): r["median_ms"] for r in results}
    print(
        f"# {args.label}: sign {by[('member_sign', None)]:.3f} ms | "
        f"agg-verify n=10 {by[('fast_aggregate_verify', 10)]:.3f} ms | "
        f"n={max(x for x in PARTICIPANTS if x <= args.max_n)} "
        f"{by[('fast_aggregate_verify', max(x for x in PARTICIPANTS if x <= args.max_n))]:.3f} ms",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
