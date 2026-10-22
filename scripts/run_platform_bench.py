"""Per-operation cryptographic cost on the host it is run on.

The evaluation's end-to-end latencies were measured on a desktop-class
Arm64 machine. A reviewer of a robotics venue is entitled to ask what
they become on a robot, because the answer is not a rescaling of one
number: the protocol's cost is dominated by pairing-based verification
at the operator and by a single hash-to-curve signature per member, and
those two scale differently across platforms.

This script measures the primitives in isolation so the same file can be
run on a workstation and on an onboard computer and the two compared
directly. It deliberately does NOT run the fleet simulation: transport
is simulated either way, so re-running it on other hardware would mix a
real difference (crypto) with an unchanged model (network) and blur the
comparison.

Emits data/platform_bench.jsonl, one record per operation, plus a
platform block identifying the host.

Run on each platform:
    python3 scripts/run_platform_bench.py --label "workstation"
    python3 scripts/run_platform_bench.py --label "jetson-orin"
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blspy import PopSchemeMPL  # noqa: E402

from ibsf import bls  # noqa: E402
from ibsf.federation import Fleet, canonical  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
# Verification cost grows with the participant count, so it is measured
# at the fleet sizes the paper reports rather than at one point.
FLEET_SIZES = (10, 100, 500)
REPS = 100
WARMUP = 10


def _cpu_model() -> str:
    """Best-effort CPU identification, for the provenance block."""
    for cmd in (
        ["sysctl", "-n", "machdep.cpu.brand_string"],          # macOS
        ["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"],
        ["sh", "-c", "tr -d '\\0' < /proc/device-tree/model"],  # Jetson
    ):
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5
            ).stdout.strip()
            if out:
                return out
        except Exception:
            continue
    return "unknown"


def _time(fn, reps: int = REPS) -> dict:
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(samples)
    return {
        "median_ms": round(float(np.median(a)), 4),
        "p95_ms": round(float(np.percentile(a, 95)), 4),
        "min_ms": round(float(a.min()), 4),
        "reps": reps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--label", required=True,
        help="platform name as it should appear in the paper, "
             "e.g. 'workstation' or 'jetson-orin'",
    )
    ap.add_argument("--append", action="store_true",
                    help="append to the existing file instead of replacing")
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "platform_bench.jsonl"

    host = {
        "label": args.label,
        "cpu": _cpu_model(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    logger.info("benchmarking on %s (%s)", args.label, host["cpu"])

    records = []
    sk = bls.keygen(b"platform-bench/key")
    msg = canonical(0, 0, b"platform-bench")

    # per-member cost: one hash-to-curve signature
    records.append({"operation": "member_sign", "n": None,
                    **_time(lambda: PopSchemeMPL.sign(sk, msg)), **host})

    # commissioning cost: PoP generation and verification
    proof, _ = bls.pop_prove(sk)
    pk = sk.get_g1()
    records.append({"operation": "pop_prove", "n": None,
                    **_time(lambda: bls.pop_prove(sk)), **host})
    records.append({"operation": "pop_verify", "n": None,
                    **_time(lambda: PopSchemeMPL.pop_verify(pk, proof)),
                    **host})

    # operator-side costs, which are the ones that scale with fleet size
    for n in FLEET_SIZES:
        fleet = Fleet.create(n, 9000 + n)
        roster = fleet.roster_pks
        t = fleet.threshold
        by_pk = {r.pk_bytes: r.sk for r in fleet.robots}
        signed = fleet.operator.rosters[fleet.epoch]
        idx = list(range(t))
        sigs = [PopSchemeMPL.sign(by_pk[signed.member_pks[i]], msg)
                for i in idx]
        agg, _ = bls.aggregate(sigs)
        bitmap = fleet.bitmap_for(idx)
        sig_o, _ = bls.countersign_subset(
            fleet.operator.sk_o, msg, agg, bitmap, roster, fleet.epoch, t
        )
        subset = [roster[i] for i in idx]

        for label, fn, reps in (
            ("aggregate", lambda: bls.aggregate(sigs), 20),
            ("fast_aggregate_verify",
             lambda: PopSchemeMPL.fast_aggregate_verify(subset, msg, agg), 20),
            ("verify_subset",
             lambda: bls.verify_subset(msg, agg, sig_o, roster, bitmap,
                                       fleet.operator.pk_o, t, fleet.epoch),
             20),
        ):
            records.append({"operation": label, "n": n,
                            **_time(fn, reps), **host})
        logger.info("n=%d done", n)

    mode = "a" if args.append else "w"
    with out.open(mode) as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    logger.info("%d records (%s) -> %s", len(records), args.label, out.name)

    med = {r["operation"]: r["median_ms"] for r in records if r["n"] is None}
    logger.info("member_sign %.3f ms | pop_verify %.3f ms",
                med["member_sign"], med["pop_verify"])


if __name__ == "__main__":
    main()
