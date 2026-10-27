"""Run all experiments sequentially, then regenerate reports.

Real BLS and HMAC wall-clock measurements vary with machine load. Run
the suite in one session to reduce differences in measurement conditions
across tables. Scheduling jitter is measured and gated; both jitter and
load are recorded so the conditions remain inspectable.

Rebuild reports from the newly emitted data in the same invocation to
keep summary.json and sign_check.json consistent with the trial logs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
REPORTS = ROOT / "reports"

# The pre-flight jitter probe imports ibsf in THIS process, unlike the
# experiment runners, which are launched as subprocesses and put the repo
# root on their own path.
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Order matters only in that make_tables must come last: it reads every
# JSONL file the preceding scripts emit.
EXPERIMENTS = (
    "run_attacks.py",
    "run_sim.py",
    "run_pbft_baseline.py",
    "run_reconstruction.py",
    "run_certsize.py",
    "run_epoch_sensitivity.py",
    "run_failure_injection.py",
    "run_wireless.py",
    "run_microbench.py",
    # Regenerates this host's row of the cross-platform table. Rows for
    # other hardware come from the standalone probe run there and are
    # left alone: the point of that table is to compare hosts, so a
    # local pass must refresh its own row without touching the others.
    ("bls_platform_probe.py", "--label", "workstation-m1max",
     "--out", "data/platform_probe_workstation.json"),
)
# make_tables reads every JSONL the experiments emit. verify_cert runs
# twice on purpose: once with --export to rebuild the committed artifact
# under the current signed formats, then once with no arguments, which is
# the file-only path. The second run loads nothing but the exported files
# and the operator public key in a fresh process, so it demonstrates that
# an external party can verify the shipped artifact rather than that our
# own in-memory objects agree with themselves.
AGGREGATORS = (
    "make_tables.py",
    ("verify_cert.py", "--export"),
    "verify_cert.py",

)


def _loadavg() -> list[float]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:                       # not available on this platform
        return []


def _run(step: str | tuple[str, ...]) -> tuple[str, float]:
    name, *extra = (step,) if isinstance(step, str) else step
    label = " ".join([name, *extra])
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / name), *extra], cwd=ROOT, check=False
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {proc.returncode}")
    logger.info("%s finished in %.1fs", label, elapsed)
    return label, elapsed


JITTER_CEILING = 2.0            # p95 / median of a fixed probe
PROBE_REPS = 40
WAIT_TIMEOUT_S = 1800
WAIT_POLL_S = 30


def _scheduling_jitter() -> float:
    """Measure p95/median over a fixed single-threaded cryptographic probe.

    Load average includes threads in wait states and is not a direct measure
    of this process's scheduling interference. Repeated operation timings
    measure the spread relevant to wall-clock cryptographic benchmarks.
    """
    from ibsf import bls                     # local: keeps import cost off
    from ibsf.federation import Fleet        # the fast path

    fleet = Fleet.create(100, 999)
    roster = fleet.roster_pks
    t = fleet.threshold
    samples = []
    for i in range(PROBE_REPS):
        t0 = time.perf_counter()
        bls.key_list_digest(roster, 0, t)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= 5:                            # discard warm-up
            samples.append(dt)
    samples.sort()
    median = samples[len(samples) // 2]
    p95 = samples[int(len(samples) * 0.95) - 1]
    return p95 / median if median else float("inf")


def _await_stable_machine() -> tuple[list[float], float]:
    """Wait for scheduling jitter to meet the measurement threshold.

    Abort if the threshold cannot be met, rather than producing benchmark
    timings after the measurement-condition gate has failed.
    """
    jitter = _scheduling_jitter()
    if jitter <= JITTER_CEILING:
        return _loadavg(), jitter
    waited = 0
    logger.warning(
        "scheduling jitter p95/median = %.2f exceeds the %.1f ceiling for "
        "wall-clock crypto timing; waiting for the machine to settle",
        jitter, JITTER_CEILING,
    )
    while waited < WAIT_TIMEOUT_S:
        time.sleep(WAIT_POLL_S)
        waited += WAIT_POLL_S
        jitter = _scheduling_jitter()
        if jitter <= JITTER_CEILING:
            logger.info("jitter settled to %.2f after %ds", jitter, waited)
            return _loadavg(), jitter
    raise SystemExit(
        f"scheduling jitter is still {jitter:.2f} after {waited}s; refusing "
        "to produce timings the artifact itself declares unquotable. Re-run "
        "on a less contended machine, or raise JITTER_CEILING deliberately "
        "and say so in the paper."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manuscript", type=Path,
        help="optional TeX manuscript to check against regenerated reports",
    )
    args = parser.parse_args(argv)
    manuscript = None
    if args.manuscript is not None:
        manuscript = args.manuscript.expanduser().resolve()
        if not manuscript.is_file() or manuscript.suffix.lower() != ".tex":
            parser.error("--manuscript must name an existing .tex file")
    else:
        logger.info("Skipping manuscript number gate: no --manuscript supplied")
    steps = EXPERIMENTS + AGGREGATORS
    if manuscript is not None:
        steps += (("check_manuscript_numbers.py", str(manuscript)),)

    REPORTS.mkdir(exist_ok=True)
    load_before, jitter_before = _await_stable_machine()
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    for step in steps:
        label, elapsed = _run(step)
        timings[label] = round(elapsed, 1)
    total = time.perf_counter() - t0
    load_after = _loadavg()

    provenance = {
        "total_wall_clock_s": round(total, 1),
        "per_script_s": timings,
        "loadavg_before": load_before,
        "loadavg_after": load_after,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "single_pass": True,
        # Load average is recorded for transparency but is NOT the gate:
        # it counts waiting threads and so runs high on a desktop machine
        # with idle cores. The gate is scheduling jitter, which measures
        # the hazard that actually corrupts a wall-clock crypto timing.
        "scheduling_jitter_p95_over_median": round(jitter_before, 3),
        "jitter_ceiling": JITTER_CEILING,
        "jitter_gate_enforced": jitter_before <= JITTER_CEILING,
    }
    (REPORTS / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    logger.info(
        "full suite regenerated in %.1fs (load %s -> %s) -> reports/",
        total,
        load_before,
        load_after,
    )


if __name__ == "__main__":
    main()
