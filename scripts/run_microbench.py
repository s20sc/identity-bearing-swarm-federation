"""Measure micro-costs used in the latency decomposition.

Emit timings for key-list digest recomputation and distinct-participant
counting, two operations that a deployment can amortise. Store the
measurements in data/microbench.jsonl for report regeneration.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibsf import bls  # noqa: E402
from ibsf.federation import Fleet  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

FLEET_SIZES = (100, 500)
REPS = 200
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _time(fn, reps: int) -> list[float]:
    """Per-call wall-clock in ms, discarding a warm-up call."""
    fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "microbench.jsonl"
    with out.open("w") as fh:
        for n in FLEET_SIZES:
            fleet = Fleet.create(n, 7000 + n)
            roster = fleet.roster_pks
            t = fleet.threshold
            subset = roster[:t]

            digest = _time(
                lambda: bls.key_list_digest(roster, 0, t), REPS
            )
            distinct = _time(lambda: len({bytes(pk) for pk in subset}), REPS)

            for label, samples, detail in (
                ("key_list_digest", digest, {"keys": n}),
                ("distinct_key_count", distinct, {"keys": t}),
            ):
                rec = {
                    "n": n,
                    "operation": label,
                    "median_ms": round(float(np.median(samples)), 4),
                    "p95_ms": round(float(np.percentile(samples, 95)), 4),
                    "reps": REPS,
                }
                rec.update(detail)
                fh.write(json.dumps(rec) + "\n")
            logger.info("n=%d micro-benchmarks done", n)
    logger.info("micro-benchmarks -> %s", out.name)


if __name__ == "__main__":
    main()
