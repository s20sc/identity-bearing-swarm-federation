"""RQ1: Sybil admission under 4 attack scenarios + key-compromise bound.

Emits data/attacks.jsonl: 5 scenarios x 4 fleet sizes x 30 seeds.

Besides the accept/reject verdict, each record carries the scenario's
per-trial diagnostic fields. For the rogue-key scenario these state
whether the cancellation attack was actually realisable against a naive
aggregator and which of our two defences rejected it; for the
operator-key-compromise bound they state how many honest secret keys the
adversary used (it must be zero for the bound to mean anything).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibsf.attacks import SCENARIOS, run_scenario  # noqa: E402
from ibsf.federation import Fleet  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

FLEET_SIZES = (10, 50, 100, 500)
SEEDS = 30
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ALL_SCENARIOS = SCENARIOS + ("operator-key-compromise",)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "attacks.jsonl"
    t_start = time.perf_counter()
    records = 0
    with out.open("w") as fh:
        for n in FLEET_SIZES:
            for seed in range(SEEDS):
                fleet = Fleet.create(n, 1000 + seed)
                for scenario in ALL_SCENARIOS:
                    res = run_scenario(fleet, scenario, seed)
                    rec = {
                        "seed": seed,
                        "n": n,
                        "scenario": scenario,
                        "admitted": res.admitted,
                        "verify_time_ms": round(res.verify_ms, 4),
                    }
                    rec.update(res.detail)
                    fh.write(json.dumps(rec) + "\n")
                    records += 1
            logger.info("n=%d attacks done (%.1fs)", n, time.perf_counter() - t_start)
    logger.info("%d records -> %s", records, out.name)


if __name__ == "__main__":
    main()
