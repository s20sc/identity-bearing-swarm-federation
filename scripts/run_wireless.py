"""Wireless-tail transport sensitivity (Discussion / Setup robustness).

Re-runs the latency experiment at N in {100, 500} under a declared
enterprise-Wi-Fi-like transport model (median 2.0 ms, p99 45 ms,
heavier log-normal tail) with the same real crypto. Emits
data/wireless.jsonl.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibsf.federation import Fleet, run_decision  # noqa: E402
from ibsf.network import NetworkModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

FLEET_SIZES = (100, 500)
SEEDS = 30
DECISIONS_PER_SEED = 20
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    net = NetworkModel(median_ms=2.0, p99_ms=45.0)
    out = DATA_DIR / "wireless.jsonl"
    with out.open("w") as fh:
        for n in FLEET_SIZES:
            t_start = time.perf_counter()
            for seed in range(SEEDS):
                rng = np.random.default_rng(40_000 + seed)
                fleet = Fleet.create(n, 6000 + seed)
                for d in range(DECISIONS_PER_SEED):
                    msg = f"wifi/{seed}/{d}".encode()
                    res = run_decision(fleet, msg, net, rng)
                    if not res.verified:
                        raise AssertionError("honest decision failed verify")
                    fh.write(
                        json.dumps(
                            {
                                "seed": seed,
                                "n": n,
                                "decision": d,
                                "total_ms": round(res.total_ms, 4),
                                "transport_ms": round(res.transport_ms, 4),
                            }
                        )
                        + "\n"
                    )
            logger.info("wifi n=%d done (%.1fs)", n, time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
