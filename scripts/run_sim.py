"""RQ2: end-to-end decision latency at N in {10,50,100,500}.

Emits data/sim_n{N}.jsonl, one line per (seed, decision) trial.
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

FLEET_SIZES = (10, 50, 100, 500)
SEEDS = 30
DECISIONS_PER_SEED = 20
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    net = NetworkModel()
    for n in FLEET_SIZES:
        out = DATA_DIR / f"sim_n{n}.jsonl"
        t_start = time.perf_counter()
        with out.open("w") as fh:
            for seed in range(SEEDS):
                rng = np.random.default_rng(seed)
                fleet = Fleet.create(n, seed)
                for d in range(DECISIONS_PER_SEED):
                    msg = f"fleet-decision/{seed}/{d}".encode()
                    res = run_decision(fleet, msg, net, rng)
                    if not res.verified:
                        raise AssertionError(
                            f"honest decision failed verify at n={n} seed={seed}"
                        )
                    fh.write(
                        json.dumps(
                            {
                                "seed": seed,
                                "n": n,
                                "decision": d,
                                "total_ms": round(res.total_ms, 4),
                                "transport_ms": round(res.transport_ms, 4),
                                "member_sign_ms": round(res.member_sign_ms, 4),
                                "aggregate_ms": round(res.aggregate_ms, 4),
                                "operator_overhead_ms": round(
                                    res.operator_overhead_ms, 4
                                ),
                                "countersign_ms": round(res.countersign_ms, 4),
                                "verify_ms": round(res.verify_ms, 4),
                                "messages": res.messages,
                                "participants": res.participants,
                                "threshold": res.threshold,
                            }
                        )
                        + "\n"
                    )
        logger.info(
            "n=%d done: %d trials in %.1fs -> %s",
            n,
            SEEDS * DECISIONS_PER_SEED,
            time.perf_counter() - t_start,
            out.name,
        )


if __name__ == "__main__":
    main()
