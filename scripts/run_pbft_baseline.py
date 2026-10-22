"""Baseline: PBFT consensus latency over the same network substrate.

Emits data/pbft.jsonl. Both protocols are exercised over the same
network distribution, the same fleet sizes, and the same 30-seed set,
but the runs are NOT paired and must not be read as if they were. PBFT
consumes O(N^2) delay draws per decision against the federation's O(N),
so a shared seed produces two streams that diverge on the first
decision: the two protocols never see the same realised link delays.
The comparison is therefore between distributions, and the intervals
are computed independently per protocol rather than on differences.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibsf.network import NetworkModel  # noqa: E402
from ibsf.pbft import run_pbft_decision  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

FLEET_SIZES = (10, 50, 100, 500)
SEEDS = 30
DECISIONS_PER_SEED = 20
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "pbft.jsonl"
    net = NetworkModel()
    t_start = time.perf_counter()
    with out.open("w") as fh:
        for n in FLEET_SIZES:
            for seed in range(SEEDS):
                # same seed set as run_sim.py, but not a paired design:
                # the draw counts differ, so the streams diverge at once
                rng = np.random.default_rng(seed)
                for d in range(DECISIONS_PER_SEED):
                    res = run_pbft_decision(n, net, rng, seed * 100 + d)
                    fh.write(
                        json.dumps(
                            {
                                "seed": seed,
                                "n": n,
                                "decision": d,
                                "total_ms": round(res.total_ms, 4),
                                "transport_ms": round(res.transport_ms, 4),
                                "mac_ms": round(res.mac_ms, 4),
                                "messages": res.messages,
                            }
                        )
                        + "\n"
                    )
            logger.info("pbft n=%d done (%.1fs)", n, time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
