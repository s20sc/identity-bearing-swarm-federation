"""Epoch-length sensitivity (Section V-F promise).

Over a 1-hour horizon with Poisson admission/revocation requests
(rate 4/h each), epoch length L in {10, 30, 60, 300} s trades
revocation time-to-effect against per-hour certificate growth and
operator signing load. Emits data/epoch_sensitivity.jsonl.
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
from ibsf.certificate import export_certificate  # noqa: E402
from ibsf.federation import Fleet  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

EPOCH_LENGTHS_S = (10, 30, 60, 300)
HORIZON_S = 3600
EVENT_RATE_PER_H = 4.0  # revocation requests per hour
SEEDS = 30
N = 100
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "epoch_sensitivity.jsonl"
    with out.open("w") as fh:
        for length in EPOCH_LENGTHS_S:
            n_epochs = HORIZON_S // length
            t_start = time.perf_counter()
            # one real fleet: measure actual per-epoch close+sign cost
            fleet = Fleet.create(N, 4000)
            chains = {r.robot_id: r.chain for r in fleet.robots}
            for r in fleet.robots:
                r.chain.append("capability-invocation", b"\x01" * 32)
            members = [
                (r.pk_bytes, bytes(bls.pop_prove(r.sk)[0])) for r in fleet.robots
            ]
            caps = {r.pk_bytes: r.manifest.cap for r in fleet.robots}
            castes = {r.pk_bytes: r.caste for r in fleet.robots}
            # robot id -> roster key: epoch leaves are identity-bound, and
            # membership is static here so the map is computed once
            member_pks = fleet.member_pks
            sign_ms_total = 0.0
            for e in range(n_epochs):
                if e > 0:
                    # Re-sign with the SAME capability and caste binding the fleet
                    # was commissioned under. Omitting either left epochs 1..k-1
                    # with weaker rosters than epoch 0, so the sweep silently
                    # measured a policy that CHANGED at epoch 1, not the static
                    # one it reports.
                    fleet.operator.sign_roster(
                        e, members, t_open=e * length, capabilities=caps, castes=castes
                    )
                _, _, sign_ms = fleet.operator.close_and_sign_epoch(
                    e, chains, e * length, (e + 1) * length, member_pks
                )
                sign_ms_total += sign_ms
            cert = export_certificate(fleet.operator, list(range(n_epochs)))
            cert_bytes_per_h = len(cert.to_json(embed_rosters=False))

            for seed in range(SEEDS):
                rng = np.random.default_rng(20_000 + seed)
                n_req = rng.poisson(EVENT_RATE_PER_H * HORIZON_S / 3600)
                req_times = rng.uniform(0, HORIZON_S, size=max(n_req, 1))
                # Time-to-effect is ANALYTIC, not measured. A revocation
                # requested at time t takes effect at the next epoch
                # boundary, so tte = ceil(t/L)*L - t, which for uniform
                # arrivals is exactly Uniform(0, L): population median
                # L/2, population p95 0.95L. No protocol code runs here
                # and none needs to; the quantity is a property of the
                # epoch discipline, and we report it as such rather than
                # dressing a closed form as an experiment. What IS
                # measured in this sweep is the certificate growth and
                # the operator's epoch-close signing load per hour, both
                # of which come from real epoch closes above.
                tte = [
                    (np.ceil(t / length) * length) - t for t in req_times
                ]
                fh.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "epoch_length_s": length,
                            "n": N,
                            # per-seed request times are emitted so the
                            # aggregate percentile is computed over the
                            # POOLED arrival process rather than as a
                            # median of per-seed percentiles taken over
                            # ~4 samples each, which is badly biased and
                            # disagreed with the closed form it samples
                            "revocation_tte_samples_s": [
                                round(float(x), 4) for x in tte
                            ],
                            "revocation_tte_median_s": round(
                                float(np.median(tte)), 3
                            ),
                            "revocation_tte_analytic_median_s": length / 2.0,
                            "revocation_tte_analytic_p95_s": 0.95 * length,
                            "epochs_per_hour": int(n_epochs),
                            "cert_bytes_per_hour": cert_bytes_per_h,
                            "operator_sign_ms_per_hour": round(sign_ms_total, 3),
                        }
                    )
                    + "\n"
                )
            logger.info(
                "L=%ds done (%.1fs)", length, time.perf_counter() - t_start
            )


if __name__ == "__main__":
    main()
