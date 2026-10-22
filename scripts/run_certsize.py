"""RQ4: compliance-certificate compactness vs fleet size and epochs.

Emits data/certsize.jsonl with root-level bytes (epoch entries only)
and full-document bytes (deduplicated roster attachment included).
Also validates regulator-side offline verification on each export.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibsf import bls  # noqa: E402
from ibsf.certificate import export_certificate, regulator_verify  # noqa: E402
from ibsf.federation import Fleet  # noqa: E402
from ibsf.manifest import sha256  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

FLEET_SIZES = (10, 100, 500)
EPOCH_COUNTS = (1, 10, 100, 1000)
EVENTS_PER_ROBOT = 5
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "certsize.jsonl"
    with out.open("w") as fh:
        for n in FLEET_SIZES:
            t_start = time.perf_counter()
            fleet = Fleet.create(n, 3000)
            chains = {r.robot_id: r.chain for r in fleet.robots}
            # robot id -> roster key; membership is static across the
            # sweep, so the epoch-leaf identity binding is the same map
            # every epoch and is computed once
            member_pks = fleet.member_pks
            # static policy: precompute each member's PoP once, reuse
            # across epochs (sign_roster re-verifies every epoch).
            members = [
                (r.pk_bytes, bytes(bls.pop_prove(r.sk)[0])) for r in fleet.robots
            ]
            caps = {r.pk_bytes: r.manifest.cap for r in fleet.robots}
            castes = {r.pk_bytes: r.caste for r in fleet.robots}
            # Chain contents are written once, before the epoch sweep, and
            # not extended per epoch. Certificate size is a function of
            # (N, k) alone -- every field in the emitted record is either a
            # byte count over epoch entries and rosters or the regulator
            # verify time, and none of them reads an audit chain. Appending
            # per epoch would add ~3M member-chain signature verifications
            # at epoch close (the admission gate in
            # `close_and_sign_epoch`) and dominate this sweep's runtime
            # without moving any reported number. `run_reconstruction.py`
            # is where per-epoch event volume is actually exercised.
            for robot in fleet.robots:
                for j in range(EVENTS_PER_ROBOT):
                    robot.chain.append(
                        "capability-invocation",
                        sha256(f"{robot.robot_id}/{j}".encode()),
                    )
            # Exports happen at CHECKPOINTS during the sweep, when exactly
            # k epochs have been closed, not retroactively over the first
            # k of 1000. A compliance certificate is the full record up to
            # the operator's attested latest epoch: the coverage check in
            # `regulator_verify` rejects a document that stops short of
            # that attestation, which is the whole anti-truncation
            # guarantee. Exporting range(k) of an operator that had closed
            # 1000 epochs is exactly the truncation the verifier exists to
            # refuse, so each measured document is instead issued at the
            # moment its k is the complete history.
            k_max = max(EPOCH_COUNTS)
            for e in range(k_max):
                if e > 0:
                    # Re-sign with the SAME capability and caste binding the fleet
                    # was commissioned under. Omitting either left epochs 1..k-1
                    # with weaker rosters than epoch 0, so the sweep silently
                    # measured a policy that CHANGED at epoch 1, not the static
                    # one it reports.
                    fleet.operator.sign_roster(
                        e, members, t_open=60.0 * e, capabilities=caps, castes=castes
                    )
                fleet.operator.close_and_sign_epoch(
                    e, chains, 60.0 * e, 60.0 * (e + 1), member_pks
                )
                k = e + 1
                if k not in EPOCH_COUNTS:
                    continue
                cert = export_certificate(
                    fleet.operator, list(range(k)), issued_at=60.0 * k
                )
                t0 = time.perf_counter()
                ok = regulator_verify(cert)
                verify_ms = (time.perf_counter() - t0) * 1000.0
                if not ok:
                    raise AssertionError(f"regulator verify failed n={n} k={k}")
                root_bytes = len(cert.to_json(embed_rosters=False))
                full_bytes = len(cert.to_json(embed_rosters=True))
                fh.write(
                    json.dumps(
                        {
                            "n": n,
                            "k": k,
                            "root_level_bytes": root_bytes,
                            "full_doc_bytes": full_bytes,
                            "regulator_verify_ms": round(verify_ms, 3),
                        }
                    )
                    + "\n"
                )
            logger.info("n=%d certsize done (%.1fs)", n, time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
