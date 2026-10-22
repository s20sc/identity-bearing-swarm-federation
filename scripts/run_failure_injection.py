"""Liveness under member failure: t-of-N sub-roster aggregation.

Crash a fraction c of roster members mid-epoch (they neither receive
the call-for-sign nor return a share). The decision aggregates over
the surviving participant set S, binds S's bitmap into aux, and
succeeds iff |S| >= t. The threshold is taken from `Fleet.threshold`
(t = ceil(0.8 N) over the operator-signed roster), the same definition
the main decision path uses, rather than being redefined here.
Also measures the commissioning proof-of-possession cost per robot.

Emits data/failure_injection.jsonl and data/pop_cost.jsonl.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blspy import G2Element  # noqa: E402

from ibsf import bls  # noqa: E402
from ibsf.federation import Fleet, canonical  # noqa: E402
from ibsf.network import NetworkModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

FLEET_SIZES = (100, 500)
CRASH_FRACTIONS = (0.0, 0.05, 0.1, 0.2, 0.3)
SEEDS = 30
DECISIONS_PER_SEED = 10
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def bitmap_over_roster(roster_indices: set[int], n: int) -> bytes:
    """Bitmap over the sorted signed roster (indices are roster positions)."""
    bits = bytearray((n + 7) // 8)
    for i in roster_indices:
        bits[i // 8] |= 1 << (i % 8)
    return bytes(bits)


def run_decision_with_crashes(
    fleet: Fleet, decision_payload: bytes, crashed: set[int],
    net: NetworkModel, rng: np.random.Generator
) -> dict:
    """One decision with `crashed` members silent.

    The quorum threshold is read from the fleet (`Fleet.threshold`,
    t = ceil(0.8 N) over the signed roster) rather than recomputed here,
    so this experiment and the main decision path cannot drift apart.
    """
    n = len(fleet.roster_pks)
    threshold = fleet.threshold
    epoch = fleet.epoch
    seq = fleet.next_seq()
    msg = canonical(epoch, seq, decision_payload)
    survivors = [r for r in fleet.robots if r.robot_id not in crashed]
    # Same timing model as the main decision path: each surviving member
    # contributes out-leg + signing + in-leg, and the operator proceeds on
    # the fastest t arrivals. When fewer than t survive, no quorum forms
    # and the operator waits out the survivors before refusing.
    m = max(len(survivors), 1)
    out_leg = net.one_way(rng, m)
    in_leg = net.one_way(rng, m)
    sigs_all: list[G2Element] = []
    sign_all: list[float] = []
    for robot in survivors:
        sig, ms = bls.member_sign(robot.sk, msg)
        sigs_all.append(sig)
        sign_all.append(ms)
    arrival = out_leg[: len(survivors)] + np.asarray(sign_all) + in_leg[: len(survivors)] \
        if survivors else np.asarray([0.0])
    order = list(np.argsort(arrival)) if survivors else []
    if len(survivors) >= threshold:
        chosen = [int(i) for i in order[:threshold]]
        quorum_wait = float(arrival[order[threshold - 1]])
    else:
        chosen = [int(i) for i in order]
        quorum_wait = float(arrival[order[-1]]) if survivors else 0.0
    participants = [survivors[i] for i in chosen]
    sigs = [sigs_all[i] for i in chosen] or sigs_all
    agg, agg_ms = bls.aggregate(sigs) if sigs else (None, 0.0)
    overhead = net.operator_overhead(rng)
    # bitmap is over the OPERATOR-SIGNED roster; participants map to their
    # roster positions so the verifier resolves keys from the roster only.
    pos = fleet.roster_positions()
    roster_idx = {pos[r.pk_bytes] for r in participants}
    pmap = bitmap_over_roster(roster_idx, n)
    sig_o, cs_ms = bls.countersign_subset(
        fleet.operator.sk_o, msg, agg, pmap, fleet.roster_pks, epoch, threshold
    )
    ok, verify_ms = bls.verify_subset(
        msg, agg, sig_o, fleet.roster_pks, pmap, fleet.operator.pk_o,
        threshold, epoch,
    )
    total = quorum_wait + agg_ms + overhead + cs_ms + verify_ms
    return {
        "succeeded": bool(ok),
        "survivors": len(survivors),
        "participants": len(participants),
        "threshold": threshold,
        "quorum_met": len(survivors) >= threshold,
        "total_ms": round(total, 4),
        "verify_ms": round(verify_ms, 4),
    }


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    net = NetworkModel()
    out = DATA_DIR / "failure_injection.jsonl"
    with out.open("w") as fh:
        for n in FLEET_SIZES:
            t_start = time.perf_counter()
            for seed in range(SEEDS):
                rng = np.random.default_rng(30_000 + seed)
                fleet = Fleet.create(n, 5000 + seed)
                for c in CRASH_FRACTIONS:
                    # Each member crashes INDEPENDENTLY with probability c,
                    # so the survivor count is Binomial(N, 1-c) rather than
                    # exactly N - round(cN). With a fixed crash count the
                    # success column is decided before the experiment runs:
                    # survivors >= t holds identically for c <= 1 - t/N and
                    # fails identically above it, so the reported 1.0/0.0
                    # rates would merely restate the threshold. Independent
                    # crashes put probability mass on both sides of t at
                    # the boundary, which makes availability there a
                    # measured quantity: at c = 0.2 with t = 0.8N the
                    # survivor mean sits exactly on the threshold and the
                    # success rate is genuinely uncertain per trial.
                    crashed = {
                        i for i in range(n) if rng.random() < c
                    }
                    for d in range(DECISIONS_PER_SEED):
                        msg = f"fi/{seed}/{c}/{d}".encode()
                        rec = run_decision_with_crashes(
                            fleet, msg, crashed, net, rng
                        )
                        rec.update({"seed": seed, "n": n, "crash_fraction": c})
                        fh.write(json.dumps(rec) + "\n")
            logger.info("n=%d failure injection done (%.1fs)", n, time.perf_counter() - t_start)

    # Proof-of-possession commissioning cost (per robot, real ops)
    pop_out = DATA_DIR / "pop_cost.jsonl"
    with pop_out.open("w") as fh:
        for i in range(200):
            sk = bls.keygen(f"pop/{i}".encode())
            proof, prove_ms = bls.pop_prove(sk)
            ok, verify_ms = bls.pop_verify(sk.get_g1(), proof)
            if not ok:
                raise AssertionError("PoP verification failed")
            fh.write(
                json.dumps(
                    {"i": i, "prove_ms": round(prove_ms, 4),
                     "verify_ms": round(verify_ms, 4)}
                )
                + "\n"
            )
    logger.info("PoP cost measured (200 robots) -> %s", pop_out.name)


if __name__ == "__main__":
    main()
