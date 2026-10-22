"""RQ3: forensic reconstruction under partial agent loss and corruption.

For each (n, f, seed): robots log events through one epoch, the operator
closes and signs the epoch, and then each robot is INDEPENDENTLY
corrupted with probability f. The number of affected robots is therefore
Binomial(n, f) rather than a fixed ceil(f*n): a fixed count would make
the recovered fraction exactly 1-f in every trial, which is arithmetic
restated as a measurement, not a measurement.

Each affected robot draws one of three corruption modes uniformly:

  * `total_loss` -- the whole chain is erased. The event bodies,
    signatures and hash pointers no longer exist anywhere in the process.
  * `tail_truncation` -- a uniformly random non-empty suffix of the chain
    is erased. What remains is a genesis-rooted, internally consistent,
    correctly signed chain: it is only short.
  * `interior_tamper` -- one uniformly chosen event's payload is mutated
    in place. The chain is full length and every byte but one is
    authentic.

The auditor is a separate function that receives only what survives:

  * the operator-signed epoch root (`SignedEpochRoot`),
  * the per-slot chain-head commitments recorded by the operator at
    epoch close (32 bytes per roster slot),
  * the roster public keys (which slots exist, and whose key each is),
  * whatever chain data still exists.

It does not receive the Merkle leaves. It rebuilds them as
`epoch_leaf(epoch, slot, member_pk, head)` from the heads and the roster
keys, then recomputes the tree and compares against the signed root, so
the leaf vector it works from is authenticated AND identity-bound: a
head that verifies at slot i also proves the slot-i member is the one
who produced it. Nothing about slot-to-robot attribution is taken on
trust from an out-of-band table.

It then ATTEMPTS, for every roster slot and every event index, to
produce and check a proof binding that event to the signed epoch root:
replay the hash pointers from the event to the chain head, check the
head against the slot's head commitment, and verify the Merkle path.
Per-event BLS signatures are verified against the roster key. Every
count below is the measured outcome of that loop; nothing is assigned
from the number of corrupted agents.

Two consequences are measured rather than engineered around.

1. Because the epoch leaf commits to the chain HEAD, a truncated or
   interior-tampered chain can no longer be replayed to the committed
   head, so NONE of its events stay provable even though almost all of
   its bytes survive and (for truncation) every remaining signature is
   still valid. Corruption is all-or-nothing per chain under head-only
   commitment. `events_present` and `events_signature_verified` are
   emitted alongside `events_provable` so the size of that gap is
   visible rather than implied. This is a limitation of the
   construction, and the trace is what shows it.

2. What must hold at 100% is DETECTION: no corrupted chain may be
   silently accepted. Each trial records, per mode, whether the auditor
   flagged the chain and why (absent, broken hash link, invalid
   signature, or head mismatch against the committed leaf), how many
   corrupted chains still contributed provable events
   (`silent_acceptances`, which must be 0), and how many intact chains
   were wrongly flagged (`intact_chains_flagged`, also 0). Detection is
   compared against the runner's ground-truth record of which slots it
   corrupted, so the rate can come out below 1.

Emits data/reconstruction.jsonl.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from blspy import G1Element

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibsf.audit_chain import AuditChain, MerkleTree, epoch_leaf  # noqa: E402
from ibsf.federation import Fleet  # noqa: E402
from ibsf.manifest import sha256  # noqa: E402
from ibsf.operator import SignedEpochRoot, verify_epoch_root  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

FLEET_SIZES = (10, 100, 500)
CORRUPTION_RATES = (0.1, 0.3, 0.5, 0.7)
MODES = ("total_loss", "tail_truncation", "interior_tamper")
SEEDS = 30
EVENTS_PER_ROBOT = 20
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# --------------------------------------------------------------------------
# corruption
# --------------------------------------------------------------------------

def apply_corruption(
    chain: AuditChain, mode: str, rng: np.random.Generator
) -> None:
    """Destructively corrupt one chain in place."""
    if not chain.events:
        return
    if mode == "total_loss":
        chain.events.clear()
    elif mode == "tail_truncation":
        # uniformly random NON-EMPTY suffix; may remove the whole chain
        k = int(rng.integers(1, len(chain.events) + 1))
        del chain.events[-k:]
    elif mode == "interior_tamper":
        i = int(rng.integers(len(chain.events)))
        ev = chain.events[i]
        chain.events[i] = replace(ev, payload=ev.payload + b"\x00")
    else:
        raise ValueError(f"unknown corruption mode {mode}")


# --------------------------------------------------------------------------
# auditor
# --------------------------------------------------------------------------

def rebuild_leaves(
    epoch: int,
    slots: list[int],
    heads: list[bytes],
    roster_pks: dict[int, G1Element],
) -> list[bytes]:
    """Identity-bound leaf vector, rebuilt from heads and roster keys.

    The auditor never receives leaves directly. Each leaf is recomputed
    as `epoch_leaf(epoch, slot, member_pk, head)`, so authenticating the
    vector against the signed root simultaneously authenticates which
    member each head belongs to.
    """
    return [
        epoch_leaf(epoch, pos, bytes(roster_pks[slot]), heads[pos])
        for pos, slot in enumerate(slots)
    ]


def _audit_slot(
    chain: AuditChain | None,
    head_commitment: bytes,
    pk: G1Element,
) -> dict:
    """Audit one slot's surviving chain against its head commitment.

    Returns the measured verdict: how many of the slot's events can be
    replayed to the committed head, how many carry a valid signature,
    and if the slot is not clean, the first reason the auditor can name.
    """
    if chain is None or not chain.events:
        return {
            "present": False,
            "events_present": 0,
            "provable": 0,
            "signature_verified": 0,
            "flagged": True,
            "reason": "absent",
        }

    events = chain.events
    n_present = len(events)

    # hash-pointer replay over the held events
    cur = events[0].prev_hash
    broken_at = None
    for ev in events:
        if ev.prev_hash != cur:
            broken_at = ev.seq
            break
        cur = ev.event_hash
    replayed_head = cur

    # per-event signatures against the slot's roster key
    sigs_ok = chain.verify_signatures(pk)
    n_sig = n_present if sigs_ok else 0

    if broken_at is not None:
        reason = "broken_link"
    elif not sigs_ok:
        reason = "bad_signature"
    elif replayed_head != head_commitment:
        reason = "head_mismatch"
    else:
        reason = None

    if reason is not None:
        return {
            "present": True,
            "events_present": n_present,
            "provable": 0,
            "signature_verified": n_sig,
            "flagged": True,
            "reason": reason,
        }

    # clean slot: count, per event index, whether the pointers from that
    # event replay to the committed head. This is a measured loop, not
    # len(events): a chain that failed any check above never reaches it.
    provable = 0
    for j in range(n_present):
        run = events[j].prev_hash
        intact = True
        for ev in events[j:]:
            if ev.prev_hash != run:
                intact = False
                break
            run = ev.event_hash
        if intact and run == head_commitment:
            provable += 1
    return {
        "present": True,
        "events_present": n_present,
        "provable": provable,
        "signature_verified": n_sig,
        "flagged": False,
        "reason": None,
    }


def audit_reconstruct(
    root: SignedEpochRoot,
    pk_o: G1Element,
    slots: list[int],
    heads: list[bytes],
    roster_pks: dict[int, G1Element],
    evidence: dict[int, AuditChain],
) -> dict:
    """Auditor: measure what is still provable, and what it can detect.

    Receives no fleet handle, no secret material, and no record of which
    slots were corrupted. Returns measured counts and the set of slots it
    flagged; the caller compares that set against ground truth.
    """
    if not verify_epoch_root(pk_o, root):
        return {
            "epoch_root_recoverable": False,
            "leaf_vector_authenticated": False,
            "events_present": 0,
            "events_provable": 0,
            "events_signature_verified": 0,
            "slots_unavailable": len(slots),
            "flagged": {s: "epoch_root_invalid" for s in slots},
            "provable_by_slot": {s: 0 for s in slots},
        }

    # rebuild the identity-bound leaves, then authenticate the whole
    # vector against the signed root
    leaves = rebuild_leaves(root.epoch, slots, heads, roster_pks)
    tree = MerkleTree(leaves)
    leaf_ok = tree.root == root.fleet_root

    present = provable = verified = unavailable = 0
    flagged: dict[int, str] = {}
    provable_by_slot: dict[int, int] = {}
    for pos, slot in enumerate(slots):
        res = _audit_slot(evidence.get(slot), heads[pos], roster_pks[slot])
        present += res["events_present"]
        verified += res["signature_verified"]
        if not res["present"]:
            unavailable += 1
        if res["flagged"]:
            flagged[slot] = res["reason"]
        # the Merkle path is what ties this slot's head to the signed
        # root; without an authenticated leaf vector nothing is provable
        credit = res["provable"] if leaf_ok else 0
        if credit and not MerkleTree.verify_proof(
            leaves[pos], tree.proof(pos), root.fleet_root
        ):
            credit = 0
        provable += credit
        provable_by_slot[slot] = credit
    return {
        "epoch_root_recoverable": True,
        "leaf_vector_authenticated": bool(leaf_ok),
        "events_present": present,
        "events_provable": provable,
        "events_signature_verified": verified,
        "slots_unavailable": unavailable,
        "flagged": flagged,
        "provable_by_slot": provable_by_slot,
    }


def negative_control(
    root: SignedEpochRoot,
    pk_o: G1Element,
    slots: list[int],
    heads: list[bytes],
    roster_pks: dict[int, G1Element],
    evidence: dict[int, AuditChain],
    baseline: dict,
    corrupted: dict[int, str],
) -> bool:
    """Falsification check: does tampering actually reduce the counts?

    A recoverability measurement is only meaningful if it can come out
    lower. One UNCORRUPTED surviving chain has an event body mutated; the
    auditor is re-run and must lose exactly that chain's events from BOTH
    the provable and the signature-verified counts, and must flag the
    slot. The mutation is reverted afterwards, so the recorded
    measurement is unaffected.
    """
    victim = next(
        (
            s for s in slots
            if s not in corrupted and evidence.get(s) and evidence[s].events
        ),
        None,
    )
    if victim is None:
        return True
    chain = evidence[victim]
    original = list(chain.events)
    k = len(original)
    ev0 = original[0]
    chain.events[0] = replace(ev0, payload=ev0.payload + b"\x00")
    try:
        after = audit_reconstruct(
            root, pk_o, slots, heads, roster_pks, evidence
        )
    finally:
        chain.events[:] = original
    return (
        after["events_provable"] == baseline["events_provable"] - k
        and after["events_signature_verified"]
        == baseline["events_signature_verified"] - k
        and victim in after["flagged"]
    )


def timed_event_proof(
    slots: list[int],
    heads: list[bytes],
    roster_pks: dict[int, G1Element],
    root: SignedEpochRoot,
    chain: AuditChain,
    pos: int,
    event_seq: int,
) -> tuple[float, float]:
    """Produce + verify one event -> chain head -> fleet root proof.

    Verification recomputes the slot's identity-bound leaf from the
    replayed head and the slot member's roster key, so the timing covers
    the same work an external auditor does.
    """
    member_pk = bytes(roster_pks[slots[pos]])
    t0 = time.perf_counter()
    cur = chain.events[event_seq].prev_hash
    for ev in chain.events[event_seq:]:
        if ev.prev_hash != cur:
            raise AssertionError("chain link broken")
        cur = ev.event_hash
    head = cur
    leaves = rebuild_leaves(root.epoch, slots, heads, roster_pks)
    tree = MerkleTree(leaves)
    path = tree.proof(pos)
    produce_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    leaf = epoch_leaf(root.epoch, pos, member_pk, head)
    ok = MerkleTree.verify_proof(leaf, path, root.fleet_root)
    verify_ms = (time.perf_counter() - t1) * 1000.0
    if not ok:
        raise AssertionError("proof verification failed")
    return produce_ms, verify_ms


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "reconstruction.jsonl"
    with out.open("w") as fh:
        for n in FLEET_SIZES:
            t_start = time.perf_counter()
            for seed in range(SEEDS):
                rng = np.random.default_rng(seed)
                fleet = Fleet.create(n, 2000 + seed)
                for robot in fleet.robots:
                    for k in range(EVENTS_PER_ROBOT):
                        robot.chain.append(
                            "capability-invocation",
                            sha256(f"{robot.robot_id}/{k}".encode()),
                        )
                chains = {r.robot_id: r.chain for r in fleet.robots}
                root, closed, _ = fleet.operator.close_and_sign_epoch(
                    fleet.epoch, chains, 0.0, 60.0, fleet.member_pks
                )
                # the auditor's key material comes from the OPERATOR-SIGNED
                # roster, not from the robot objects: slot i of the leaf
                # vector is roster position i, so the roster alone tells
                # the auditor whose head sits in each slot.
                signed_roster = fleet.operator.rosters[fleet.epoch]
                roster_pks = {
                    slot: G1Element.from_bytes(signed_roster.member_pks[i])
                    for i, slot in enumerate(closed.slots)
                }
                # operator's epoch-close record: ordered slots + the
                # per-slot head commitments. The leaves themselves are
                # NOT handed to the auditor; it rebuilds them from these
                # heads and the roster keys.
                slots = closed.slots
                heads = [closed.heads[s] for s in slots]
                events_total = n * EVENTS_PER_ROBOT
                pristine = {s: list(chains[s].events) for s in slots}

                for f in CORRUPTION_RATES:
                    # each f is an independent draw on the pristine fleet,
                    # so the corruption rate is not applied cumulatively
                    for s in slots:
                        chains[s].events[:] = pristine[s]

                    hit = rng.random(n) < f
                    picks = rng.integers(0, len(MODES), size=n)
                    corrupted: dict[int, str] = {}
                    for i, s in enumerate(slots):
                        if hit[i]:
                            mode = MODES[int(picks[i])]
                            corrupted[s] = mode
                            apply_corruption(chains[s], mode, rng)

                    res = audit_reconstruct(
                        root, fleet.operator.pk_o, slots, heads,
                        roster_pks, chains,
                    )
                    flagged = res["flagged"]

                    # detection accounting, per injected mode
                    by_mode = {m: 0 for m in MODES}
                    detected_by_mode = {m: 0 for m in MODES}
                    absent_by_mode = {m: 0 for m in MODES}
                    present_unverifiable_by_mode = {m: 0 for m in MODES}
                    reasons: dict[str, int] = {}
                    silent = 0
                    for s, mode in corrupted.items():
                        by_mode[mode] += 1
                        reason = flagged.get(s)
                        if reason is not None:
                            detected_by_mode[mode] += 1
                            reasons[reason] = reasons.get(reason, 0) + 1
                            if reason == "absent":
                                absent_by_mode[mode] += 1
                            else:
                                present_unverifiable_by_mode[mode] += 1
                        if res["provable_by_slot"].get(s, 0) > 0:
                            silent += 1
                    false_flags = sum(
                        1 for s in flagged if s not in corrupted
                    )
                    n_corrupt = len(corrupted)
                    detected = sum(detected_by_mode.values())

                    # falsification check (seed 0 only: it doubles the
                    # audit cost and the outcome does not vary by seed)
                    control = (
                        negative_control(
                            root, fleet.operator.pk_o, slots, heads,
                            roster_pks, chains, res, corrupted,
                        )
                        if seed == 0
                        else None
                    )
                    # timing: one proof from a slot that is still clean
                    clean = [
                        p for p, s in enumerate(slots)
                        if s not in corrupted and chains[s].events
                    ]
                    if clean:
                        pos = clean[int(rng.integers(len(clean)))]
                        produce_ms, verify_ms = timed_event_proof(
                            slots, heads, roster_pks, root,
                            chains[slots[pos]], pos,
                            int(rng.integers(EVENTS_PER_ROBOT)),
                        )
                    else:
                        produce_ms = verify_ms = float("nan")

                    fh.write(
                        json.dumps(
                            {
                                "seed": seed,
                                "n": n,
                                "f": f,
                                "robots_corrupted": n_corrupt,
                                "corrupted_by_mode": by_mode,
                                "detected_by_mode": detected_by_mode,
                                "absent_by_mode": absent_by_mode,
                                "present_unverifiable_by_mode":
                                    present_unverifiable_by_mode,
                                "detection_reasons": reasons,
                                "chains_detected": detected,
                                "detection_rate": (
                                    round(detected / n_corrupt, 6)
                                    if n_corrupt else None
                                ),
                                "silent_acceptances": silent,
                                "intact_chains_flagged": false_flags,
                                "events_total": events_total,
                                "events_present": res["events_present"],
                                "epoch_root_recoverable": res[
                                    "epoch_root_recoverable"
                                ],
                                "leaf_vector_authenticated": res[
                                    "leaf_vector_authenticated"
                                ],
                                "events_provable": res["events_provable"],
                                "events_signature_verified": res[
                                    "events_signature_verified"
                                ],
                                "slots_unavailable": res["slots_unavailable"],
                                "negative_control_detects_tamper": control,
                                "events_recoverable": round(
                                    res["events_provable"] / events_total, 6
                                ),
                                "events_present_fraction": round(
                                    res["events_present"] / events_total, 6
                                ),
                                "proof_production_time_ms": round(
                                    produce_ms, 4
                                ),
                                "proof_verify_time_ms": round(verify_ms, 4),
                            }
                        )
                        + "\n"
                    )
            logger.info(
                "n=%d reconstruction done (%.1fs)",
                n, time.perf_counter() - t_start,
            )


if __name__ == "__main__":
    main()
