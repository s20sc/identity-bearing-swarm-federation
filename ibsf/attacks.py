"""Scripted Sybil attack scenarios (Section VIII-A).

Every scenario runs the REAL verify pipeline of Algorithm 1 with real
BLS material and reports `admitted` = the pipeline's actual accept
verdict. Verification is always roster-resolved: the participating set
is a bitmap over the operator-signed roster (`verify_subset`), so an
attacker cannot substitute caller-supplied keys it controls.

Two scenarios deserve a note on threat model, because a rigged
adversary would make the result vacuous:

  * `rogue_subset_addition` runs the TEXTBOOK cancellation attack, not a
    weakened stand-in. The adversary chooses its own secret x and
    publishes pk' = g^x * (prod_{i in S} pk_i)^{-1} over a set S of
    honest roster keys. The aggregate key over S u {pk'} then collapses
    to g^x, so the adversary ALONE forges a signature that satisfies a
    naive `fast_aggregate_verify` over that key set. The scenario
    measures both halves: that the attack really works against an
    aggregator without proof-of-possession (`naive_aggregate_admits`),
    and that our construction still rejects it (`admitted`).

  * `operator_key_compromise` gives the adversary sk_O and NOTHING ELSE.
    It mints its own roster out of keys it generated itself (whose PoPs
    it can therefore produce) and drives a decision under that roster.
    It never borrows an honest robot's secret, so admission=1.0 measures
    the real theoretical bound (Sybil resistance reduces to operator-key
    custody) rather than an artefact of handing the adversary the fleet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from blspy import G1Element, G2Element, PopSchemeMPL

from . import bls
from .federation import (
    Fleet,
    canonical,
    submit_decision_tuple,
)
from .manifest import IdentityManifest
from .operator import Operator, verify_manifest

SCENARIOS = (
    "single-impersonation",
    "rogue-subset-addition",
    "caste-forgery",
    "caste-laundering",
    "operator-signature-replay",
    "malformed-bitmap",
)


@dataclass(frozen=True)
class AttackResult:
    scenario: str
    admitted: bool
    verify_ms: float
    detail: dict = field(default_factory=dict)


def _full_bitmap(n: int) -> bytes:
    bits = bytearray((n + 7) // 8)
    for i in range(n):
        bits[i // 8] |= 1 << (i % 8)
    return bytes(bits)


def _bitmap_from(indices: list[int], n: int) -> bytes:
    bits = bytearray((n + 7) // 8)
    for i in indices:
        bits[i // 8] |= 1 << (i % 8)
    return bytes(bits)


def single_impersonation(fleet: Fleet, seed: int) -> AttackResult:
    """Fresh pk' outside the roster tries to pass as a fleet member.

    Two admission surfaces are exercised, because the Sybil-resistance
    property claims both. First, member-side admission: the adversary
    builds a well-formed identity manifest for its own key, claiming the
    same operator root and a capability set of its choosing, and presents
    it for verification. Second, the decision path: it contributes a
    share to the aggregate, and the verifier, which resolves keys from
    the signed roster only, finds the pairing broken.

    A manifest is cheap to fabricate, so the first check is the one that
    carries the property: what an unlisted key cannot do is appear in a
    roster the operator signed.
    """
    epoch = fleet.epoch
    seq = fleet.next_seq()
    msg = canonical(epoch, seq, f"decision/{seed}".encode())
    adv_sk = bls.keygen(f"adversary/{seed}".encode())

    # (1) member-side admission of a fabricated manifest for pk'
    adv_manifest = IdentityManifest(
        pk=bytes(adv_sk.get_g1()),
        root_o=fleet.operator.root_cert_digest,   # claims our operator
        cap=("navigate", "report", "supervise"),  # claims what it likes
        audit_head=b"\x00" * 32,
    )
    manifest_admitted = verify_manifest(
        fleet.operator.pk_o,
        fleet.operator.rosters[epoch],
        adv_manifest,
        fleet.operator.root_cert_digest,
        expected_epoch=epoch,
    )

    # (2) decision-path admission of an injected signature share
    sigs = [PopSchemeMPL.sign(r.sk, msg) for r in fleet.robots]
    sigs.append(PopSchemeMPL.sign(adv_sk, msg))  # injected share
    agg, _ = bls.aggregate(sigs)
    n = len(fleet.roster_pks)
    bitmap = _full_bitmap(n)
    sig_o, _ = bls.countersign_subset(
        fleet.operator.sk_o, msg, agg, bitmap, fleet.roster_pks, epoch, fleet.threshold
    )
    ok, ms = bls.verify_subset(
        msg, agg, sig_o, fleet.roster_pks, bitmap,
        fleet.operator.pk_o, fleet.threshold, epoch,
    )
    return AttackResult(
        "single-impersonation",
        bool(ok or manifest_admitted),
        ms,
        {"manifest_admitted": bool(manifest_admitted),
         "decision_path_admitted": bool(ok)},
    )


def rogue_subset_addition(fleet: Fleet, seed: int, k: int = 3) -> AttackResult:
    """Textbook rogue-key CANCELLATION attack against a k-member subset.

    Construction (Boneh-Drijvers-Neven rogue-key attack). Let S be the
    first k keys of the operator-signed roster, all of them public. The
    adversary draws its own secret x and publishes

        pk' = g^x * (prod_{i in S} pk_i)^{-1}

    so that the aggregate key over S u {pk'} collapses to exactly g^x.
    The adversary can therefore produce, on its own and with no honest
    participation whatsoever, a single signature sigma = H(m)^x that
    satisfies the aggregate pairing check over the key set S u {pk'}.

    The scenario measures three things.

    (a) `naive_aggregate_admits` -- whether the forgery is accepted by a
        bare `fast_aggregate_verify` over the attacker-chosen key set.
        This is the check an aggregator without proof-of-possession
        performs, and the attack is expected to SUCCEED against it. The
        experiment is worthless unless this is genuinely realisable, so
        it is measured and reported rather than asserted.

    (b) `pop_admits_rogue_key` -- whether pk' can be commissioned. The
        adversary knows x but not the discrete log of pk', so it cannot
        produce a PoP for pk'; every constructible forgery attempt is
        tried and `Operator.sign_roster` is invoked for real.

    (c) `admitted` -- the verdict of our actual verify path. Two
        sub-checks are reported separately so the rejection cause is not
        ambiguous. Presenting the honest roster fails the member check
        (the resolved key set is S, whose aggregate is not g^x).
        Presenting a roster substituted to contain pk' passes the member
        check -- the cancellation really does work on the resolved key
        set -- and is stopped solely by the key-list digest bound into the
        operator countersignature.

    The adversary is handed an operator countersignature over its own
    forged aggregate (an assumption far stronger than the threat model
    allows, since it does not hold sk_O) precisely so that the roster
    binding is tested in isolation rather than masked by the operator
    signature failing for an unrelated reason.
    """
    epoch = fleet.epoch
    seq = fleet.next_seq()
    msg = canonical(epoch, seq, f"decision/{seed}".encode())
    roster = fleet.operator.rosters[epoch]

    # --- adversary side: only PUBLIC honest keys are used ---
    target_idx = list(range(min(k, len(roster.member_pks))))
    target_pks = [G1Element.from_bytes(roster.member_pks[i]) for i in target_idx]
    adv_sk = bls.keygen(f"rogue-cancel/{seed}".encode())
    adv_pk = adv_sk.get_g1()

    prod = target_pks[0]
    for pk in target_pks[1:]:
        prod = prod + pk
    rogue_pk = adv_pk + prod.negate()      # pk' = g^x * (prod pk_i)^{-1}

    # the aggregate key over S u {pk'} must collapse to the adversary's key
    collapsed = rogue_pk
    for pk in target_pks:
        collapsed = collapsed + pk
    cancellation_exact = bytes(collapsed) == bytes(adv_pk)

    # the adversary's SOLO signature -- no honest robot signed anything
    solo = PopSchemeMPL.sign(adv_sk, msg)

    # (a) does the attack work against an aggregator with no PoP?
    naive_admits = bool(
        PopSchemeMPL.fast_aggregate_verify(target_pks + [rogue_pk], msg, solo)
    )

    # (b) can pk' be commissioned? try every forgery the adversary can build
    sk_by_pk = {r.pk_bytes: r.sk for r in fleet.robots}
    honest_pops = [
        bls.pop_prove(sk_by_pk[roster.member_pks[i]])[0] for i in target_idx
    ]
    forged_pops = {
        "adversary_own_pop": bls.pop_prove(adv_sk)[0],
        "honest_pop_replay": honest_pops[0],
        "aggregated_honest_pops": PopSchemeMPL.aggregate(honest_pops),
        "cancellation_style_pop": PopSchemeMPL.aggregate(
            [bls.pop_prove(adv_sk)[0], PopSchemeMPL.aggregate(honest_pops).negate()]
        ),
    }
    pop_admits = False
    commissioned = False
    for proof in forged_pops.values():
        ok_pop, _ = bls.pop_verify(rogue_pk, proof)
        pop_admits = pop_admits or bool(ok_pop)
    # exercise the real commissioning gate with the best forgery available
    probe_op = Operator.create(f"rogue-probe/{seed}".encode())
    try:
        probe_op.sign_roster(
            epoch,
            [(bytes(rogue_pk), bytes(forged_pops["adversary_own_pop"]))],
            t_open=0.0,
        )
        commissioned = True
    except ValueError:
        commissioned = False

    # (c) our pipeline. Generous assumption: the adversary is GIVEN an
    # operator countersignature over its forged aggregate and the honest
    # roster, so the roster binding is what has to do the work.
    n = len(fleet.roster_pks)
    honest_bitmap = _bitmap_from(target_idx, n)
    sig_o, _ = bls.countersign_subset(
        fleet.operator.sk_o, msg, solo, honest_bitmap, fleet.roster_pks, epoch, len(target_idx)
    )
    ok_honest_roster, ms = bls.verify_subset(
        msg, solo, sig_o, fleet.roster_pks, honest_bitmap,
        fleet.operator.pk_o, len(target_idx), epoch,
    )

    # substituted roster: honest roster augmented with the rogue key
    forged_roster = fleet.roster_pks + [rogue_pk]
    forged_bitmap = _bitmap_from(
        target_idx + [len(forged_roster) - 1], len(forged_roster)
    )
    ok_forged_roster, ms2 = bls.verify_subset(
        msg, solo, sig_o, forged_roster, forged_bitmap,
        fleet.operator.pk_o, len(target_idx) + 1, epoch,
    )
    # decompose the substituted-roster verdict: which half rejected it?
    resolved = [forged_roster[i] for i in target_idx + [len(forged_roster) - 1]]
    member_check = bool(
        PopSchemeMPL.fast_aggregate_verify(resolved, msg, solo)
    )
    # Same threshold the countersignature above was formed under, so that
    # the substituted roster is the only thing that differs and the
    # decomposition attributes the rejection to the roster rather than to
    # a threshold mismatch.
    aux = bls.aux_binding(
        solo, forged_bitmap, forged_roster, epoch, len(target_idx)
    )
    operator_check = bool(
        PopSchemeMPL.verify(fleet.operator.pk_o, msg + aux, sig_o)
    )

    admitted = bool(ok_honest_roster or ok_forged_roster or commissioned)
    return AttackResult(
        "rogue-subset-addition",
        admitted,
        ms + ms2,
        {
            "k": len(target_idx),
            "cancellation_exact": cancellation_exact,
            "naive_aggregate_admits": naive_admits,
            "pop_admits_rogue_key": pop_admits,
            "commissioning_admits_rogue_key": commissioned,
            "honest_roster_admits": bool(ok_honest_roster),
            "substituted_roster_admits": bool(ok_forged_roster),
            "substituted_roster_member_check": member_check,
            "substituted_roster_operator_check": operator_check,
        },
    )


def caste_forgery(fleet: Fleet, seed: int) -> AttackResult:
    """An unauthorised issuer signs a caste certificate.

    Named for what it tests. The adversary holds no operator key, so it
    creates its own and signs itself a `supervisor` certificate; the
    honest operator's public key does not verify that signature. This is
    a positive control on the signature path and nothing more: rejection
    follows from BLS verification alone and would hold for any scheme.
    An earlier revision ran exactly this scenario under the name
    "caste-laundering", which made a table row read as evidence for a
    property the experiment never exercised. Laundering is the separate
    scenario below.
    """
    robot = fleet.robots[seed % len(fleet.robots)]
    forged = Operator.create(f"forged/{seed}".encode())  # adversary's own key
    cert = forged.issue_caste(robot.pk_bytes, "supervisor", issued_at=1.0)
    roster = fleet.operator.rosters[fleet.epoch]
    admitted = fleet.operator.verify_caste(
        cert, roster, expected_epoch=fleet.epoch
    )
    return AttackResult("caste-forgery", admitted, 0.0)


def caste_laundering(fleet: Fleet, seed: int) -> AttackResult:
    """Replay of a genuinely issued certificate after demotion.

    This is the realistic vector: no forgery is involved and every
    signature verifies. The operator promotes a robot to `supervisor`,
    then demotes it to `observer` at the next epoch boundary, and the
    robot presents the `supervisor` certificate it was genuinely issued.
    Under an issuance-only check the certificate is indistinguishable
    from a current one, because the operator really did sign it. It is
    refused here because the epoch roster carries the caste in force and
    the presented caste no longer matches it.

    The `pre_demotion_admitted` field is the positive control: the same
    certificate against the roster of the epoch it was issued for must be
    admitted, or the rejection below would be evidence of nothing.

    This scenario promotes then demotes, which re-signs rosters and
    advances the operator's current epoch. It does that on a private
    fleet of its own rather than on the shared one the caller reuses
    across scenarios, so it cannot leave the operator on a later epoch
    and make a subsequent scenario's in-epoch decision fail its replay
    guard. The scenarios are otherwise side-effect free on the fleet
    they are handed.
    """
    local = Fleet.create(len(fleet.robots), 500_000 + seed)
    robot = local.robots[seed % len(local.robots)]
    op = local.operator
    e0 = local.epoch

    # Epoch e0: the operator genuinely promotes this robot.
    members = [(r.pk_bytes, bytes(bls.pop_prove(r.sk)[0])) for r in local.robots]
    castes = {r.pk_bytes: r.caste for r in local.robots}
    castes[robot.pk_bytes] = "supervisor"
    roster_promoted = op.sign_roster(
        e0, members, t_open=float(e0),
        capabilities={r.pk_bytes: r.manifest.cap for r in local.robots},
        castes=castes,
    )
    cert = op.issue_caste(robot.pk_bytes, "supervisor", issued_at=float(e0))
    pre = op.verify_caste(cert, roster_promoted, expected_epoch=e0)

    # Epoch e0+1: demoted. The certificate above is untouched and still
    # carries a valid operator signature.
    castes[robot.pk_bytes] = "observer"
    roster_demoted = op.sign_roster(
        e0 + 1, members, t_open=float(e0 + 1),
        capabilities={r.pk_bytes: r.manifest.cap for r in local.robots},
        castes=castes,
    )
    admitted = op.verify_caste(cert, roster_demoted, expected_epoch=e0 + 1)
    return AttackResult(
        "caste-laundering",
        admitted,
        0.0,
        detail={
            "pre_demotion_admitted": bool(pre),
            "signature_still_valid": bool(
                PopSchemeMPL.verify(
                    op.pk_o,
                    cert.signed_payload,
                    G2Element.from_bytes(cert.operator_signature),
                )
            ),
        },
    )


def operator_signature_replay(fleet: Fleet, seed: int) -> AttackResult:
    """Whole-tuple replay: resubmit a once-valid decision tuple later.

    The adversary captures a complete, valid tuple
    (msg, agg, sig_o, bitmap) accepted at (epoch, seq0) and resubmits it
    verbatim under a later sequence. All signatures still verify, but the
    operator's replay guard has already recorded (epoch, seq0), and the
    epoch binding in msg fixes the sequence, so the resubmission is
    rejected. Admitted is the verifier's REAL verdict on the replay.
    """
    epoch = fleet.epoch
    seq0 = fleet.next_seq()
    msg = canonical(epoch, seq0, f"decision/{seed}".encode())
    sigs = [PopSchemeMPL.sign(r.sk, msg) for r in fleet.robots]
    agg, _ = bls.aggregate(sigs)
    n = len(fleet.roster_pks)
    bitmap = _full_bitmap(n)
    sig_o, _ = bls.countersign_subset(
        fleet.operator.sk_o, msg, agg, bitmap, fleet.roster_pks, epoch, fleet.threshold
    )

    # first submission is accepted (records (epoch, seq0))
    first = submit_decision_tuple(
        fleet, epoch, seq0, msg, bytes(agg), bytes(sig_o), bitmap
    )
    if not first:
        raise AssertionError("legitimate first submission was rejected")

    # adversary resubmits the SAME tuple verbatim -> must be rejected
    replay = submit_decision_tuple(
        fleet, epoch, seq0, msg, bytes(agg), bytes(sig_o), bitmap
    )
    # and relabels it under a fresh sequence -> must also be rejected
    relabel = submit_decision_tuple(
        fleet, epoch, seq0 + 10_000, msg, bytes(agg), bytes(sig_o), bitmap
    )
    return AttackResult(
        "operator-signature-replay",
        bool(replay or relabel),
        0.0,
        {"verbatim_replay_admits": bool(replay),
         "relabelled_replay_admits": bool(relabel)},
    )


def malformed_bitmap(fleet: Fleet, seed: int) -> AttackResult:
    """Malformed participant bitmaps submitted to the public verifier.

    `verify_subset` is reachable by an untrusted prover, so a malformed
    bitmap must produce a rejection rather than an exception: a verifier
    that raises on bad input is a denial-of-service surface, and one that
    accepts out-of-range or repeated slots miscounts the quorum. Four
    malformations are submitted against an otherwise valid tuple:

      * truncated  -- fewer bytes than ceil(N/8), which would index past
        the end of the buffer,
      * over-long  -- extra trailing bytes carrying set bits,
      * high-bits  -- bits set above roster position N-1 in the final
        byte, claiming members that do not exist, and
      * short-quorum -- a well-formed bitmap naming fewer than t members.

    `admitted` is true if ANY malformation is accepted; `raised` records
    whether any of them crashed the verifier instead of returning False.
    """
    epoch = fleet.epoch
    seq = fleet.next_seq()
    msg = canonical(epoch, seq, f"decision/{seed}/malformed".encode())
    n = len(fleet.roster_pks)
    t = fleet.threshold
    quorum_idx = list(range(t))
    roster = fleet.operator.rosters[epoch]
    by_pk = {r.pk_bytes: r.sk for r in fleet.robots}
    sigs = [
        PopSchemeMPL.sign(by_pk[roster.member_pks[i]], msg) for i in quorum_idx
    ]
    agg, _ = bls.aggregate(sigs)
    honest_bitmap = _bitmap_from(quorum_idx, n)
    sig_o, _ = bls.countersign_subset(
        fleet.operator.sk_o, msg, agg, honest_bitmap, fleet.roster_pks, epoch, t
    )

    n_bytes = (n + 7) // 8
    variants = {
        "truncated": honest_bitmap[: max(n_bytes - 1, 0)],
        "over_long": honest_bitmap + b"\xff",
        "high_bits": (
            honest_bitmap[:-1] + bytes([honest_bitmap[-1] | 0x80])
            if n % 8
            else honest_bitmap + b"\x01"
        ),
        "short_quorum": _bitmap_from(quorum_idx[: max(t - 1, 0)], n),
    }

    detail: dict = {}
    admitted = False
    raised = False
    t0 = time.perf_counter()
    for name, bad in variants.items():
        try:
            ok, _ = bls.verify_subset(
                msg, agg, sig_o, fleet.roster_pks, bad,
                fleet.operator.pk_o, t, epoch,
            )
        except Exception as exc:                      # verifier crashed
            raised = True
            ok = False
            detail[f"{name}_raised"] = type(exc).__name__
        detail[f"{name}_admits"] = bool(ok)
        admitted = admitted or bool(ok)
    ms = (time.perf_counter() - t0) * 1000.0
    detail["any_raised"] = raised
    return AttackResult("malformed-bitmap", admitted or raised, ms, detail)


def operator_key_compromise(fleet: Fleet, seed: int) -> AttackResult:
    """Adversary holds sk_O and NOTHING else; it mints its own fleet.

    The adversary generates N key pairs of its own, produces valid PoPs
    for them (it knows those secrets), commissions them as the roster for
    a fresh epoch using the stolen sk_O, and drives a decision under that
    roster through the ordinary verifier entry point. No honest robot's
    secret key is used anywhere in this scenario -- `honest_keys_used` is
    reported and is 0 -- so admission=1.0 is the genuine theoretical
    bound: sk_O alone suffices to admit an arbitrary fleet.

    Side effect: this advances `fleet.epoch` to the compromised epoch, so
    it is run last in the scenario sequence for a given fleet.
    """
    epoch = fleet.epoch + 1
    n = len(fleet.robots)
    honest_pks = {r.pk_bytes for r in fleet.robots}

    # keys the adversary generated itself, with genuine PoPs
    adv_sks = [bls.keygen(f"adversary-c/{seed}/{i}".encode()) for i in range(n)]
    by_pk = {bytes(sk.get_g1()): sk for sk in adv_sks}
    members = [
        (pk, bytes(bls.pop_prove(sk)[0])) for pk, sk in by_pk.items()
    ]
    honest_keys_used = len(honest_pks & set(by_pk))

    # sk_O misused: an entirely adversarial roster is commissioned
    fleet.operator.sign_roster(epoch, members, t_open=1.0)
    fleet.epoch = epoch  # verifier now on the compromised epoch

    seq = fleet.next_seq()
    msg = canonical(epoch, seq, f"decision/{seed}/compromise".encode())
    roster = fleet.operator.rosters[epoch]
    t = fleet.threshold
    quorum_idx = list(range(t))
    sigs = [
        PopSchemeMPL.sign(by_pk[roster.member_pks[i]], msg) for i in quorum_idx
    ]
    agg, _ = bls.aggregate(sigs)
    bitmap = fleet.bitmap_for(quorum_idx)
    sig_o, _ = bls.countersign_subset(
        fleet.operator.sk_o, msg, agg, bitmap, fleet.roster_pks, epoch, fleet.threshold
    )
    t0 = time.perf_counter()
    ok = submit_decision_tuple(
        fleet, epoch, seq, msg, bytes(agg), bytes(sig_o), bitmap
    )
    ms = (time.perf_counter() - t0) * 1000.0
    return AttackResult(
        "operator-key-compromise",
        bool(ok),
        ms,
        {"honest_keys_used": honest_keys_used,
         "adversarial_roster_size": len(roster.member_pks),
         "quorum": t},
    )


def run_scenario(fleet: Fleet, scenario: str, seed: int) -> AttackResult:
    table = {
        "single-impersonation": single_impersonation,
        "rogue-subset-addition": rogue_subset_addition,
        "caste-forgery": caste_forgery,
        "caste-laundering": caste_laundering,
        "operator-signature-replay": operator_signature_replay,
        "malformed-bitmap": malformed_bitmap,
        "operator-key-compromise": operator_key_compromise,
    }
    return table[scenario](fleet, seed)
