"""Regression tests for the security properties the paper claims.

These are deliberately property-shaped rather than coverage-shaped. Each
test names a claim made in the paper and tries to break it, and every
rejection test is paired with a positive control, because a verifier that
rejects everything passes every negative test while being useless.

Several of these encode defects that were live at some point in this
implementation's history and were caught by review rather than by the
code: a participant bitmap short enough to index past its buffer, a
quorum satisfied by one member repeated across slots, an epoch root that
could be dropped from an exported certificate without detection, and an
identity manifest that no code path ever checked. They are tests so that
those specific regressions fail here instead of in a reviewer's report.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from blspy import G1Element, G2Element, PopSchemeMPL

from ibsf import bls
from ibsf.audit_chain import AuditChain, epoch_leaf
from ibsf.certificate import export_certificate, regulator_verify
from ibsf.federation import (
    Fleet,
    canonical,
    parse_decision_event_payload,
    run_decision,
    submit_decision_tuple,
)
from ibsf.manifest import IdentityManifest
from ibsf.network import NetworkModel
from ibsf.operator import (
    Operator,
    SignedRoster,
    default_quorum,
    verify_epoch_root,
    verify_manifest,
)

N = 12


@pytest.fixture(scope="module")
def fleet() -> Fleet:
    return Fleet.create(N, seed=4242)


def _signed_decision(fleet: Fleet, payload: bytes):
    """A genuine, verifying decision tuple over the quorum."""
    epoch, seq = fleet.epoch, fleet.next_seq()
    msg = canonical(epoch, seq, payload)
    roster = fleet.operator.rosters[epoch]
    by_pk = {r.pk_bytes: r.sk for r in fleet.robots}
    t = fleet.threshold
    idx = list(range(t))
    sigs = [PopSchemeMPL.sign(by_pk[roster.member_pks[i]], msg) for i in idx]
    agg, _ = bls.aggregate(sigs)
    bitmap = fleet.bitmap_for(idx)
    sig_o, _ = bls.countersign_subset(
        fleet.operator.sk_o, msg, agg, bitmap, fleet.roster_pks, epoch, t
    )
    return epoch, seq, msg, agg, sig_o, bitmap, t


# --------------------------------------------------------------------
# positive controls: the verifier must accept what it should
# --------------------------------------------------------------------

def test_honest_decision_verifies(fleet: Fleet) -> None:
    """Without this, every rejection test below is vacuous."""
    epoch, _, msg, agg, sig_o, bitmap, t = _signed_decision(fleet, b"honest")
    ok, _ = bls.verify_subset(
        msg, agg, sig_o, fleet.roster_pks, bitmap,
        fleet.operator.pk_o, t, epoch,
    )
    assert ok


def test_honest_manifest_admitted(fleet: Fleet) -> None:
    robot = fleet.robots[0]
    assert verify_manifest(
        fleet.operator.pk_o,
        fleet.operator.rosters[fleet.epoch],
        robot.manifest,
        fleet.operator.root_cert_digest,
        expected_epoch=fleet.epoch,
    )


# --------------------------------------------------------------------
# Property: an unlisted key is not a fleet member (Sybil resistance)
# --------------------------------------------------------------------

def test_fabricated_manifest_rejected(fleet: Fleet) -> None:
    """A manifest is cheap to forge; roster membership is not."""
    adv = bls.keygen(b"adversary/manifest")
    forged = IdentityManifest(
        pk=bytes(adv.get_g1()),
        root_o=fleet.operator.root_cert_digest,
        cap=("navigate", "supervise"),
        audit_head=b"\x00" * 32,
    )
    assert not verify_manifest(
        fleet.operator.pk_o,
        fleet.operator.rosters[fleet.epoch],
        forged,
        fleet.operator.root_cert_digest,
        expected_epoch=fleet.epoch,
    )


@pytest.mark.parametrize("mutate", ["escalate", "reduce", "borrow"])
def test_capability_claim_must_match_the_signed_roster(
    fleet: Fleet, mutate: str
) -> None:
    """A listed member may not claim capabilities it was not granted.

    Roster membership authenticates *who*; without a capability binding
    it says nothing about *what*, and a genuine member could hand over a
    manifest naming any privilege it liked. This is the case that was
    live until the roster began covering per-member capability digests.
    """
    robot = fleet.robots[0]
    if mutate == "escalate":
        cap = tuple(sorted(robot.manifest.cap + ("override_safety",)))
    elif mutate == "reduce":
        cap = tuple(sorted(robot.manifest.cap[:-1]))
    else:
        cap = fleet.robots[1].manifest.cap
        if cap == robot.manifest.cap:
            pytest.skip("neighbour holds an identical capability set")
    forged = IdentityManifest(
        pk=robot.pk_bytes,
        root_o=fleet.operator.root_cert_digest,
        cap=cap,
        audit_head=b"\x00" * 32,
    )
    assert not verify_manifest(
        fleet.operator.pk_o,
        fleet.operator.rosters[fleet.epoch],
        forged,
        fleet.operator.root_cert_digest,
        expected_epoch=fleet.epoch,
    )


def test_capability_digest_is_injective_over_separators() -> None:
    """Delimiter joining would collide these two distinct surfaces."""
    from ibsf.operator import capability_digest
    assert capability_digest(("a", "b")) != capability_digest(("a\x1fb",))


def test_manifest_from_a_different_operator_rejected(fleet: Fleet) -> None:
    robot = fleet.robots[0]
    wrong_root = IdentityManifest(
        pk=robot.pk_bytes,
        root_o=b"\x11" * 32,                 # some other operator
        cap=robot.manifest.cap,
        audit_head=b"\x00" * 32,
    )
    assert not verify_manifest(
        fleet.operator.pk_o,
        fleet.operator.rosters[fleet.epoch],
        wrong_root,
        fleet.operator.root_cert_digest,
        expected_epoch=fleet.epoch,
    )


# --------------------------------------------------------------------
# Verifier robustness: malformed input rejects, never raises
# --------------------------------------------------------------------

@pytest.mark.parametrize("mangle", ["truncate", "extend", "high_bits"])
def test_malformed_bitmap_rejected_without_raising(
    fleet: Fleet, mangle: str
) -> None:
    """A public verifier that raises on bad input is a DoS surface."""
    epoch, _, msg, agg, sig_o, bitmap, t = _signed_decision(fleet, b"mal")
    if mangle == "truncate":
        bad = bitmap[:-1]
    elif mangle == "extend":
        bad = bitmap + b"\xff"
    else:
        bad = bitmap[:-1] + bytes([bitmap[-1] | 0x80])
    ok, _ = bls.verify_subset(
        msg, agg, sig_o, fleet.roster_pks, bad,
        fleet.operator.pk_o, t, epoch,
    )
    assert ok is False


def test_subquorum_bitmap_rejected(fleet: Fleet) -> None:
    epoch, _, msg, agg, sig_o, _, t = _signed_decision(fleet, b"sub")
    short = fleet.bitmap_for(list(range(t - 1)))
    ok, _ = bls.verify_subset(
        msg, agg, sig_o, fleet.roster_pks, short,
        fleet.operator.pk_o, t, epoch,
    )
    assert ok is False


def test_stale_roster_does_not_admit_a_removed_member(fleet: Fleet) -> None:
    """Admission on yesterday's roster is the Sybil check's quiet failure.

    A signed roster is valid for ever, so every condition verify_manifest
    checks can pass on a document five epochs old, admitting a robot the
    operator has since removed. The verifying member therefore states the
    epoch it believes current, and a roster for any other epoch is refused
    before its contents are considered. Here the stale roster is otherwise
    impeccable: right operator, right member, right capability digests.
    """
    op = fleet.operator
    robot = fleet.robots[0]
    members = [(r.pk_bytes, bytes(bls.pop_prove(r.sk)[0])) for r in fleet.robots]
    caps = {r.pk_bytes: r.manifest.cap for r in fleet.robots}

    old_epoch, new_epoch = 70, 71
    old_roster = op.sign_roster(
        old_epoch, members, t_open=70.0, capabilities=caps
    )
    # The next epoch's roster excludes robot 0 entirely.
    op.sign_roster(
        new_epoch, members[1:], t_open=71.0,
        capabilities={k: v for k, v in caps.items() if k != robot.pk_bytes},
    )

    # Positive control: the old roster admitted the robot in its own epoch.
    assert verify_manifest(
        op.pk_o, old_roster, robot.manifest, op.root_cert_digest,
        expected_epoch=old_epoch,
    )
    # The same impeccable document, presented when epoch 71 is current.
    assert not verify_manifest(
        op.pk_o, old_roster, robot.manifest, op.root_cert_digest,
        expected_epoch=new_epoch,
    )


# --------------------------------------------------------------------
# Property: a caste claim must be current, not merely once-issued
# --------------------------------------------------------------------

def test_identity_hash_is_byte_stable_across_caste_change(
    fleet: Fleet,
) -> None:
    """Property 1, the abstract's headline, previously had no test at all.

    The only test touching `identity_hash` asserted the OPPOSITE
    direction (capability injectivity), so a change folding caste into
    the hash -- which Section IV itself discusses as a design alternative
    -- would have passed the whole suite. Here one robot receives k
    successive caste assignments and the hash must not move by a byte.
    """
    op = fleet.operator
    robot = fleet.robots[0]
    h0 = robot.manifest.identity_hash
    for i, caste in enumerate(("worker", "scout", "supervisor", "observer")):
        cert = op.issue_caste(robot.pk_bytes, caste, issued_at=float(i))
        assert cert.caste_name == caste
        assert robot.manifest.identity_hash == h0, (
            "identity hash moved on caste change; Definition 1's hash "
            "boundary has been violated"
        )


def test_superseded_caste_certificate_is_refused(fleet: Fleet) -> None:
    """The demotion case: genuine signature, stale claim.

    No forgery here. The operator really did issue this certificate, and
    its signature still verifies at the end of the test. What changed is
    the roster: the next epoch declares the robot an `observer`, so the
    `supervisor` claim no longer matches the caste in force. An
    issuance-only check cannot tell the two apart, which is why an
    earlier revision accepted this indefinitely.
    """
    op = fleet.operator
    robot = fleet.robots[0]
    members = [(r.pk_bytes, bytes(bls.pop_prove(r.sk)[0])) for r in fleet.robots]
    caps = {r.pk_bytes: r.manifest.cap for r in fleet.robots}
    castes = {r.pk_bytes: r.caste for r in fleet.robots}

    castes[robot.pk_bytes] = "supervisor"
    promoted = op.sign_roster(
        50, members, t_open=50.0, capabilities=caps, castes=castes
    )
    cert = op.issue_caste(robot.pk_bytes, "supervisor", issued_at=50.0)
    assert op.verify_caste(cert, promoted, expected_epoch=50), (
        "positive control: current claim"
    )

    castes[robot.pk_bytes] = "observer"
    demoted = op.sign_roster(
        51, members, t_open=51.0, capabilities=caps, castes=castes
    )
    assert op.verify_caste(cert, demoted, expected_epoch=51) is False

    # The rejection must come from currency, not from a broken signature.
    sig = G2Element.from_bytes(cert.operator_signature)
    assert PopSchemeMPL.verify(op.pk_o, cert.signed_payload, sig)


def test_authorised_decision_enters_the_audit_chain() -> None:
    """Algorithm 1 says the tuple is appended to the audit chain.

    The regulator-facing guarantees rest on an authorised fleet decision
    being present in the record the compliance certificate aggregates. An
    earlier revision verified the tuple and returned it without appending
    anything, so a hazardous decision left every chain empty and could not
    be forensically audited. This checks that an authorised decision adds
    exactly one signed fleet-decision event, and a rejected one adds
    nothing.
    """
    f = Fleet.create(10, seed=71)
    before = sum(len(r.chain.events) for r in f.robots)
    res = run_decision(f, b"hazardous", NetworkModel(), np.random.default_rng(3))
    assert res.verified, "positive control: the decision must verify"
    after = sum(len(r.chain.events) for r in f.robots)
    assert after == before + 1, "authorised decision must add one chain event"
    kinds = [e.kind for r in f.robots for e in r.chain.events]
    assert "fleet-decision" in kinds
    # the event carries a signature (96-byte BLS), not a bare record
    ev = next(e for r in f.robots for e in r.chain.events
              if e.kind == "fleet-decision")
    assert len(ev.signature) == 96

    # Negative control. Without it this test still passes if a later edit
    # moves the append outside the `if ok` guard: the positive case would
    # be unaffected and rejected decisions would silently enter the chain.
    # Burning the label the next round will draw makes that round fail the
    # replay guard, so nothing may be appended.
    lengths = [len(r.chain.events) for r in f.robots]
    assert f.operator.check_and_record_decision(f.epoch, f._seq)
    rejected = run_decision(
        f, b"hazardous", NetworkModel(), np.random.default_rng(3)
    )
    assert not rejected.verified, "negative control: the replay must be refused"
    assert [len(r.chain.events) for r in f.robots] == lengths, (
        "a rejected decision must leave every chain untouched"
    )


def test_decision_anchor_is_the_lowest_roster_position() -> None:
    """The paper names the anchor; a verifier must be able to find it.

    The anchor was `fleet.robots[quorum_idx[0]]`, the lowest
    CONSTRUCTION-order participant. The roster is sorted by public key,
    so that is usually a different member from the lowest roster
    position, and construction order is not visible to a verifier
    holding only the roster and the bitmap. Algorithm 1 names the roster
    position, so that is what the code must pick.

    Several seeds, not one: with a single fleet the two rules can select
    the same member by coincidence, and a first version of this test did
    exactly that -- it still passed when the old rule was put back. The
    `distinguishing` counter is the guard against that, and it is
    asserted, so this test fails rather than silently going vacuous if
    the fixture ever stops separating the two rules.
    """
    distinguishing = 0
    for seed in (71, 73, 77, 79, 83):
        f = Fleet.create(10, seed=seed)
        pos = f.roster_positions()
        res = run_decision(
            f, b"hazardous", NetworkModel(), np.random.default_rng(seed)
        )
        assert res.verified

        holders = [r for r in f.robots
                   if any(e.kind == "fleet-decision" for e in r.chain.events)]
        assert len(holders) == 1, "exactly one chain carries the record"

        # bitmap_for sets bit (i % 8) of byte (i // 8): LSB-first.
        # Decoding it MSB-first silently drops and invents participants,
        # which is how the first draft of this test read a member that
        # was not in the quorum.
        bitmap = res.bitmap
        participants = [
            i for i in range(len(f.roster_pks))
            if bitmap[i // 8] >> (i % 8) & 1
        ]
        assert len(participants) == f.threshold, (
            "positive control: the decoded bitmap must hold exactly t members"
        )
        assert pos[holders[0].pk_bytes] == min(participants), (
            f"seed {seed}: the record must live on the lowest "
            f"participating roster position"
        )

        # Would the discarded rule have chosen someone else here?
        by_construction = min(
            (r for r in f.robots if pos[r.pk_bytes] in participants),
            key=lambda r: r.robot_id,
        )
        if by_construction is not holders[0]:
            distinguishing += 1

    assert distinguishing > 0, (
        "no seed separates roster position from construction order, so this "
        "test would pass under the defect it exists to catch"
    )


def test_committed_decision_event_repopulates_the_replay_guard() -> None:
    """Section VII's stated recovery path must be a parse, not an inversion.

    The manuscript says a production operator rebuilds its (epoch, seq)
    uniqueness state from committed events. That is only realizable if
    the labels survive in the record: a payload of 32 opaque digest bytes
    cannot be inverted, so a restarted operator would come back with an
    empty guard and re-admit a captured tuple. The event therefore
    carries the labels in fixed-width fields beside the digest.
    """
    f = Fleet.create(10, seed=72)
    res = run_decision(f, b"hazardous", NetworkModel(), np.random.default_rng(5))
    assert res.verified
    ev = next(e for r in f.robots for e in r.chain.events
              if e.kind == "fleet-decision")
    epoch, seq, digest = parse_decision_event_payload(ev.payload)
    assert (epoch, seq) == (res.epoch, res.seq)
    assert len(digest) == 32

    # A fresh operator that has only the committed chains can rebuild the
    # guard from them, and then refuses the tuple it has already seen.
    recovered = Operator(sk_o=f.operator.sk_o)
    for robot in f.robots:
        for event in robot.chain.events:
            if event.kind == "fleet-decision":
                e, s, _ = parse_decision_event_payload(event.payload)
                assert recovered.check_and_record_decision(e, s)
    assert not recovered.check_and_record_decision(res.epoch, res.seq)


def test_verify_caste_rejects_a_forged_roster(fleet: Fleet) -> None:
    """Currency evidence is only as good as the roster's own signature.

    A caller could hand verify_caste a roster it built itself, with a
    caste digest chosen to match a stale certificate. If the method
    trusted the roster's contents without checking the roster's operator
    signature, the stale certificate would be accepted. It must not: the
    roster is verified under root_O first. This is the flaw verify_manifest
    also had, closed the same way, and a reviewer caught that verify_caste
    had not been given the same treatment.
    """
    op = fleet.operator
    robot = fleet.robots[0]
    cert = op.issue_caste(robot.pk_bytes, "supervisor", issued_at=90.0)

    # A roster the operator never signed, whose caste digest is crafted to
    # match the stale certificate. Signed by an adversary's key, not root_O.
    from ibsf.manifest import caste_digest
    adversary = Operator.create(b"forged-roster-adversary")
    forged = op.sign_roster(  # start from a genuine structure...
        90, [(r.pk_bytes, bytes(bls.pop_prove(r.sk)[0])) for r in fleet.robots],
        t_open=90.0,
        capabilities={r.pk_bytes: r.manifest.cap for r in fleet.robots},
        castes={r.pk_bytes: "supervisor" for r in fleet.robots},
    )
    # ...then re-sign the same digest under the adversary key, so the caste
    # digest matches the cert but the signature is not the operator's.
    forged_bad = SignedRoster(
        forged.epoch, forged.member_pks, forged.t_open,
        bytes(PopSchemeMPL.sign(adversary.sk_o, forged.digest)),
        forged.cap_digests, forged.t_quorum, forged.caste_digests,
    )
    assert forged_bad.caste_digest_for(robot.pk_bytes) == caste_digest(
        "supervisor"
    ), "test setup: the forged roster must carry the matching caste digest"
    assert op.verify_caste(cert, forged_bad, expected_epoch=90) is False


def test_revocation_takes_effect_at_the_next_boundary(fleet: Fleet) -> None:
    """A revoked member stops verifying once the next roster is signed.

    Revocation is a roster operation: the operator signs the next epoch
    without the member, and the epoch-currency check in verify_manifest
    does the rest. The exposure bound the paper states is one epoch
    length, which is what the positive control below fixes -- the member
    is admissible right up to the boundary, and refused after it.
    """
    op = fleet.operator
    victim = fleet.robots[0]
    members = [(r.pk_bytes, bytes(bls.pop_prove(r.sk)[0])) for r in fleet.robots]
    caps = {r.pk_bytes: r.manifest.cap for r in fleet.robots}

    before = op.sign_roster(80, members, t_open=80.0, capabilities=caps)
    assert verify_manifest(
        op.pk_o, before, victim.manifest, op.root_cert_digest,
        expected_epoch=80,
    ), "positive control: admissible before revocation"

    after = op.revoke_member(
        81, members, victim.pk_bytes, t_open=81.0, capabilities=caps
    )
    assert victim.pk_bytes not in after.member_pks
    assert not verify_manifest(
        op.pk_o, after, victim.manifest, op.root_cert_digest,
        expected_epoch=81,
    )
    # a survivor is unaffected
    assert verify_manifest(
        op.pk_o, after, fleet.robots[1].manifest, op.root_cert_digest,
        expected_epoch=81,
    )


def test_revoking_absent_member_raises(fleet: Fleet) -> None:
    """A caller cannot believe it revoked a key that was never present."""
    op = fleet.operator
    members = [(r.pk_bytes, bytes(bls.pop_prove(r.sk)[0])) for r in fleet.robots]
    stranger = bls.keygen(b"never-a-member")
    with pytest.raises(ValueError, match="not in the prior member set"):
        op.revoke_member(
            82, members, bytes(stranger.get_g1()), t_open=82.0
        )


def test_caste_certificate_payload_is_domain_separated(fleet: Fleet) -> None:
    """Every sk_O-signed family carries a distinct constant prefix."""
    from ibsf.manifest import CASTE_DOMAIN

    cert = fleet.operator.issue_caste(
        fleet.robots[0].pk_bytes, "worker", issued_at=1.0
    )
    assert cert.signed_payload.startswith(CASTE_DOMAIN)


# --------------------------------------------------------------------
# Property: the quorum threshold is operator policy, not a caller's choice
# --------------------------------------------------------------------

def test_operator_can_require_a_larger_quorum_than_the_default(
    fleet: Fleet,
) -> None:
    """An operator raising t for a hazardous action must be obeyed.

    The paper describes t as operator policy with ceil(0.8N) as a default.
    An earlier revision made that description false: t was a module
    constant, nothing signed carried it, and a fleet whose operator wanted
    unanimity had no way to say so. This is that scenario -- the operator
    signs t = N, and a decision carrying only the default quorum must not
    verify.
    """
    n = len(fleet.robots)
    epoch = fleet.epoch + 1
    members = [(r.pk_bytes, bytes(bls.pop_prove(r.sk)[0])) for r in fleet.robots]
    roster = fleet.operator.sign_roster(
        epoch, members, t_open=float(epoch), t_quorum=n,
        capabilities={r.pk_bytes: r.manifest.cap for r in fleet.robots},
    )
    assert roster.t_quorum == n

    roster_pks = [G1Element.from_bytes(b) for b in roster.member_pks]
    by_pk = {r.pk_bytes: r.sk for r in fleet.robots}
    idx = list(range(default_quorum(n)))          # the default, not n
    assert len(idx) < n, "default quorum must be a strict subset for this test"

    msg = canonical(epoch, 1, b"hazardous")
    sigs = [PopSchemeMPL.sign(by_pk[roster.member_pks[i]], msg) for i in idx]
    agg, _ = bls.aggregate(sigs)
    bitmap = bytearray((n + 7) // 8)
    for i in idx:
        bitmap[i // 8] |= 1 << (i % 8)
    bitmap = bytes(bitmap)

    sig_o, _ = bls.countersign_subset(
        fleet.operator.sk_o, msg, agg, bitmap, roster_pks, epoch, roster.t_quorum,
    )
    ok, _ = bls.verify_subset(
        msg, agg, sig_o, roster_pks, bitmap,
        fleet.operator.pk_o, roster.t_quorum, epoch,
    )
    assert ok is False, "default quorum satisfied a roster that demanded all N"


def test_verifier_cannot_lower_the_signed_threshold(fleet: Fleet) -> None:
    """Passing a smaller t must fail, not weaken the check.

    `threshold` is an argument to `verify_subset`, so the guarantee cannot
    rest on callers choosing it honestly. It rests on the countersignature
    aux committing to the key-list digest, which covers t: verify the
    same decision under any other threshold and the recomputed aux differs,
    so the operator half fails even though the member half would pass.
    """
    epoch, _, msg, agg, sig_o, bitmap, t = _signed_decision(fleet, b"lower")
    ok_honest, _ = bls.verify_subset(
        msg, agg, sig_o, fleet.roster_pks, bitmap,
        fleet.operator.pk_o, t, epoch,
    )
    assert ok_honest, "positive control: the honest threshold must verify"

    ok_lowered, _ = bls.verify_subset(
        msg, agg, sig_o, fleet.roster_pks, bitmap,
        fleet.operator.pk_o, t - 1, epoch,
    )
    assert ok_lowered is False, "a caller-chosen threshold was accepted"


# --------------------------------------------------------------------
# Property: distinct members, not distinct slots, make a quorum
# --------------------------------------------------------------------

def test_duplicate_key_cannot_be_commissioned(fleet: Fleet) -> None:
    """One member occupying t slots would satisfy t-of-N on its own."""
    robot = fleet.robots[0]
    proof, _ = bls.pop_prove(robot.sk)
    dup = [(robot.pk_bytes, bytes(proof))] * 2
    with pytest.raises(ValueError, match="duplicate"):
        fleet.operator.sign_roster(99, dup, t_open=0.0)


def test_pop_is_enforced_at_commissioning(fleet: Fleet) -> None:
    adv = bls.keygen(b"adversary/nopop")
    other = bls.keygen(b"adversary/other")
    wrong_proof, _ = bls.pop_prove(other)      # PoP for a different key
    with pytest.raises(ValueError, match="possession"):
        fleet.operator.sign_roster(
            98, [(bytes(adv.get_g1()), bytes(wrong_proof))], t_open=0.0
        )


# --------------------------------------------------------------------
# Property: a captured decision tuple cannot be replayed or relabelled
# --------------------------------------------------------------------

def test_replay_and_relabel_rejected() -> None:
    f = Fleet.create(N, seed=777)
    epoch, seq, msg, agg, sig_o, bitmap, _ = _signed_decision(f, b"replay")
    assert submit_decision_tuple(
        f, epoch, seq, msg, bytes(agg), bytes(sig_o), bitmap
    )
    assert not submit_decision_tuple(
        f, epoch, seq, msg, bytes(agg), bytes(sig_o), bitmap
    )
    assert not submit_decision_tuple(
        f, epoch, seq + 5000, msg, bytes(agg), bytes(sig_o), bitmap
    )


def test_substituted_roster_fails_the_countersignature(fleet: Fleet) -> None:
    """The online path authenticates the roster, not just the keys."""
    epoch, _, msg, agg, sig_o, bitmap, t = _signed_decision(fleet, b"subst")
    adversary_roster = list(fleet.roster_pks)
    adversary_roster[0] = bls.keygen(b"adversary/roster").get_g1()
    ok, _ = bls.verify_subset(
        msg, agg, sig_o, adversary_roster, bitmap,
        fleet.operator.pk_o, t, epoch,
    )
    assert ok is False


# --------------------------------------------------------------------
# Property: the audit chain is tamper-evident and identity-bound
# --------------------------------------------------------------------

def test_event_hash_covers_the_signature() -> None:
    """Stripping a signature must move the chain commitment."""
    sk = bls.keygen(b"chain/1")
    chain = AuditChain(sk=sk)
    ev = chain.append("capability-invocation", b"payload")
    stripped = type(ev)(ev.seq, ev.kind, ev.payload, ev.prev_hash, ev.signer_pk, b"")
    assert ev.event_hash != stripped.event_hash


def test_tampered_event_breaks_verification() -> None:
    sk = bls.keygen(b"chain/2")
    chain = AuditChain(sk=sk)
    for i in range(4):
        chain.append("capability-invocation", f"e{i}".encode())
    assert chain.verify(sk.get_g1())
    ev = chain.events[1]
    chain.events[1] = type(ev)(
        ev.seq, ev.kind, b"mutated", ev.prev_hash, ev.signer_pk, ev.signature
    )
    assert not chain.verify(sk.get_g1())


def test_capability_encoding_is_injective() -> None:
    """Delimiter joining would collide on names containing the delimiter.

    ("a","b") and ("a\\x1fb",) are different certified capability sets;
    a delimiter-joined encoding gives them one identity hash, so a robot
    could be certified for one and present as the other.
    """
    base = dict(pk=b"\x01" * 48, root_o=b"\x02" * 32, audit_head=b"\x00" * 32)
    two = IdentityManifest(cap=("a", "b"), **base)
    one = IdentityManifest(cap=("a\x1fb",), **base)
    assert two.identity_hash != one.identity_hash


def test_pbft_preprepare_counts_the_leader() -> None:
    """The leader holds the message at t=0 and counts toward the quorum.

    Waiting for the quorum-th follower instead of the (quorum-1)-th
    overstates the round, which biases the baseline comparison in our
    favour.
    """
    from ibsf import pbft

    net = NetworkModel()
    n, quorum = 500, 2 * ((500 - 1) // 3) + 1
    got = pbft._broadcast_quorum_ms(net, np.random.default_rng(7), n, quorum)
    draws = np.sort(net.one_way(np.random.default_rng(7), n - 1))
    assert got == pytest.approx(float(draws[quorum - 2]))
    assert got < float(draws[quorum - 1])


def test_epoch_leaf_binds_the_member_key() -> None:
    """A head-only leaf would not say whose chain ended there."""
    head = b"\xab" * 32
    a = epoch_leaf(3, 0, b"\x01" * 48, head)
    b = epoch_leaf(3, 0, b"\x02" * 48, head)
    assert a != b
    assert epoch_leaf(3, 1, b"\x01" * 48, head) != a   # slot is bound too
    assert epoch_leaf(4, 0, b"\x01" * 48, head) != a   # epoch is bound too


def test_epoch_close_rejects_chains_that_do_not_match_the_roster() -> None:
    f = Fleet.create(6, seed=8181)
    chains = {r.robot_id: r.chain for r in f.robots}
    for r in f.robots:
        r.chain.append("capability-invocation", b"e")
    partial = {k: v for k, v in chains.items() if k != 0}
    with pytest.raises(ValueError):
        f.operator.close_and_sign_epoch(
            f.epoch, partial, 0.0, 60.0, f.member_pks
        )


# --------------------------------------------------------------------
# Property: an exported certificate is complete, not just internally
# consistent -- omitted epochs must be detectable
# --------------------------------------------------------------------

def _fleet_with_epochs(k: int) -> Fleet:
    f = Fleet.create(6, seed=9090)
    members = [(r.pk_bytes, bytes(bls.pop_prove(r.sk)[0])) for r in f.robots]
    caps = {r.pk_bytes: r.manifest.cap for r in f.robots}
    castes = {r.pk_bytes: r.caste for r in f.robots}
    chains = {r.robot_id: r.chain for r in f.robots}
    for e in range(k):
        if e > 0:
            f.operator.sign_roster(
                e, members, t_open=60.0 * e, capabilities=caps,
                castes=castes,
            )
        for r in f.robots:
            r.chain.append("capability-invocation", f"{e}".encode())
        f.operator.close_and_sign_epoch(
            e, chains, 60.0 * e, 60.0 * (e + 1), f.member_pks
        )
    return f


def test_complete_certificate_verifies() -> None:
    f = _fleet_with_epochs(4)
    assert regulator_verify(export_certificate(f.operator, [0, 1, 2, 3]))


def _fleet_with_thresholds(thresholds: list[int]) -> Fleet:
    """One epoch per entry, identical membership, the given quorum sizes."""
    f = Fleet.create(6, seed=9091)
    members = [(r.pk_bytes, bytes(bls.pop_prove(r.sk)[0])) for r in f.robots]
    caps = {r.pk_bytes: r.manifest.cap for r in f.robots}
    chains = {r.robot_id: r.chain for r in f.robots}
    for e, t in enumerate(thresholds):
        f.operator.sign_roster(
            e, members, t_open=60.0 * e, capabilities=caps, t_quorum=t,
        )
        f.operator.close_and_sign_epoch(
            e, chains, 60.0 * e, 60.0 * (e + 1), f.member_pks
        )
    return f


def test_changed_policy_travels_with_the_certificate() -> None:
    """A policy change under static membership must not be deduplicated away.

    The epoch root once committed the member set alone, and the exporter
    deduplicated on it. Two epochs with the same members but different
    signed quorum thresholds therefore shared a commitment: only the
    earlier roster was embedded, the later epoch's threshold vanished
    from the document, and `regulator_verify` still returned true. An
    auditor holding that certificate would check an epoch-1 decision
    against a threshold the operator never set for epoch 1. Committing
    the whole policy makes the second epoch carry its own roster.
    """
    f = _fleet_with_thresholds([5, 6])
    cert = export_certificate(f.operator, [0, 1])
    assert regulator_verify(cert)
    carried = sorted((r.epoch, r.t_quorum) for r in cert.rosters)
    assert carried == [(0, 5), (1, 6)], (
        "each epoch's signed threshold must be present in the document"
    )


def test_static_policy_still_embeds_one_roster() -> None:
    """The O(k+N) claim: unchanged policy is one O(N) attachment for k epochs.

    Paired with the test above, this is what fixes the scope of the
    complexity statement: it holds for static POLICY, and the cost of a
    policy change is one further roster rather than a silent omission.
    """
    f = _fleet_with_thresholds([5, 5, 5, 5])
    cert = export_certificate(f.operator, [0, 1, 2, 3])
    assert regulator_verify(cert)
    assert len(cert.rosters) == 1


@pytest.mark.parametrize("keep", [0, 1])
def test_certificate_missing_a_policy_roster_is_rejected(keep: int) -> None:
    """Dropping either epoch's policy must break verification, not pass it."""
    f = _fleet_with_thresholds([5, 6])
    cert = export_certificate(f.operator, [0, 1])
    stripped = replace(cert, rosters=(cert.rosters[keep],))
    assert not regulator_verify(stripped)


@pytest.mark.parametrize("epochs", [[0, 1, 3], [1, 2, 3], [0, 1, 2]])
def test_certificate_with_a_missing_epoch_is_rejected(epochs: list[int]) -> None:
    """Interior, prefix, and suffix omission must all be caught.

    Without a previous-root commitment inside each signed epoch payload,
    every one of these still verifies: each signature is genuine, and
    nothing says which epochs were supposed to be present.
    """
    f = _fleet_with_epochs(4)
    assert not regulator_verify(export_certificate(f.operator, epochs))


def test_epoch_root_cannot_be_relabelled() -> None:
    f = _fleet_with_epochs(2)
    root = f.operator.epoch_roots[1]
    assert verify_epoch_root(f.operator.pk_o, root)
    relabelled = type(root)(
        0, root.fleet_root, root.t_open, root.t_close,
        root.roster_digest, root.signature, root.prev_root_digest,
    )
    assert not verify_epoch_root(f.operator.pk_o, relabelled)


# --------------------------------------------------------------------
# Quorum timing: the decision path must not wait on the whole roster
# --------------------------------------------------------------------

def test_decision_uses_a_quorum_not_the_full_roster() -> None:
    f = Fleet.create(20, seed=31337)
    res = run_decision(f, b"quorum", NetworkModel(), np.random.default_rng(0))
    assert res.verified
    assert res.participants == res.threshold < len(f.robots)
    # call-for-sign to all, replies from the quorum, outcome back to all.
    # The dissemination leg is counted so the message figure is comparable
    # with a consensus baseline, whose commit round delivers the decision
    # to every replica as part of the protocol.
    assert res.messages == 2 * len(f.robots) + res.threshold
