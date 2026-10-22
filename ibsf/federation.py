"""Fleet construction and one end-to-end decision round (Sections V, VIII).

The decision path implements t-of-N quorum semantics: the operator
issues a call-for-sign to the whole roster and proceeds as soon as the
fastest t members have replied, where t = ceil(QUORUM_RATIO * N) is the
operator-set threshold used throughout the evaluation. End-to-end
latency is therefore the t-th order statistic of

    out-leg transport  +  member signing (real crypto)  +  in-leg transport

over the roster, plus aggregation, operator overhead, countersignature,
and verification (all real crypto).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from blspy import G1Element, G2Element, PrivateKey

from . import bls
from .audit_chain import AuditChain
from .manifest import IdentityManifest, sha256
from .network import NetworkModel
from .operator import CommissioningStats, Operator, default_quorum

DECISION_DOMAIN = b"ibsf/decision/v1"
# Separate domain for the audit-chain event that records an authorised
# decision, so a decision event digest can never collide with the signed
# decision message itself.
DECISION_EVENT_DOMAIN = b"ibsf/decision-event/v1"


def decision_event_payload(epoch: int, seq: int, tuple_digest: bytes) -> bytes:
    """Fixed-width, injective payload for a committed fleet decision.

    The labels are carried in the clear beside the digest rather than
    folded into it. Section VII describes an operator repopulating its
    (epoch, seq) uniqueness state from committed events after a restart,
    and a payload that were only `SHA256(domain || tuple)` cannot support
    that: the labels are not recoverable from 32 opaque bytes, so the
    guard would come back empty and a captured tuple would be accepted a
    second time. Laying out epoch and seq as fixed-width fields makes the
    described recovery a parse. The digest stays alongside them, so the
    event still binds the exact authorised tuple and a relabelled event
    is detectable against it.
    """
    return (
        epoch.to_bytes(8, "big") + seq.to_bytes(8, "big") + tuple_digest
    )


def parse_decision_event_payload(payload: bytes) -> tuple[int, int, bytes]:
    """Inverse of `decision_event_payload`: (epoch, seq, tuple_digest)."""
    if len(payload) != 48:
        raise ValueError(
            f"decision event payload must be 48 bytes, got {len(payload)}"
        )
    return (
        int.from_bytes(payload[:8], "big"),
        int.from_bytes(payload[8:16], "big"),
        payload[16:],
    )


def canonical(epoch: int, seq: int, decision_payload: bytes) -> bytes:
    """Canonical signed message binding epoch + monotonic seq + payload."""
    return (
        DECISION_DOMAIN
        + epoch.to_bytes(8, "big")
        + seq.to_bytes(8, "big")
        + sha256(decision_payload)
    )


def msg_binds(msg: bytes, epoch: int, seq: int) -> bool:
    """Check that `msg` is a canonical message for exactly (epoch, seq).

    Without this check a captured, once-valid tuple could be resubmitted
    under a fresh (epoch, seq) label: the signatures would still verify
    over the unchanged `msg` while the replay guard, seeing an unused
    label, would treat it as new. Binding the label to the signed bytes
    is what makes the guard sound (Section V-E).
    """
    prefix = (
        DECISION_DOMAIN + epoch.to_bytes(8, "big") + seq.to_bytes(8, "big")
    )
    return len(msg) == len(prefix) + 32 and msg.startswith(prefix)


def quorum_threshold(n: int) -> int:
    """The default quorum size for a fleet of `n` members.

    Retained as the name the experiment scripts use. It returns the stated
    *default*; it is not what the protocol enforces. The enforced value is
    the one the operator signed into the epoch roster, which
    `Fleet.threshold` reads and `bls.verify_subset` binds into the
    countersignature.
    """
    return default_quorum(n)


@dataclass
class Robot:
    """Per-robot state: key pair, manifest, audit chain."""

    robot_id: int
    sk: PrivateKey
    manifest: IdentityManifest
    chain: AuditChain = field(default_factory=AuditChain)
    # Caste is task-cadence role, deliberately outside the identity hash
    # (Definition 1). It is carried here and re-declared in each epoch
    # roster, which is what lets a verifier tell a current assignment from
    # a superseded one.
    caste: str = "worker"

    def __post_init__(self) -> None:
        # bind the audit chain to this robot's signing key so appended
        # events are signed (Section VI, signed audit events).
        if self.chain.sk is None:
            self.chain.sk = self.sk

    @property
    def pk_bytes(self) -> bytes:
        return bytes(self.sk.get_g1())


@dataclass
class Fleet:
    """Operator plus N member robots under one signed roster epoch."""

    operator: Operator
    robots: list[Robot]
    epoch: int
    commissioning: CommissioningStats = field(default_factory=CommissioningStats)
    _seq: int = 0

    @classmethod
    def create(cls, n: int, seed: int, epoch: int = 0) -> "Fleet":
        op = Operator.create(seed.to_bytes(8, "big"))
        robots: list[Robot] = []
        commissioning = CommissioningStats()
        members: list[tuple[bytes, bytes]] = []
        for i in range(n):
            sk = bls.keygen(f"robot/{seed}/{i}".encode())
            pk = bytes(sk.get_g1())
            # each robot proves possession of its secret at commissioning
            proof, _ = bls.pop_prove(sk)
            members.append((pk, bytes(proof)))
            manifest = IdentityManifest(
                pk=pk,
                root_o=op.root_cert_digest,
                cap=tuple(sorted(("navigate", "report", f"slot{i % 4}"))),
                audit_head=b"\x00" * 32,
            )
            robots.append(Robot(i, sk, manifest))
        # sign_roster verifies every PoP and rejects any key that fails,
        # recording per-commissioning PoP-verify timings for the paper.
        # The capability map goes in with the keys so that the operator's
        # roster signature covers what each member is certified to do,
        # not only that it is a member.
        op.sign_roster(
            epoch, members, t_open=0.0, stats=commissioning,
            capabilities={r.pk_bytes: r.manifest.cap for r in robots},
            castes={r.pk_bytes: r.caste for r in robots},
        )
        return cls(op, robots, epoch, commissioning)

    def next_seq(self) -> int:
        """Monotonic per-decision sequence number for this fleet."""
        s = self._seq
        self._seq += 1
        return s

    @property
    def roster_pks(self) -> list[G1Element]:
        roster = self.operator.rosters[self.epoch]
        return [G1Element.from_bytes(b) for b in roster.member_pks]

    @property
    def threshold(self) -> int:
        """The quorum size the operator signed into the current epoch roster.

        Read from the roster rather than recomputed, so that a decision is
        checked against the operator's declared policy for this epoch and
        not against whatever the default happens to be. Rosters signed
        before the threshold became part of the signed payload carry zero;
        for those the default stands in, which keeps older exports
        verifiable.
        """
        roster = self.operator.rosters[self.epoch]
        return roster.t_quorum or default_quorum(len(roster.member_pks))

    def bitmap_for(self, roster_indices: list[int]) -> bytes:
        """Participant bitmap over the current signed roster."""
        n = len(self.operator.rosters[self.epoch].member_pks)
        bits = bytearray((n + 7) // 8)
        for i in roster_indices:
            bits[i // 8] |= 1 << (i % 8)
        return bytes(bits)

    @property
    def roster_bitmap(self) -> bytes:
        """Full-participation bitmap over the current signed roster."""
        n = len(self.operator.rosters[self.epoch].member_pks)
        return self.bitmap_for(list(range(n)))

    @property
    def member_pks(self) -> dict[int, bytes]:
        """Map robot id -> that robot's key in the signed roster.

        Epoch close needs to attribute each Merkle leaf to a named member
        (`Operator.close_and_sign_epoch`), and the only correct source for
        that mapping is the robots' own keys checked against the roster.
        Providing it here keeps every experiment from rebuilding it, and
        the lookup below fails loudly if a robot is not in the roster.
        """
        positions = self.roster_positions()
        out: dict[int, bytes] = {}
        for robot in self.robots:
            pk = robot.pk_bytes
            if pk not in positions:
                raise KeyError(
                    f"robot {robot.robot_id} is not in the epoch "
                    f"{self.epoch} roster"
                )
            out[robot.robot_id] = pk
        return out

    def roster_index(self, pk_bytes: bytes) -> int:
        """Index of a member's key within the sorted signed roster."""
        return self.operator.rosters[self.epoch].member_pks.index(pk_bytes)

    def roster_positions(self) -> dict[bytes, int]:
        """Map member key -> position in the sorted signed roster.

        The signed roster is sorted by key bytes, which is NOT the order
        in which robots were constructed. Any code that selects a subset
        of `self.robots` and needs a participant bitmap must translate
        through this map, because bitmap bit i denotes ROSTER position i.
        """
        return {
            pk: i
            for i, pk in enumerate(self.operator.rosters[self.epoch].member_pks)
        }


@dataclass(frozen=True)
class DecisionResult:
    """One decision round: outcome plus latency decomposition (ms)."""

    verified: bool
    total_ms: float
    transport_ms: float
    member_sign_ms: float
    aggregate_ms: float
    operator_overhead_ms: float
    countersign_ms: float
    verify_ms: float
    epoch: int = 0
    seq: int = 0
    msg: bytes = b""
    aggregate: bytes = b""
    operator_sig: bytes = b""
    bitmap: bytes = b""
    participants: int = 0
    threshold: int = 0
    messages: int = 0


def run_decision(
    fleet: Fleet,
    decision_payload: bytes,
    net: NetworkModel,
    rng: np.random.Generator,
) -> DecisionResult:
    """Execute one fleet-level decision with real crypto, sim transport.

    Quorum semantics: the operator proceeds on the fastest t = ceil(0.8 N)
    replies rather than waiting for the whole roster, so the transport and
    signing contribution to latency is the t-th order statistic over the
    roster, not the maximum. The message members sign binds the current
    epoch and a fresh monotonic sequence number; verification resolves the
    participating set from the operator-signed roster via the bitmap and
    recomputes the key-list digest, so neither the key list nor the label
    can be substituted by the caller.
    """
    epoch = fleet.epoch
    seq = fleet.next_seq()
    msg = canonical(epoch, seq, decision_payload)
    n = len(fleet.robots)
    t = fleet.threshold

    # per-member arrival time at the operator: out-leg + sign + in-leg
    out_leg = net.one_way(rng, n)
    in_leg = net.one_way(rng, n)
    sign_times: list[float] = []
    sigs: list[G2Element] = []
    for robot in fleet.robots:
        sig, ms = bls.member_sign(robot.sk, msg)
        sigs.append(sig)
        sign_times.append(ms)
    arrival = out_leg + np.asarray(sign_times) + in_leg

    # operator proceeds on the fastest t replies (quorum order statistic)
    order = np.argsort(arrival)
    quorum_idx = sorted(int(i) for i in order[:t])
    quorum_wait = float(arrival[order[t - 1]])
    transport = float(out_leg[order[t - 1]] + in_leg[order[t - 1]])
    member_sign = float(sign_times[order[t - 1]])

    quorum_sigs = [sigs[i] for i in quorum_idx]
    agg, agg_ms = bls.aggregate(quorum_sigs)
    overhead = net.operator_overhead(rng)
    # the bitmap indexes the SORTED signed roster, not the construction
    # order of `fleet.robots`; translate the quorum members' keys to their
    # roster positions so the verifier resolves exactly the keys that
    # actually contributed a share.
    pos = fleet.roster_positions()
    quorum_roster_idx = sorted(pos[fleet.robots[i].pk_bytes] for i in quorum_idx)
    bitmap = fleet.bitmap_for(quorum_roster_idx)
    sig_o, cs_ms = bls.countersign_subset(
        fleet.operator.sk_o, msg, agg, bitmap, fleet.roster_pks, epoch, t
    )
    ok, verify_ms = bls.verify_subset(
        msg, agg, sig_o, fleet.roster_pks, bitmap,
        fleet.operator.pk_o, t, epoch,
    )
    # replay guard: the label must be bound into the signed bytes, and
    # the (epoch, seq) pair accepted at most once
    ok = ok and msg_binds(msg, epoch, seq)
    ok = ok and fleet.operator.check_and_record_decision(epoch, seq)

    # Commit the authorised decision to the audit chain. Algorithm 1's
    # Sign step outputs a tuple that "is appended to the audit chain", and
    # the regulator-facing guarantees rest on the decision being present in
    # the record the compliance certificate aggregates. An authorised
    # fleet decision is recorded on the chain of the lowest-indexed quorum
    # member (a deterministic, roster-resolved anchor), as a fleet-decision
    # event whose payload carries the decision's (epoch, seq) labels beside
    # a digest of the exact authorised tuple, so a later inclusion proof
    # ties the certificate to this decision and the replay guard's state is
    # recoverable from the record. Only a decision that actually verified
    # and passed the replay guard is committed; a rejected tuple leaves the
    # chain untouched.
    if ok:
        # Anchor by minimum ROSTER position among the participants, not
        # by construction order. `quorum_idx` indexes fleet.robots, which
        # is the order the fleet happened to be built in; the roster is
        # sorted by public key, so the two disagree for almost every
        # fleet. Only the roster position is recoverable by a verifier
        # holding the roster and the bitmap, so it is the one that can
        # appear in a claim about where the record lives.
        anchor = min(
            (fleet.robots[i] for i in quorum_idx),
            key=lambda r: pos[r.pk_bytes],
        )
        decision_digest = sha256(
            DECISION_EVENT_DOMAIN + msg + bytes(agg) + bytes(sig_o) + bitmap
        )
        anchor.chain.append(
            "fleet-decision",
            decision_event_payload(epoch, seq, decision_digest),
            anchor.sk,
        )

    total = quorum_wait + agg_ms + overhead + cs_ms + verify_ms
    # Message count is emitted per trial rather than asserted in prose, so
    # the O(N) claim is checkable against the same traces that carry the
    # latencies. The count must be like-for-like with the baseline: PBFT's
    # commit round intrinsically delivers the outcome to every replica, so
    # counting only the federation's call-for-sign and replies compares a
    # decision that has been disseminated against one that has not, and
    # overstates the separation. Three legs are counted here: one
    # call-for-sign to each of the N members, one reply from each of the
    # t quorum members, and one dissemination of the authorised decision
    # back to all N.
    messages = n + len(quorum_idx) + n
    return DecisionResult(
        ok, total, transport, member_sign, agg_ms, overhead, cs_ms,
        verify_ms, epoch, seq, msg, bytes(agg), bytes(sig_o), bitmap,
        len(quorum_idx), t, messages,
    )


def submit_decision_tuple(
    fleet: Fleet,
    epoch: int,
    seq: int,
    msg: bytes,
    aggregate: bytes,
    operator_sig: bytes,
    bitmap: bytes,
) -> bool:
    """Verifier entry point for a submitted decision tuple.

    Three independent conditions must hold: the signed bytes must encode
    exactly the claimed (epoch, seq) label; the aggregate and operator
    countersignature must verify against the roster-resolved participant
    set and the epoch's key-list digest; and the label must be fresh under
    the operator's replay guard. Relabelling a captured tuple therefore
    fails the first check, and resubmitting it verbatim fails the third.
    """
    if not msg_binds(msg, epoch, seq):
        return False
    try:
        agg_el = G2Element.from_bytes(aggregate)
        sig_o_el = G2Element.from_bytes(operator_sig)
    except Exception:
        return False
    ok, _ = bls.verify_subset(
        msg, agg_el, sig_o_el, fleet.roster_pks, bitmap,
        fleet.operator.pk_o, fleet.threshold, epoch,
    )
    if not ok:
        return False
    return fleet.operator.check_and_record_decision(epoch, seq)
