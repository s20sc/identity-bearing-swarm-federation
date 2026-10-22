"""Per-agent audit chains and fleet-level Merkle aggregation (Section VI)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from blspy import G1Element, G2Element, PopSchemeMPL, PrivateKey

from .manifest import sha256

GENESIS = b"\x00" * 32
EVENT_DOMAIN = b"ibsf/auditevent/v1"


@dataclass(frozen=True)
class AuditEvent:
    """One append-only, robot-signed audit-chain record.

    Each event carries the appending robot's BLS public key and a
    signature over the event body (seq, kind, payload, prev_hash), so a
    later auditor can confirm the record was produced by the claimed
    agent and was not forged or reordered after the fact (Section VI).
    """

    seq: int
    kind: str          # capability-invocation | caste-change | safety-intervention
    payload: bytes
    prev_hash: bytes
    signer_pk: bytes = b""    # 48-byte G1 public key of the appending robot
    signature: bytes = b""    # 96-byte BLS signature over signed_body

    @property
    def signed_body(self) -> bytes:
        """Canonical body the robot signs (excludes signer_pk/signature)."""
        return (
            EVENT_DOMAIN
            + self.seq.to_bytes(8, "big")
            + self.kind.encode()
            + b"\x1f"
            + self.payload
            + self.prev_hash
        )

    @property
    def event_hash(self) -> bytes:
        """Chain/Merkle commitment over the body AND its signature.

        The signer's key and signature are inside the committed digest,
        so the fleet Merkle root commits to *who signed what*, not merely
        to the event body. Without this, an event's signature could be
        stripped or swapped without changing the published root, and an
        inclusion proof would attest an unauthenticated record.
        """
        return sha256(
            self.signed_body + self.signer_pk + self.signature
        )


@dataclass
class AuditChain:
    """Append-only per-agent chain with SHA-256 hash pointers.

    Optionally bound to a robot signing key; when bound, every appended
    event is signed with that key so the chain is verifiable against the
    robot's public key.
    """

    events: list[AuditEvent] = field(default_factory=list)
    sk: PrivateKey | None = None

    @property
    def head(self) -> bytes:
        return self.events[-1].event_hash if self.events else GENESIS

    def append(
        self, kind: str, payload: bytes, sk: PrivateKey | None = None
    ) -> AuditEvent:
        """Append a signed event.

        The robot signing key may be supplied per call or bound to the
        chain via `self.sk`. If neither is available the event is left
        unsigned (signature check will then fail for that event).
        """
        signer = sk if sk is not None else self.sk
        seq = len(self.events)
        prev = self.head
        if signer is not None:
            pk_bytes = bytes(signer.get_g1())
            probe = AuditEvent(seq, kind, payload, prev, pk_bytes, b"")
            sig = PopSchemeMPL.sign(signer, probe.signed_body)
            ev = AuditEvent(seq, kind, payload, prev, pk_bytes, bytes(sig))
        else:
            ev = AuditEvent(seq, kind, payload, prev)
        self.events.append(ev)
        return ev

    def verify_links(self) -> bool:
        prev = GENESIS
        for ev in self.events:
            if ev.prev_hash != prev:
                return False
            prev = ev.event_hash
        return True

    def verify_signatures(self, pk: G1Element) -> bool:
        """Verify every event's signature against the robot public key."""
        pk_bytes = bytes(pk)
        for ev in self.events:
            if not ev.signature or not ev.signer_pk:
                return False
            if ev.signer_pk != pk_bytes:
                return False
            try:
                sig = G2Element.from_bytes(ev.signature)
            except Exception:
                return False
            if not PopSchemeMPL.verify(pk, ev.signed_body, sig):
                return False
        return True

    def verify(self, pk: G1Element) -> bool:
        """Full chain verification: hash-pointer links + signatures."""
        return self.verify_links() and self.verify_signatures(pk)

    def admit_segment(
        self, pk: G1Element, prev_head: bytes | None
    ) -> list[AuditEvent]:
        """Validate this chain for epoch close; return the events to sign.

        The operator must not countersign material it has not checked, so
        this is the admission gate an epoch close runs per member. It
        raises `ValueError` naming the first fault rather than returning a
        flag, because a fleet root over an unchecked chain is worse than a
        failed close.

        Three things are established.

        * Hash-pointer integrity over every held event.
        * Continuity with what this operator already committed for the
          member. `prev_head` is the head the operator signed at the
          previous close, or None if it has never closed for this member.
          With no previous head the chain must start at GENESIS; with one,
          that head must appear in the replay, so a member cannot re-root
          its chain or drop history the operator already attested.
        * A valid BLS signature by `pk` on every event appended since
          `prev_head`. Events at or before it were verified at the close
          that committed it, so re-verifying the whole retained history at
          every epoch would be quadratic in history for no added
          assurance; the events returned are exactly the ones checked here.
        """
        pk_bytes = bytes(pk)
        events = self.events
        if not events:
            if prev_head is None or prev_head == GENESIS:
                return []
            raise ValueError(
                "chain is empty but the operator already committed a "
                "non-genesis head for this member"
            )

        # hash-pointer replay; running[i] is the head after i events
        running = [events[0].prev_hash]
        for ev in events:
            if ev.prev_hash != running[-1]:
                raise ValueError(
                    f"broken hash pointer at event seq {ev.seq}"
                )
            running.append(ev.event_hash)

        if prev_head is None:
            if running[0] != GENESIS:
                raise ValueError(
                    "chain does not start at GENESIS and no prior head "
                    "was committed for this member"
                )
            new_from = 0
        else:
            try:
                new_from = running.index(prev_head)
            except ValueError as exc:
                raise ValueError(
                    "chain does not continue from the head the operator "
                    "committed at the previous close"
                ) from exc

        fresh = events[new_from:]
        for ev in fresh:
            if not ev.signature or not ev.signer_pk:
                raise ValueError(f"unsigned event at seq {ev.seq}")
            if ev.signer_pk != pk_bytes:
                raise ValueError(
                    f"event seq {ev.seq} carries a key that is not the "
                    "member's roster key"
                )
            try:
                sig = G2Element.from_bytes(ev.signature)
            except Exception as exc:
                raise ValueError(
                    f"malformed signature at event seq {ev.seq}"
                ) from exc
            if not PopSchemeMPL.verify(pk, ev.signed_body, sig):
                raise ValueError(
                    f"invalid signature at event seq {ev.seq}"
                )
        return fresh


def _pair(a: bytes, b: bytes) -> bytes:
    return sha256(b"\x01" + a + b)


class MerkleTree:
    """Binary Merkle tree over a list of 32-byte leaves."""

    def __init__(self, leaves: list[bytes]):
        if not leaves:
            raise ValueError("MerkleTree requires at least one leaf")
        self.levels: list[list[bytes]] = [list(leaves)]
        cur = self.levels[0]
        while len(cur) > 1:
            nxt = [
                _pair(cur[i], cur[i + 1] if i + 1 < len(cur) else cur[i])
                for i in range(0, len(cur), 2)
            ]
            self.levels.append(nxt)
            cur = nxt

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def proof(self, index: int) -> list[tuple[bytes, bool]]:
        """Inclusion proof: list of (sibling, sibling_is_right)."""
        path: list[tuple[bytes, bool]] = []
        for level in self.levels[:-1]:
            sib = index ^ 1
            if sib >= len(level):
                sib = index  # duplicated odd node
            path.append((level[sib], sib > index))
            index //= 2
        return path

    @staticmethod
    def verify_proof(
        leaf: bytes, proof: list[tuple[bytes, bool]], root: bytes
    ) -> bool:
        cur = leaf
        for sibling, is_right in proof:
            cur = _pair(cur, sibling) if is_right else _pair(sibling, cur)
        return cur == root


@dataclass
class EpochClose:
    """Fleet root for one epoch, plus the exact leaf vector it commits to.

    `slots` is the ordered list of robot ids: slot index i denotes
    `slots[i]`, and `leaves[i]` is the Merkle leaf at that index. Callers
    must take the slot ordering from here rather than re-deriving it,
    because the ordering depends on whether identity-bound leaves were
    built (see `close_epoch`).
    """

    epoch: int
    fleet_root: bytes
    heads: dict[int, bytes]          # robot_id -> chain head at close
    build_time_ms: float
    slots: list[int] = field(default_factory=list)
    leaves: list[bytes] = field(default_factory=list)


LEAF_DOMAIN = b"ibsf/epochleaf/v1"


def epoch_leaf(epoch: int, slot: int, member_pk: bytes, head: bytes) -> bytes:
    """Merkle leaf binding an epoch slot to a member key and chain head.

    A leaf of the bare chain head would let a verifier confirm that
    *some* chain ended at that head without establishing *whose* it was,
    so slot-to-robot attribution would rest on out-of-band data. Binding
    the epoch, the slot index, and the member's public key into the leaf
    makes attribution part of what the operator's signature over the root
    covers (Section VI-C).
    """
    return sha256(
        LEAF_DOMAIN
        + epoch.to_bytes(8, "big")
        + slot.to_bytes(4, "big")
        + member_pk
        + head
    )


def close_epoch(
    epoch: int,
    chains: dict[int, AuditChain],
    member_pks: dict[int, bytes] | None = None,
) -> EpochClose:
    """Operator-side epoch close over identity-bound per-agent leaves.

    `member_pks` maps robot id to the member's roster public key. When
    supplied, each leaf commits to (epoch, slot, member key, chain head),
    so the signed root attributes every leaf to a named member, and the
    slots are ordered by member key. That ordering is what makes slot i
    coincide with position i of the operator-signed roster, which is
    sorted by key bytes, so a verifier can check the leaf's slot against
    the roster instead of trusting an out-of-band slot table.

    `member_pks` is optional only so that a caller holding no key map can
    still build a head-only tree, which carries no attribution guarantee;
    such a tree falls back to robot-id ordering. The operator path
    (`Operator.close_and_sign_epoch`) always supplies the map.
    """
    t0 = time.perf_counter()
    if member_pks is not None:
        robot_ids = sorted(chains, key=lambda rid: member_pks[rid])
        heads = {rid: chains[rid].head for rid in robot_ids}
        leaves = [
            epoch_leaf(epoch, slot, member_pks[rid], heads[rid])
            for slot, rid in enumerate(robot_ids)
        ]
    else:
        robot_ids = sorted(chains)
        heads = {rid: chains[rid].head for rid in robot_ids}
        leaves = [heads[rid] for rid in robot_ids]
    tree = MerkleTree(leaves)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return EpochClose(epoch, tree.root, heads, elapsed, robot_ids, leaves)
