"""N-party BLS aggregation with operator countersignature (Algorithm 1).

Instantiated on BLS12-381 in the min-pk parameterisation (public keys
in G1, signatures in G2) via the blst-backed `blspy` binding. The
protocol of Algorithm 1 in the paper is convention-agnostic; only the
group assignment differs from the min-sig presentation.

Rogue-key resistance comes from mandatory proof-of-possession at
commissioning (Section V-D): `Operator.sign_roster` refuses to admit a
key whose PoP does not verify, so no adversary-structured key ever
reaches the aggregation set.

The countersignature binds four things, and verification recomputes all
of them, so none can be substituted by a caller:
  * the signed message `msg` (which itself binds epoch and sequence),
  * the aggregate signature `Sigma`,
  * the participant bitmap (which subset of the roster attested), and
  * the KEY-LIST digest for the epoch (the ordered member keys, the
    epoch index and the quorum threshold), so the online decision path
    authenticates who may sign and how many rather than trusting
    caller-supplied keys. This is deliberately narrower than the epoch
    policy the epoch root commits: capabilities and castes are not on
    the online path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from blspy import G1Element, G2Element, PopSchemeMPL, PrivateKey

from .manifest import sha256

AUX_DOMAIN = b"ibsf/countersig/v2"
# v2: the digest now covers the operator's quorum threshold as well as the
# epoch and the ordered key list. See key_list_digest.
ROSTER_DIGEST_DOMAIN = b"ibsf/rosterdigest/v2"


@dataclass(frozen=True)
class FleetDecision:
    """Output of Sign: <Sigma, sigma_O, roster epoch tag>."""

    msg: bytes
    aggregate: bytes        # 96-byte aggregate member signature Sigma
    operator_sig: bytes     # 96-byte operator countersignature sigma_O
    epoch: int              # roster epoch the aggregate was formed under


@dataclass
class Timings:
    """Wall-clock components of one decision, milliseconds."""

    member_sign_ms: list[float] = field(default_factory=list)
    aggregate_ms: float = 0.0
    countersign_ms: float = 0.0
    verify_ms: float = 0.0


def key_list_digest(
    roster_pks: list[G1Element], epoch: int, t_quorum: int
) -> bytes:
    """Digest binding an epoch to its ordered key list and quorum threshold.

    The verifier recomputes the digest from the epoch, key list and quorum
    threshold. Binding the threshold authenticates the operator's policy:
    a decision formed under one threshold cannot verify under another,
    because the recomputed auxiliary digest differs and the operator
    countersignature fails. The threshold is not an out-of-band constant.
    """
    body = b"".join(bytes(pk) for pk in roster_pks)
    return sha256(
        ROSTER_DIGEST_DOMAIN
        + epoch.to_bytes(8, "big")
        + t_quorum.to_bytes(4, "big")
        + body
    )


def _aux(agg: G2Element, participant_bitmap: bytes, rdigest: bytes) -> bytes:
    """Auxiliary binding: aggregate, participant set, and key-list digest."""
    return sha256(AUX_DOMAIN + bytes(agg) + participant_bitmap + rdigest)


def aux_binding(
    agg: G2Element,
    participant_bitmap: bytes,
    roster_pks: list[G1Element],
    epoch: int,
    t_quorum: int,
) -> bytes:
    """Public form of the countersignature aux binding.

    Exposed so that an analysis harness can decompose a verification
    verdict into its member half and its operator half without reaching
    into module internals.
    """
    return _aux(
        agg, participant_bitmap, key_list_digest(roster_pks, epoch, t_quorum)
    )


def member_sign(sk: PrivateKey, msg: bytes) -> tuple[G2Element, float]:
    """Per-robot signature share; returns (sig, elapsed ms)."""
    t0 = time.perf_counter()
    sig = PopSchemeMPL.sign(sk, msg)
    return sig, (time.perf_counter() - t0) * 1000.0


def aggregate(sigs: list[G2Element]) -> tuple[G2Element, float]:
    """Aggregate member shares by addition in G2; returns (Sigma, ms)."""
    t0 = time.perf_counter()
    agg = PopSchemeMPL.aggregate(sigs)
    return agg, (time.perf_counter() - t0) * 1000.0


def bitmap_to_indices(bitmap: bytes, roster_size: int) -> list[int]:
    """Decode a participant bitmap into sorted roster indices."""
    idx = []
    for i in range(roster_size):
        if bitmap[i // 8] & (1 << (i % 8)):
            idx.append(i)
    return idx


def countersign_subset(
    sk_o: PrivateKey,
    msg: bytes,
    agg: G2Element,
    participant_bitmap: bytes,
    roster_pks: list[G1Element],
    epoch: int,
    t_quorum: int,
) -> tuple[G2Element, float]:
    """Operator countersignature binding aggregate, subset, and key list.

    The aux field commits to the aggregate, the participating subset
    S^e of the epoch roster, and the epoch's KEY-LIST digest (ordered
    roster keys, the epoch index and the quorum threshold in force), so a
    verifier reconstructs
    exactly which members of exactly which roster attested the decision,
    and under which threshold.
    """
    t0 = time.perf_counter()
    aux = _aux(
        agg, participant_bitmap, key_list_digest(roster_pks, epoch, t_quorum)
    )
    sig_o = PopSchemeMPL.sign(sk_o, msg + aux)
    return sig_o, (time.perf_counter() - t0) * 1000.0


def verify_subset(
    msg: bytes,
    agg: G2Element,
    sig_o: G2Element,
    roster_pks: list[G1Element],
    participant_bitmap: bytes,
    pk_o: G1Element,
    threshold: int,
    epoch: int,
) -> tuple[bool, float]:
    """t-of-N verification, resolving participants from the roster itself.

    The verifier reconstructs the participating set S by selecting roster
    keys at the bitmap indices, so a prover cannot substitute
    caller-supplied keys it controls (Algorithm 1, Verify step 1). Any
    bitmap bit outside the roster range is rejected. The roster itself is
    authenticated in this path: the operator countersignature commits to
    `key_list_digest(roster_pks, epoch, threshold)`, so presenting a
    substituted key list, or claiming a threshold other than the one the
    operator signed for the epoch, makes the recomputed aux differ and the
    countersignature check fail. `threshold` is therefore not a value a
    caller may choose freely: passing anything but the operator's signed
    threshold fails verification rather than weakening it.
    """
    t0 = time.perf_counter()
    n_roster = len(roster_pks)
    n_bytes_needed = (n_roster + 7) // 8
    # A well-formed bitmap over an N-member roster is exactly ceil(N/8)
    # bytes. Anything shorter is rejected before it is indexed: this is a
    # public verifier entry point reachable by an untrusted prover, and
    # indexing a truncated bitmap would raise rather than return False,
    # turning a malformed input into a crash of the verifying process.
    if len(participant_bitmap) != n_bytes_needed:
        return False, (time.perf_counter() - t0) * 1000.0
    # reject bitmap bits set beyond the roster (out-of-range membership).
    # Trailing whole bytes cannot exist: the length check above pins the
    # bitmap to exactly ceil(N/8) bytes, so only the high bits of the
    # final byte can encode a non-existent member.
    high_ok = True
    if n_bytes_needed and n_roster % 8:
        high_ok = (participant_bitmap[n_bytes_needed - 1] >> (n_roster % 8)) == 0
    idx = bitmap_to_indices(participant_bitmap, n_roster)
    participant_pks = [roster_pks[i] for i in idx]
    # Count DISTINCT keys toward the quorum, not bitmap positions. The
    # operator already refuses duplicate keys at commissioning; checking
    # again here means a verifier presented with a malformed roster still
    # cannot be convinced that one member repeated across t slots is a
    # t-member quorum.
    distinct = len({bytes(pk) for pk in participant_pks})
    quorum_ok = distinct >= threshold and distinct == len(participant_pks)
    ok_members = (
        high_ok
        and quorum_ok
        and len(participant_pks) > 0
        and PopSchemeMPL.fast_aggregate_verify(participant_pks, msg, agg)
    )
    aux = _aux(
        agg, participant_bitmap, key_list_digest(roster_pks, epoch, threshold)
    )
    ok_operator = PopSchemeMPL.verify(pk_o, msg + aux, sig_o)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return bool(ok_members and ok_operator), elapsed


def pop_prove(sk: PrivateKey) -> tuple[G2Element, float]:
    """Proof of possession generated by a robot at commissioning."""
    t0 = time.perf_counter()
    proof = PopSchemeMPL.pop_prove(sk)
    return proof, (time.perf_counter() - t0) * 1000.0


def pop_verify(pk: G1Element, proof: G2Element) -> tuple[bool, float]:
    """Operator-side PoP check before a key is admitted to the roster."""
    t0 = time.perf_counter()
    ok = PopSchemeMPL.pop_verify(pk, proof)
    return ok, (time.perf_counter() - t0) * 1000.0


def keygen(seed: bytes) -> PrivateKey:
    """Deterministic key generation from a 32-byte seed."""
    return PopSchemeMPL.key_gen(sha256(seed))
