"""Operator-side process: roster signing, epoch close, countersignature."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from blspy import G1Element, G2Element, PopSchemeMPL, PrivateKey

from . import bls
from .audit_chain import GENESIS, AuditChain, EpochClose, close_epoch
from .manifest import (
    CasteCertificate,
    IdentityManifest,
    caste_digest,
    sha256,
)

# v2: the signed roster payload now carries the epoch's quorum size.
ROSTER_DOMAIN = b"ibsf/roster/v2"
POLICY_DOMAIN = b"ibsf/rosterpolicy/v1"
EPOCH_DOMAIN = b"ibsf/epochroot/v2"

# The quorum fraction the paper states as its default. It lives here, next
# to roster signing, because the roster is where an operator's choice of
# quorum becomes authoritative; `federation` imports it rather than
# defining a second copy.
QUORUM_RATIO = 0.8


def default_quorum(n: int) -> int:
    """The stated default quorum size, t = ceil(QUORUM_RATIO * N).

    A default, not a constant of the construction: an operator may sign any
    threshold in 1..N for an epoch, and that signed value is the one the
    verifier enforces.
    """
    return math.ceil(QUORUM_RATIO * n)

# Link value carried by the first epoch a given operator ever closes.
EPOCH_LINK_GENESIS = GENESIS


@dataclass
class CommissioningStats:
    """Per-commissioning proof-of-possession verification timings (ms)."""

    pop_verify_ms: list[float] = field(default_factory=list)


CAPABILITY_DOMAIN = b"ibsf/capability/v1"


def capability_digest(cap: tuple[str, ...]) -> bytes:
    """Digest of a canonicalised capability set, length-prefixed.

    Length prefixing rather than delimiter joining, for the same reason
    the identity hash uses it: joining on a separator is not injective
    once a capability name may contain that separator, so two different
    certified surfaces could share one digest.
    """
    body = b"".join(
        len(e).to_bytes(4, "big") + e
        for e in (c.encode() for c in tuple(sorted(cap)))
    )
    return sha256(CAPABILITY_DOMAIN + body)


def roster_policy_digest(
    member_pks: tuple[bytes, ...],
    cap_digests: tuple[bytes, ...] = (),
    t_quorum: int = 0,
    caste_digests: tuple[bytes, ...] = (),
) -> bytes:
    """Epoch- and time-independent commitment to the roster's full policy.

    Each epoch root binds this commitment. The exporter deduplicates on it,
    so epochs sharing a policy can reference one embedded roster. Only
    epoch and t_open are excluded as per-epoch bookkeeping. An unchanged
    policy therefore requires a one-off O(N) roster attachment.

    The commitment includes membership, capabilities, caste assignments
    and quorum threshold. Membership alone cannot distinguish epochs with
    identical keys but different authority. A policy change must produce
    a distinct commitment and retain its own roster attachment.
    """
    order = sorted(range(len(member_pks)), key=lambda i: member_pks[i])

    def in_key_order(seq: tuple[bytes, ...]) -> bytes:
        # cap/caste digests are positionally aligned with member_pks, so
        # they must be permuted by the same order, not sorted themselves.
        return b"".join(seq[i] for i in order) if seq else b""

    return sha256(
        POLICY_DOMAIN
        + len(member_pks).to_bytes(4, "big")
        + b"".join(member_pks[i] for i in order)
        + len(cap_digests).to_bytes(4, "big")
        + in_key_order(cap_digests)
        + t_quorum.to_bytes(4, "big")
        + len(caste_digests).to_bytes(4, "big")
        + in_key_order(caste_digests)
    )


@dataclass(frozen=True)
class SignedRoster:
    """The operator's signed declaration of policy for one epoch.

    Signed by sk_O over the epoch index, ordered member keys, per-member
    capability digests, epoch open time, quorum threshold and caste digests:

        sigma_O( e || R^e || <H(cap_i)>_i || t_open^e || t^e
                 || <H(caste_i)>_i )

    The caste digests bind the assignments used to check certificate
    currency. t_open is a time, paired with t_close in SignedEpochRoot;
    t_quorum is the number of distinct signatures required for a decision.
    """

    epoch: int
    member_pks: tuple[bytes, ...]   # 48-byte G1 keys, sorted
    t_open: float                   # epoch open TIME, not a threshold
    signature: bytes                # 96-byte BLS signature by sk_O
    # Per-member capability digest, positionally aligned with member_pks.
    # Empty tuple means the roster was signed without capability binding,
    # which older exports may be.
    cap_digests: tuple[bytes, ...] = ()
    # Quorum size the operator fixed for this epoch. Zero means a roster
    # signed before the threshold was part of policy, which older exports
    # are; callers must treat zero as "unstated" rather than as "no
    # quorum required".
    t_quorum: int = 0
    # Per-member caste digest, positionally aligned with member_pks. This
    # is what makes a caste certificate's *currency* checkable: the
    # certificate proves the operator once issued that caste, and the
    # roster says which caste is in force this epoch. An empty tuple means
    # the roster was signed without caste binding.
    caste_digests: tuple[bytes, ...] = ()

    @property
    def digest(self) -> bytes:
        """Full roster digest over every component the operator signs.

        Position-aligned capability digests bind each key to the capabilities
        certified for it in the epoch. Signing only membership would allow a
        member to present unsupported capability claims.

        The quorum threshold is also signed, allowing an operator to require a
        larger quorum for an action and giving the verifier an authenticated
        policy against which to check a decision.
        """
        return sha256(
            ROSTER_DOMAIN
            + self.epoch.to_bytes(8, "big")
            + b"".join(self.member_pks)
            + b"".join(self.cap_digests)
            + int(self.t_open * 1000).to_bytes(8, "big")
            + self.t_quorum.to_bytes(4, "big")
            + b"".join(self.caste_digests)
        )

    def cap_digest_for(self, pk: bytes) -> bytes | None:
        """The capability digest the operator signed for `pk`, if any."""
        if not self.cap_digests:
            return None
        try:
            return self.cap_digests[self.member_pks.index(pk)]
        except ValueError:
            return None

    def caste_digest_for(self, pk: bytes) -> bytes | None:
        """The caste digest the operator signed for `pk` this epoch, if any."""
        if not self.caste_digests:
            return None
        try:
            return self.caste_digests[self.member_pks.index(pk)]
        except ValueError:
            return None

    @property
    def policy_digest(self) -> bytes:
        """Epoch-independent commitment to this roster's whole policy."""
        return roster_policy_digest(
            self.member_pks, self.cap_digests, self.t_quorum,
            self.caste_digests,
        )


@dataclass(frozen=True)
class SignedEpochRoot:
    """sigma_O(e || R^e || t_e || t_{e+1} || policy_digest || prev_root_digest).

    The field order matches epoch_root_payload. prev_root_digest is the
    payload digest of the preceding closed epoch, or 32 zero bytes for the
    first root. Binding this link makes an omitted epoch detectable even
    when every remaining root has a genuine operator signature.
    """

    epoch: int
    fleet_root: bytes
    t_open: float
    t_close: float
    roster_digest: bytes
    signature: bytes
    prev_root_digest: bytes = EPOCH_LINK_GENESIS

    @property
    def payload_digest(self) -> bytes:
        """The digest sk_O signs, and the link value the next root cites."""
        return epoch_root_payload(
            self.epoch, self.fleet_root, self.t_open, self.t_close,
            self.roster_digest, self.prev_root_digest,
        )


def epoch_root_payload(
    epoch: int,
    fleet_root: bytes,
    t_open: float,
    t_close: float,
    roster_digest: bytes,
    prev_root_digest: bytes,
) -> bytes:
    """Canonical epoch-root payload; also the chain link for the next root."""
    return sha256(
        EPOCH_DOMAIN
        + epoch.to_bytes(8, "big")
        + fleet_root
        + int(t_open * 1000).to_bytes(8, "big")
        + int(t_close * 1000).to_bytes(8, "big")
        + roster_digest
        + prev_root_digest
    )


@dataclass
class Operator:
    """Fleet operator holding the root signing key sk_O."""

    sk_o: PrivateKey
    rosters: dict[int, SignedRoster] = field(default_factory=dict)
    epoch_roots: dict[int, SignedEpochRoot] = field(default_factory=dict)
    # replay guard: (epoch, seq) decision tuples already accepted
    _seen: set[tuple[int, int]] = field(default_factory=set)
    current_epoch: int = 0
    # per-member head committed at the last close, keyed by member key,
    # so chain continuity is checked against what was actually attested
    _last_head: dict[bytes, bytes] = field(default_factory=dict)
    # payload digest of the most recently closed epoch (the link value)
    _last_root_digest: bytes = EPOCH_LINK_GENESIS
    # deserialized roster keys, cached across closes
    _pk_cache: dict[bytes, G1Element] = field(default_factory=dict)

    def _pk(self, pk_bytes: bytes) -> G1Element:
        el = self._pk_cache.get(pk_bytes)
        if el is None:
            el = G1Element.from_bytes(pk_bytes)
            self._pk_cache[pk_bytes] = el
        return el

    @classmethod
    def create(cls, seed: bytes) -> "Operator":
        return cls(sk_o=bls.keygen(b"operator/" + seed))

    @property
    def pk_o(self) -> G1Element:
        return self.sk_o.get_g1()

    @property
    def root_cert_digest(self) -> bytes:
        """Stands in for SHA-256 of the operator X.509 root cert."""
        return sha256(b"ibsf/rootcert/v1" + bytes(self.pk_o))

    def sign_roster(
        self,
        epoch: int,
        members: list[tuple[bytes, bytes]],
        t_open: float,
        stats: "CommissioningStats | None" = None,
        capabilities: dict[bytes, tuple[str, ...]] | None = None,
        t_quorum: int | None = None,
        castes: dict[bytes, str] | None = None,
    ) -> SignedRoster:
        """Sign an epoch roster, enforcing proof-of-possession per member.

        `t_quorum` is the quorum size the operator fixes for this epoch. It
        is a parameter rather than a constant because the paper describes it
        as operator policy; passing None takes the stated default, which is
        `ceil(QUORUM_RATIO * N)` over the members admitted here. Whatever
        value is used is signed, and `verify_subset` will only accept a
        decision presented under that same value.

        `members` is a list of (pk, pop_proof) pairs. Each key must carry
        a valid PoP (rogue-key defence, Section V-D): a key whose PoP does
        not verify cannot be produced without its secret, so an adversary
        cannot inject a rogue key derived from honest keys. Only PoP-valid
        keys are admitted to the signed roster; any failure aborts.

        Distinctness is enforced as well as possession. A PoP-valid key
        repeated across several roster slots would let one member occupy
        t positions and satisfy a t-of-N quorum on its own, since the
        aggregate over a repeated key is satisfied by repeating that
        member's signature. The roster therefore rejects duplicates
        outright, so roster positions and distinct members coincide.
        """
        seen: set[bytes] = set()
        verified_pks: list[bytes] = []
        for pk_bytes, proof_bytes in members:
            if pk_bytes in seen:
                raise ValueError(
                    "duplicate public key at commissioning; roster "
                    "positions must correspond to distinct members"
                )
            seen.add(pk_bytes)
            try:
                pk = G1Element.from_bytes(pk_bytes)
                proof = G2Element.from_bytes(proof_bytes)
            except Exception as exc:
                raise ValueError(
                    f"malformed key/proof at commissioning: {exc}"
                ) from exc
            ok, ms = bls.pop_verify(pk, proof)
            if stats is not None:
                stats.pop_verify_ms.append(ms)
            if not ok:
                raise ValueError(
                    "proof-of-possession failed for commissioned key; "
                    "key rejected (rogue-key defence)"
                )
            verified_pks.append(pk_bytes)

        caps_by_pk = capabilities or {}
        castes_by_pk = castes or {}
        pks = tuple(sorted(verified_pks))
        digests = tuple(
            capability_digest(caps_by_pk.get(pk, ())) for pk in pks
        )
        # Caste is re-declared every epoch, which is what supersedes an
        # earlier assignment: a robot demoted at the boundary carries the
        # new digest here, so its old certificate no longer matches what
        # the operator signed for the epoch it is presented in.
        caste_digs = (
            tuple(caste_digest(castes_by_pk.get(pk, "")) for pk in pks)
            if castes_by_pk
            else ()
        )
        # Resolved here rather than left to the verifier, so that the value
        # the operator signs is the value the verifier must reproduce.
        t = default_quorum(len(pks)) if t_quorum is None else int(t_quorum)
        if not 1 <= t <= len(pks):
            raise ValueError(
                f"quorum threshold {t} outside 1..{len(pks)} for this roster"
            )
        payload = sha256(
            ROSTER_DOMAIN
            + epoch.to_bytes(8, "big")
            + b"".join(pks)
            + b"".join(digests)
            + int(t_open * 1000).to_bytes(8, "big")
            + t.to_bytes(4, "big")
            + b"".join(caste_digs)
        )
        sig = PopSchemeMPL.sign(self.sk_o, payload)
        roster = SignedRoster(
            epoch, pks, t_open, bytes(sig), digests, t, caste_digs
        )
        self.rosters[epoch] = roster
        self.current_epoch = epoch
        return roster

    def roster_contains(self, epoch: int, pk: bytes) -> bool:
        roster = self.rosters.get(epoch)
        return roster is not None and pk in roster.member_pks

    def check_and_record_decision(self, epoch: int, seq: int) -> bool:
        """Replay guard: accept a decision tuple's (epoch, seq) at most once.

        Rejects any tuple whose epoch is not the operator's current epoch,
        or whose (epoch, seq) was already seen (whole-tuple replay). On a
        first, in-epoch acceptance the tuple is recorded and True returned.
        """
        if epoch != self.current_epoch:
            return False
        if (epoch, seq) in self._seen:
            return False
        self._seen.add((epoch, seq))
        return True

    def issue_caste(
        self, robot_pk: bytes, caste: str, issued_at: float
    ) -> CasteCertificate:
        cert = CasteCertificate(robot_pk, caste, issued_at, b"")
        sig = PopSchemeMPL.sign(self.sk_o, cert.signed_payload)
        return CasteCertificate(robot_pk, caste, issued_at, bytes(sig))

    def revoke_member(
        self,
        epoch: int,
        prior_members: list[tuple[bytes, bytes]],
        revoked_pk: bytes,
        t_open: float,
        capabilities: dict[bytes, tuple[str, ...]] | None = None,
        castes: dict[bytes, str] | None = None,
        t_quorum: int | None = None,
    ) -> SignedRoster:
        """Remove a member by signing the next roster without it.

        Revocation in this construction is a roster operation, not a
        separate revocation list: a member is revoked precisely by being
        absent from the roster the operator signs for the next epoch, and
        `verify_manifest` already refuses a manifest resolved against any
        epoch but the current one, so a revoked key stops verifying at the
        boundary. This makes the exposure bound explicit and honest: a
        member detected compromised mid-epoch retains quorum standing until
        the next roster is signed, i.e. for up to one epoch length. A
        deployment needing tighter revocation shortens the epoch, at the
        certificate-growth cost quantified in the evaluation.

        Returns the new roster. Raises if `revoked_pk` is not among
        `prior_members`, so a caller cannot believe it revoked a key that
        was never present.
        """
        if revoked_pk not in {pk for pk, _ in prior_members}:
            raise ValueError(
                "revoked key is not in the prior member set; nothing to remove"
            )
        surviving = [(pk, proof) for pk, proof in prior_members if pk != revoked_pk]
        caps = {k: v for k, v in (capabilities or {}).items() if k != revoked_pk}
        cst = {k: v for k, v in (castes or {}).items() if k != revoked_pk}
        return self.sign_roster(
            epoch, surviving, t_open=t_open,
            capabilities=caps or None,
            castes=cst or None,
            t_quorum=t_quorum,
        )

    def verify_caste(
        self,
        cert: CasteCertificate,
        roster: SignedRoster,
        expected_epoch: int,
    ) -> bool:
        """Check a caste claim for both issuance and currency.

        Issuance requires the operator's signature on the caste certificate.
        Currency requires the caste assigned to that key in the current epoch.
        A superseded certificate remains validly signed, so issuance alone
        cannot detect replay after demotion.

        The operator-signed epoch roster supplies the current caste digest.
        Validate its signature under root_O and require expected_epoch before
        using its contents. Otherwise a forged roster could support a stale
        certificate. A caste change takes effect at the next epoch boundary,
        so the exposure window is one epoch, as for roster membership.
        """
        if roster.epoch != expected_epoch:
            return False
        # The roster must be the operator's own before its caste digests
        # can testify to anything.
        try:
            rsig = G2Element.from_bytes(roster.signature)
        except Exception:
            return False
        if not PopSchemeMPL.verify(self.pk_o, roster.digest, rsig):
            return False
        try:
            sig = G2Element.from_bytes(cert.operator_signature)
        except Exception:
            return False
        if not PopSchemeMPL.verify(self.pk_o, cert.signed_payload, sig):
            return False
        signed_caste = roster.caste_digest_for(cert.robot_pk)
        if signed_caste is None:
            # The roster names no caste for this key: either the key is not
            # a member this epoch, or the roster predates caste binding.
            # Neither is a basis for honouring a caste claim.
            return False
        return signed_caste == caste_digest(cert.caste_name)

    def close_and_sign_epoch(
        self,
        epoch: int,
        chains: dict[int, AuditChain],
        t_open: float,
        t_close: float,
        member_pks: dict[int, bytes],
    ) -> tuple[SignedEpochRoot, EpochClose, float]:
        """Close an epoch with a Merkle root over identity-bound leaves.

        member_pks maps each robot id in chains to its public key in the signed
        roster. Each leaf binds epoch, slot, member key and chain head. The
        supplied keys must equal the roster's member set without repeats; both
        are ordered by key bytes, allowing the verifier to check each slot.

        Before signing, validate each chain's hash pointers, continuity with
        the operator's previously committed head, and the member signature on
        every new event. Unsigned, malformed or wrong-key material aborts the
        close with ValueError.

        Bind the epoch-independent full policy digest, allowing unchanged
        policies to share one O(N) roster attachment instead of O(Nk) repeated
        rosters. A changed quorum threshold or other policy field must require
        a distinct attachment. Also bind the preceding epoch payload digest,
        so omission of an epoch breaks the root sequence.
        """
        roster = self.rosters[epoch]
        missing = sorted(rid for rid in chains if rid not in member_pks)
        if missing:
            raise ValueError(
                f"no roster public key supplied for robot ids {missing}; "
                "epoch leaves must be identity-bound"
            )
        supplied = [member_pks[rid] for rid in chains]
        if len(set(supplied)) != len(supplied):
            raise ValueError(
                "duplicate public key across closing chains; roster "
                "slots must correspond to distinct members"
            )
        if set(supplied) != set(roster.member_pks):
            raise ValueError(
                "closing chains do not match the signed roster for epoch "
                f"{epoch}; slot indices would not correspond to roster "
                "positions"
            )
        # admission gate: nothing is signed until every chain checks out
        for rid, chain in chains.items():
            pk_bytes = member_pks[rid]
            try:
                chain.admit_segment(self._pk(pk_bytes), self._last_head.get(pk_bytes))
            except ValueError as exc:
                raise ValueError(
                    f"epoch {epoch} close refused: chain for robot {rid} "
                    f"failed admission ({exc})"
                ) from exc

        closed = close_epoch(
            epoch, chains, {rid: member_pks[rid] for rid in chains}
        )
        mdigest = roster.policy_digest
        prev_link = self._last_root_digest
        t0 = time.perf_counter()
        payload = epoch_root_payload(
            epoch, closed.fleet_root, t_open, t_close, mdigest, prev_link
        )
        sig = PopSchemeMPL.sign(self.sk_o, payload)
        sign_ms = (time.perf_counter() - t0) * 1000.0
        root = SignedEpochRoot(
            epoch, closed.fleet_root, t_open, t_close, mdigest, bytes(sig),
            prev_link,
        )
        self.epoch_roots[epoch] = root
        for rid, pk_bytes in member_pks.items():
            if rid in chains:
                self._last_head[pk_bytes] = closed.heads[rid]
        self._last_root_digest = payload
        return root, closed, sign_ms


def verify_manifest(
    pk_o: G1Element,
    roster: SignedRoster,
    manifest: IdentityManifest,
    root_cert_digest: bytes,
    expected_epoch: int,
) -> bool:
    """Member-side admission check for a presented identity manifest.

    Require a roster signed under the trusted operator public key, member
    key inclusion, and a matching operator root pointer. Membership alone
    would permit an adversarial key list; signature verification without a
    root pointer check would permit a different operator's member.

    A roster remains validly signed after expiry. The verifying member
    supplies expected_epoch from its view of the current epoch, for example
    its last countersigned decision. Refuse any other epoch before using
    the roster, preventing replay of a removed member's old admission.

    Compare the claimed capability digest with the signed digest for that
    key and epoch. Key membership alone does not authenticate capabilities.

    This function supplies the admission predicate. The reference decision
    path does not consume manifests: members sign the operator's decision
    message, with no peer role-request flow. Ensuring that deployment
    requests pass through this predicate remains a deployment obligation.
    """
    if roster.epoch != expected_epoch:
        return False
    if manifest.root_o != root_cert_digest:
        return False
    if manifest.pk not in roster.member_pks:
        return False
    signed_caps = roster.cap_digest_for(manifest.pk)
    if signed_caps is None:
        return False
    if signed_caps != capability_digest(manifest.cap):
        return False
    try:
        sig = G2Element.from_bytes(roster.signature)
    except Exception:
        return False
    return PopSchemeMPL.verify(pk_o, roster.digest, sig)


def verify_epoch_root(pk_o: G1Element, root: SignedEpochRoot) -> bool:
    """Regulator-side check using only the operator public key.

    The epoch identifier is inside the signed payload, so a signed root
    cannot be relabelled to a different epoch, duplicated under a new
    label, or reordered without invalidating the signature. The previous
    root's digest is inside it too, so the linkage a verifier checks
    across a sequence of roots is itself operator-signed.
    """
    try:
        sig = G2Element.from_bytes(root.signature)
    except Exception:
        return False
    return PopSchemeMPL.verify(pk_o, root.payload_digest, sig)
