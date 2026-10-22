"""Identity manifest and caste certificate (Definition 1, Section IV)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def sha256(data: bytes) -> bytes:
    """SHA-256 digest."""
    return hashlib.sha256(data).digest()


# Every payload signed under sk_O carries a distinct constant prefix, so
# that a signature produced for one message family cannot be replayed as
# one from another. The caste certificate was the one family without a
# separator: no collision was reachable, because a compressed G1 point
# begins with a byte >= 0x80 while every other prefix is ASCII, but the
# invariant was unstated and unguarded, and a later payload family
# beginning with attacker-influenced bytes would have reopened it.
CASTE_DOMAIN = b"ibsf/caste/v1"
CASTE_DIGEST_DOMAIN = b"ibsf/castedigest/v1"


def caste_digest(caste_name: str) -> bytes:
    """Digest of a caste name, as bound into the epoch roster.

    Length-prefixed for the same injectivity reason as the capability
    digest: a bare concatenation is not injective once a name may contain
    whatever character is used to join.
    """
    body = caste_name.encode()
    return sha256(CASTE_DIGEST_DOMAIN + len(body).to_bytes(4, "big") + body)


@dataclass(frozen=True)
class IdentityManifest:
    """Per-robot identity manifest M_i = <pk_i, root_O, cap_i, audit_i>.

    The identity hash covers pk, operator root pointer, and the
    canonicalised capability set. The caste certificate and the
    audit-chain head are referenced but NOT part of the hash, so the
    hash is byte-stable across caste change and audit growth.
    """

    pk: bytes              # 48-byte BLS12-381 G1 public key (min-pk)
    root_o: bytes          # 32-byte SHA-256 of operator root certificate
    cap: tuple[str, ...]   # canonicalised, sorted capability names
    audit_head: bytes      # 32-byte SHA-256 head of agent's audit chain

    def __post_init__(self) -> None:
        if len(self.pk) != 48:
            raise ValueError(f"pk must be 48 bytes, got {len(self.pk)}")
        if len(self.root_o) != 32:
            raise ValueError("root_o must be a 32-byte digest")
        if tuple(sorted(self.cap)) != self.cap:
            raise ValueError("cap must be sorted (canonical form)")

    @property
    def identity_hash(self) -> bytes:
        """h_i = H(pk_i || root_O || len-prefixed(cap_i)). Caste excluded.

        Capability names are serialised length-prefixed rather than
        joined on a delimiter character. Delimiter joining is not
        injective when a name may contain the delimiter: ("a", "b") and
        ("a\\x1fb",) would hash identically, so two different certified
        capability sets would share one identity. Length prefixing makes
        the encoding injective for arbitrary names, with no reserved
        characters to police.
        """
        cap_bytes = b"".join(
            len(e).to_bytes(4, "big") + e
            for e in (c.encode() for c in self.cap)
        )
        return sha256(self.pk + self.root_o + cap_bytes)


@dataclass(frozen=True)
class CasteCertificate:
    """Operator-issued caste assignment, separate from identity hash.

    The signature authenticates *issuance*: this operator did assign this
    caste to this key at this time. It does not by itself authenticate
    *currency*, because a certificate the operator later supersedes stays
    validly signed forever. Freshness comes from the epoch roster, which
    carries a per-member caste digest the operator re-signs each epoch;
    `Operator.verify_caste` checks the certificate against that roster
    rather than against the signature alone. An earlier revision checked
    only the signature, so a robot demoted from `supervisor` to
    `observer` could present its old, genuinely signed `supervisor`
    certificate indefinitely.
    """

    robot_pk: bytes        # 48-byte member public key
    caste_name: str
    issued_at: float       # epoch seconds (simulated clock)
    operator_signature: bytes  # 96-byte BLS signature by sk_O

    @property
    def signed_payload(self) -> bytes:
        return (
            CASTE_DOMAIN
            + self.robot_pk
            + len(self.caste_name.encode()).to_bytes(4, "big")
            + self.caste_name.encode()
            + int(self.issued_at * 1000).to_bytes(8, "big")
        )
