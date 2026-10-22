"""Compliance certificate export and regulator-side verification."""

from __future__ import annotations

import json
from dataclasses import dataclass

from blspy import G1Element, G2Element, PopSchemeMPL

from .manifest import sha256
from .operator import (
    EPOCH_LINK_GENESIS,
    ROSTER_DOMAIN,
    Operator,
    SignedEpochRoot,
    SignedRoster,
    roster_policy_digest,
    verify_epoch_root,
)


def _roster_policy_key(roster: SignedRoster) -> bytes:
    """Policy identity of a roster: the digest the epoch root commits to.

    Independent of epoch and open time, so k epochs that share the same
    policy deduplicate to a single embedded roster (Section VIII-D: the
    roster attachment is a one-off O(N) component for static policy).

    Deduplicating on the member set alone was wrong. It made two epochs
    with the same members but different signed quorum thresholds share a
    key, so the later epoch's threshold was dropped from the document
    while verification still passed.
    """
    return roster_policy_digest(
        roster.member_pks, roster.cap_digests, roster.t_quorum,
        roster.caste_digests,
    )


def _roster_digest(roster: SignedRoster) -> bytes:
    """Recompute the full digest sk_O signed over this roster.

    This deliberately duplicates `SignedRoster.digest` rather than
    calling it: a verifier that re-derives the signed bytes from the
    fields it was handed cannot be fooled by an object whose property
    was overridden. The consequence is that the two must be kept in
    step, and the per-member capability digests, the quorum threshold and
    the per-member caste digests are all part of what the operator signs.
    Keeping them in step is not automatic: adding caste binding to
    `sign_roster` without adding it here made every exported certificate
    fail to verify, which is the failure mode this duplication is meant
    to have -- loud, and caught by the certificate test.
    """
    return sha256(
        ROSTER_DOMAIN
        + roster.epoch.to_bytes(8, "big")
        + b"".join(roster.member_pks)
        + b"".join(roster.cap_digests)
        + int(roster.t_open * 1000).to_bytes(8, "big")
        + roster.t_quorum.to_bytes(4, "big")
        + b"".join(roster.caste_digests)
    )


COVERAGE_DOMAIN = b"ibsf/certcoverage/v1"


@dataclass(frozen=True)
class CoverageAttestation:
    """Operator's signed statement of what this certificate covers.

    Chaining each epoch root to its predecessor makes an omitted epoch
    detectable in the middle of a document and at its start, because the
    citation it left behind has nothing to satisfy it. It does not make a
    TRUNCATED document detectable: epochs 0..2 of a fleet that has closed
    0..9 form a perfectly well-formed chain, and nothing inside the
    document says otherwise. An operator wishing to hide recent activity
    would simply stop early.

    So the operator is made to say, under its own key, how far its epoch
    sequence had actually run when the document was issued. Verification
    then requires the roots to be contiguous and to end exactly at
    `latest_epoch`, which turns truncation from an undetectable omission
    into a signed false statement, attributable to the operator and
    falsifiable against any later root a regulator obtains by another
    route.

    We are explicit about the residual: this binds the operator to a
    claim, it does not prove the claim. An operator that closed epochs it
    never told anyone about can attest a `latest_epoch` consistent with
    that concealment. Detecting THAT needs an external observer of the
    epoch stream, which is outside what a single-operator document can
    establish.
    """

    latest_epoch: int
    issued_at: float
    signature: bytes
    # Payload digest of the root at `latest_epoch`. Without it the
    # attestation named an epoch INDEX but no particular chain, so one
    # operator running two fleets whose documents end at the same index
    # could satisfy fleet A's attestation with fleet B's document. Naming
    # the terminal root makes the statement "my sequence ran to epoch k"
    # into "my sequence ran to THIS root at epoch k".
    terminal_root_digest: bytes = b""

    @property
    def payload(self) -> bytes:
        return sha256(
            COVERAGE_DOMAIN
            + self.latest_epoch.to_bytes(8, "big")
            + int(self.issued_at * 1000).to_bytes(8, "big")
            + self.terminal_root_digest
        )


@dataclass(frozen=True)
class ComplianceCertificate:
    """Single-document certificate: O(k) epoch roots + policy digests.

    Full rosters are embedded once per distinct POLICY (not per epoch),
    so for a policy that is static across k epochs the document holds one
    O(N) roster attachment plus O(k) epoch-root entries. A policy change
    (membership, capabilities, quorum threshold or castes) forces its own
    attachment, which is what keeps every epoch's governing policy
    present in the document. Merkle
    proofs for individual events are appended on demand and are not part
    of the base export.
    """

    operator_pk: bytes
    epoch_roots: tuple[SignedEpochRoot, ...]
    rosters: tuple[SignedRoster, ...]   # deduplicated by policy digest
    coverage: "CoverageAttestation | None" = None

    def to_json(self, embed_rosters: bool = True) -> bytes:
        doc = {
            "format": "ibsf-compliance-cert/v1",
            "operator_pk": self.operator_pk.hex(),
            "epochs": [
                {
                    "epoch": r.epoch,
                    "fleet_root": r.fleet_root.hex(),
                    "t_open": r.t_open,
                    "t_close": r.t_close,
                    "roster_digest": r.roster_digest.hex(),
                    "prev_root_digest": r.prev_root_digest.hex(),
                    "signature": r.signature.hex(),
                }
                for r in self.epoch_roots
            ],
        }
        if self.coverage is not None:
            doc["coverage"] = {
                "latest_epoch": self.coverage.latest_epoch,
                "issued_at": self.coverage.issued_at,
                "terminal_root_digest":
                    self.coverage.terminal_root_digest.hex(),
                "signature": self.coverage.signature.hex(),
            }
        if embed_rosters:
            doc["rosters"] = [
                {
                    "epoch": r.epoch,
                    "digest": r.digest.hex(),
                    "member_pks": [pk.hex() for pk in r.member_pks],
                    "cap_digests": [d.hex() for d in r.cap_digests],
                    "t_open": r.t_open,
                    "t_quorum": r.t_quorum,
                    "caste_digests": [d.hex() for d in r.caste_digests],
                    "signature": r.signature.hex(),
                }
                for r in self.rosters
            ]
        return json.dumps(doc, separators=(",", ":")).encode()

    @classmethod
    def from_json(cls, blob: bytes) -> "ComplianceCertificate":
        """Reconstruct a certificate from its serialized JSON form.

        Used by an external verifier that loads only the exported
        document (no live fleet, no secret keys).
        """
        doc = json.loads(blob)
        if doc.get("format") != "ibsf-compliance-cert/v1":
            raise ValueError("unrecognised certificate format")
        roots = tuple(
            SignedEpochRoot(
                epoch=e["epoch"],
                fleet_root=bytes.fromhex(e["fleet_root"]),
                t_open=e["t_open"],
                t_close=e["t_close"],
                roster_digest=bytes.fromhex(e["roster_digest"]),
                signature=bytes.fromhex(e["signature"]),
                prev_root_digest=bytes.fromhex(e["prev_root_digest"]),
            )
            for e in doc["epochs"]
        )
        rosters = tuple(
            SignedRoster(
                epoch=r["epoch"],
                member_pks=tuple(bytes.fromhex(pk) for pk in r["member_pks"]),
                t_open=r["t_open"],
                signature=bytes.fromhex(r["signature"]),
                cap_digests=tuple(
                    bytes.fromhex(d) for d in r.get("cap_digests", [])
                ),
                # Absent in documents exported before the quorum size was
                # part of the signed payload; zero reproduces the digest
                # those documents were signed over, so they still verify.
                t_quorum=r.get("t_quorum", 0),
                # Likewise for caste binding: an empty tuple reproduces the
                # digest a pre-caste document was signed over.
                caste_digests=tuple(
                    bytes.fromhex(d) for d in r.get("caste_digests", [])
                ),
            )
            for r in doc.get("rosters", [])
        )
        cov_doc = doc.get("coverage")
        coverage = (
            CoverageAttestation(
                latest_epoch=cov_doc["latest_epoch"],
                issued_at=cov_doc["issued_at"],
                terminal_root_digest=bytes.fromhex(
                    cov_doc.get("terminal_root_digest", "")
                ),
                signature=bytes.fromhex(cov_doc["signature"]),
            )
            if cov_doc is not None
            else None
        )
        return cls(
            bytes.fromhex(doc["operator_pk"]), roots, rosters, coverage
        )


def export_certificate(
    operator: Operator, epochs: list[int], issued_at: float = 0.0
) -> ComplianceCertificate:
    """Export a certificate over the given epochs.

    Embedded rosters are deduplicated by policy digest: epochs sharing
    the same policy contribute one roster attachment. For each distinct
    policy we keep the earliest-epoch roster; verification recomputes the
    policy digest per epoch from whichever embedded roster matches the
    commitment in that epoch's root.

    The export also carries a coverage attestation stating, under the
    operator's key, the highest epoch the operator had closed at issue
    time. Verification requires the document to run contiguously up to
    that epoch, so a truncated document is a signed false statement
    rather than a well-formed one (see `CoverageAttestation`).
    """
    roots = tuple(operator.epoch_roots[e] for e in epochs)
    seen: dict[bytes, SignedRoster] = {}
    for e in epochs:
        roster = operator.rosters.get(e)
        if roster is None:
            continue
        key = _roster_policy_key(roster)
        if key not in seen:
            seen[key] = roster
    latest = max(operator.epoch_roots) if operator.epoch_roots else 0
    terminal = (
        operator.epoch_roots[latest].payload_digest
        if operator.epoch_roots else b""
    )
    cov = CoverageAttestation(latest, issued_at, b"", terminal)
    sig = PopSchemeMPL.sign(operator.sk_o, cov.payload)
    coverage = CoverageAttestation(latest, issued_at, bytes(sig), terminal)
    return ComplianceCertificate(
        bytes(operator.pk_o), roots, tuple(seen.values()), coverage
    )


def verify_roster_signature(pk_o: G1Element, roster: SignedRoster) -> bool:
    """Check the operator signature over an embedded roster.

    Public because an event-proof verifier needs it too: before it can
    trust a roster to say which member owns a Merkle slot, it has to
    confirm the operator signed that roster.
    """
    payload = _roster_digest(roster)
    try:
        sig = G2Element.from_bytes(roster.signature)
    except Exception:
        return False
    return PopSchemeMPL.verify(pk_o, payload, sig)


def regulator_verify(cert: ComplianceCertificate) -> bool:
    """Offline verification using only the operator public key.

    Checks, in order:
      0. the certificate covers at least one epoch (an empty certificate
         attests nothing and is rejected rather than passing vacuously);
      1. every epoch-root operator signature verifies (the epoch
         identifier is inside the signed payload, so a root cannot be
         relabelled or duplicated under another epoch);
      2. epoch identifiers are unique, strictly increasing, and their
         intervals are non-decreasing and non-overlapping, so a reordered
         or duplicated root is rejected even though each signature is
         individually valid;
      2b. the roots form an unbroken chain: the first cites the genesis
         link value and each later one cites its predecessor's payload
         digest. Individually valid signatures do not make a certificate
         complete -- without this an epoch could simply be left out of
         the export and nothing in the remaining roots would say so.
         What this establishes is that the exported window starts at the
         operator's first epoch and has no interior gaps. It cannot, on
         its own, establish that the window extends to the present:
         dropping a suffix of the most recent epochs leaves a chain that
         is internally consistent, and detecting that needs a freshness
         anchor outside the certificate;
      3. every embedded roster's operator signature verifies;
      4. each epoch root's policy commitment (roster_digest field)
         matches the policy digest of at least one embedded,
         operator-signed roster.
    Any failure rejects. This closes the auditor-bypass in which a
    tampered or inserted roster is presented without a matching,
    operator-signed policy binding (Section VIII). Because the binding is
    over the epoch-independent policy rather than the member set alone,
    an epoch whose threshold, capabilities or castes differ cannot borrow
    another epoch's roster: it must carry its own. A policy that is
    static across k epochs is still satisfied by a single embedded O(N)
    roster.
    """
    try:
        pk_o = G1Element.from_bytes(cert.operator_pk)
    except Exception:
        return False

    # 0. a certificate attesting no epoch attests nothing
    if not cert.epoch_roots:
        return False

    # 1. epoch-root signatures
    if not all(verify_epoch_root(pk_o, r) for r in cert.epoch_roots):
        return False

    # 2. epoch sequence: unique, strictly increasing, intervals coherent
    epochs = [r.epoch for r in cert.epoch_roots]
    if len(set(epochs)) != len(epochs):
        return False
    if epochs != sorted(epochs):
        return False
    prev_close: float | None = None
    for r in cert.epoch_roots:
        if r.t_close < r.t_open:
            return False
        if prev_close is not None and r.t_open < prev_close:
            return False
        prev_close = r.t_close

    # 2b. epoch-root chain: genesis-anchored, no gaps. Each root cites the
    #     previous root's payload digest inside its own signed payload, so
    #     an omitted epoch leaves a citation nothing in the document can
    #     satisfy. The first root must cite the genesis link value, which
    #     is what rules out silently dropping a prefix.
    if cert.epoch_roots[0].prev_root_digest != EPOCH_LINK_GENESIS:
        return False
    for prev, cur in zip(cert.epoch_roots, cert.epoch_roots[1:]):
        if cur.prev_root_digest != prev.payload_digest:
            return False

    # 2c. coverage: the chain above catches an epoch removed from the
    #     middle or the front, but not one removed from the END, since a
    #     truncated prefix is itself a well-formed chain. The operator
    #     must therefore state under its own key how far its epoch
    #     sequence had run, and the document must run contiguously up to
    #     that epoch.
    cov = cert.coverage
    if cov is None:
        return False
    try:
        cov_sig = G2Element.from_bytes(cov.signature)
    except Exception:
        return False
    if not PopSchemeMPL.verify(pk_o, cov.payload, cov_sig):
        return False
    if epochs[-1] != cov.latest_epoch:
        return False
    # The attested terminal root must be the document's actual last root,
    # which is what makes the attestation specific to this fleet's chain
    # rather than to any chain that happens to end at the same index.
    if cov.terminal_root_digest != cert.epoch_roots[-1].payload_digest:
        return False
    if epochs != list(range(epochs[0], epochs[-1] + 1)):
        return False

    # 2. roster signatures; index policy digests of verified rosters
    verified_policies: set[bytes] = set()
    for roster in cert.rosters:
        if not verify_roster_signature(pk_o, roster):
            return False
        verified_policies.add(_roster_policy_key(roster))

    # 3. every epoch root's policy commitment must bind to a verified
    #    embedded roster. A tampered/inserted roster (wrong members or a
    #    forged signature) either fails step 2 or has no matching digest.
    for root in cert.epoch_roots:
        if root.roster_digest not in verified_policies:
            return False
    return True
