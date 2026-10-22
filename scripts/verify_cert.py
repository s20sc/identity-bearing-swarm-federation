"""External-regulator verification from an exported certificate alone.

By default this verifies the artifact already committed under
`data/verify_cert/` and regenerates nothing. That default is the point:
a harness that re-exports and then verifies what it just wrote only
tests the exporter against itself, and would pass even if the committed
files were stale, hand-edited, or written by different code. Pass
`--export` to regenerate the artifact first, which is needed after any
change to the signed formats. Tamper tests run in both modes.

Two clean phases:

  1. EXPORT (only under `--export`) — build a fleet, run an epoch, and
     write a compliance certificate plus one per-event inclusion proof
     to JSON files on disk. The event proof is self-contained: it
     carries the event body, the signer's public key and the event
     signature, the remainder of that agent's chain up to its head, the
     slot index, and the Merkle path from the identity-bound leaf to the
     signed fleet root.

  2. VERIFY — in functions that receive ONLY the certificate JSON, the
     proof JSON, and the operator public key (no live fleet, no secret
     keys, no fresh crypto material beyond the operator pk):
       (a) verify every epoch-root operator signature,
       (b) verify every embedded roster's operator signature,
       (c) verify the roster-policy binding of each epoch root,
       (d) verify the event's own BLS signature under the signer key,
       (e) replay the hash pointers from that event to the chain head,
       (f) recompute the identity-bound Merkle leaf for the slot,
       (g) verify the Merkle path from that leaf to the fleet root, and
       (h) bind the fleet root to a signed epoch root whose roster puts
           the signer at exactly that slot.

Tamper tests then mutate the certificate and the proof and confirm each
mutation is rejected. Exit code is non-zero if any expected outcome does
not hold.

Note on imports: this file imports the library's format definitions
(`AuditEvent`, `epoch_leaf`, `MerkleTree`) because a verifier has to
agree with the prover on encodings. It imports no fleet state and uses
no secret material; every input to the verify functions comes from the
three exported files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blspy import G1Element, G2Element, PopSchemeMPL  # noqa: E402

from ibsf import bls  # noqa: E402
from ibsf.audit_chain import AuditEvent, MerkleTree, epoch_leaf  # noqa: E402
from ibsf.certificate import (  # noqa: E402
    ComplianceCertificate,
    export_certificate,
    regulator_verify,
    verify_roster_signature,
)
from ibsf.federation import Fleet  # noqa: E402
from ibsf.manifest import sha256  # noqa: E402
from ibsf.operator import roster_policy_digest, verify_epoch_root  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "verify_cert"
CERT_PATH = OUT_DIR / "compliance_cert.json"
PROOF_PATH = OUT_DIR / "event_proof.json"
PUBKEY_PATH = OUT_DIR / "operator_pk.hex"

PROOF_FORMAT = "ibsf-event-proof/v1"
EPOCHS = 3                    # >1, so root-chain gaps are demonstrable
EVENTS_PER_ROBOT = 4          # per epoch
PROVED_EVENT_SEQ = 1          # not the head event, so the proof has a segment


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def _event_doc(ev: AuditEvent) -> dict:
    """Serialize one audit event body plus its signature."""
    return {
        "seq": ev.seq,
        "kind": ev.kind,
        "payload": ev.payload.hex(),
        "prev_hash": ev.prev_hash.hex(),
        "signature": ev.signature.hex(),
    }


def export_phase() -> None:
    """Build a fleet, run several epochs, and write cert + proof + pubkey.

    More than one epoch is closed on purpose: with a single epoch the
    root chain has nothing to link, so the certificate could not
    demonstrate that an omitted epoch is detectable. The proved event
    sits in an interior epoch for the same reason.
    """
    OUT_DIR.mkdir(exist_ok=True)
    fleet = Fleet.create(10, seed=7)
    chains = {r.robot_id: r.chain for r in fleet.robots}
    member_pks = fleet.member_pks
    members = [
        (r.pk_bytes, bytes(bls.pop_prove(r.sk)[0])) for r in fleet.robots
    ]
    caps = {r.pk_bytes: r.manifest.cap for r in fleet.robots}
    castes = {r.pk_bytes: r.caste for r in fleet.robots}

    closes = {}
    for e in range(EPOCHS):
        if e > 0:
            # Re-sign with the SAME capability and caste binding the fleet
            # was commissioned under. Omitting either left epochs 1..k-1
            # with weaker rosters than epoch 0, so the sweep silently
            # measured a policy that CHANGED at epoch 1, not the static
            # one it reports.
            fleet.operator.sign_roster(
                e, members, t_open=60.0 * e, capabilities=caps, castes=castes
            )
        for r in fleet.robots:
            for j in range(EVENTS_PER_ROBOT):
                r.chain.append(
                    "capability-invocation",
                    sha256(f"demo/{r.robot_id}/{e}/{j}".encode()),
                )
        _, closes[e], _ = fleet.operator.close_and_sign_epoch(
            e, chains, 60.0 * e, 60.0 * (e + 1), member_pks
        )

    cert = export_certificate(fleet.operator, list(range(EPOCHS)))
    CERT_PATH.write_bytes(cert.to_json(embed_rosters=True))
    PUBKEY_PATH.write_text(bytes(fleet.operator.pk_o).hex())

    # One per-event inclusion proof for robot 0 in an interior epoch.
    # `closed.slots` is ordered by member key, so the slot index is also
    # robot 0's position in the operator-signed roster, which is what
    # lets a verifier check the signer against the roster rather than
    # against a caller-supplied tag.
    proof_epoch = EPOCHS // 2
    closed = closes[proof_epoch]
    target_id = 0
    slot = closed.slots.index(target_id)
    chain = chains[target_id]
    signer_pk = member_pks[target_id]
    events = chain.events
    head = closed.heads[target_id]
    head_idx = next(
        i for i, ev in enumerate(events) if ev.event_hash == head
    )
    leaf = epoch_leaf(proof_epoch, slot, signer_pk, head)
    tree = MerkleTree(closed.leaves)
    assert tree.root == closed.fleet_root
    path = tree.proof(slot)

    proof_doc = {
        "format": PROOF_FORMAT,
        "epoch": proof_epoch,
        "slot": slot,
        "signer_pk": signer_pk.hex(),
        "event": _event_doc(events[PROVED_EVENT_SEQ]),
        # the chain between the event and the head THIS epoch committed;
        # the live chain runs past it into later epochs
        "successors": [
            _event_doc(ev)
            for ev in events[PROVED_EVENT_SEQ + 1: head_idx + 1]
        ],
        "chain_head": head.hex(),
        "leaf": leaf.hex(),
        "path": [[sib.hex(), is_right] for sib, is_right in path],
        "fleet_root": closed.fleet_root.hex(),
    }
    PROOF_PATH.write_text(json.dumps(proof_doc, separators=(",", ":")))
    logger.info(
        "export: cert=%d bytes, event proof=%d bytes (%d successors)",
        CERT_PATH.stat().st_size,
        PROOF_PATH.stat().st_size,
        len(proof_doc["successors"]),
    )


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------

def _load_event(doc: dict, signer_pk: bytes) -> AuditEvent:
    return AuditEvent(
        seq=int(doc["seq"]),
        kind=str(doc["kind"]),
        payload=bytes.fromhex(doc["payload"]),
        prev_hash=bytes.fromhex(doc["prev_hash"]),
        signer_pk=signer_pk,
        signature=bytes.fromhex(doc["signature"]),
    )


def verify_event_proof(
    proof_blob: bytes, cert: ComplianceCertificate
) -> bool:
    """Check one exported event proof against a compliance certificate.

    Everything below is recomputed from the proof document; nothing is
    accepted on the exporter's word. In order: the event's signature
    under the declared signer key, the hash-pointer replay from that
    event to the claimed chain head, the identity-bound leaf, the Merkle
    path to the fleet root, and the operator signature over the epoch
    root carrying that fleet root. The last step also requires the
    signer to occupy the claimed slot in the epoch's operator-signed
    roster, so the proof attributes the event to a named member instead
    of to an anonymous tree position.
    """
    proof = json.loads(proof_blob)
    if proof.get("format") != PROOF_FORMAT:
        logger.error("unrecognised event-proof format")
        return False

    try:
        pk_o = G1Element.from_bytes(cert.operator_pk)
        signer_pk = bytes.fromhex(proof["signer_pk"])
        signer = G1Element.from_bytes(signer_pk)
        epoch = int(proof["epoch"])
        slot = int(proof["slot"])
        chain_head = bytes.fromhex(proof["chain_head"])
        claimed_leaf = bytes.fromhex(proof["leaf"])
        claimed_root = bytes.fromhex(proof["fleet_root"])
        target = _load_event(proof["event"], signer_pk)
        successors = [
            _load_event(d, signer_pk) for d in proof["successors"]
        ]
    except Exception as exc:
        logger.error("malformed event proof: %s", exc)
        return False

    if slot < 0:
        logger.error("negative slot index")
        return False

    # (a) the event's own signature, under the key the proof names
    for ev in [target] + successors:
        try:
            sig = G2Element.from_bytes(ev.signature)
        except Exception:
            logger.error("malformed event signature at seq %d", ev.seq)
            return False
        if not PopSchemeMPL.verify(signer, ev.signed_body, sig):
            logger.error("event signature invalid at seq %d", ev.seq)
            return False

    # (b) the event hashes into the claimed chain head. `event_hash`
    # commits to the signature as well as the body, so a stripped or
    # swapped signature also breaks this replay.
    cur = target.event_hash
    expect_seq = target.seq
    for ev in successors:
        expect_seq += 1
        if ev.seq != expect_seq:
            logger.error("chain segment is not contiguous at seq %d", ev.seq)
            return False
        if ev.prev_hash != cur:
            logger.error("chain link broken at seq %d", ev.seq)
            return False
        cur = ev.event_hash
    if cur != chain_head:
        logger.error("chain segment does not reach the claimed head")
        return False

    # (c) the identity-bound leaf must recompute from (epoch, slot,
    # signer key, head); a leaf over the bare head would not say whose
    # chain it was.
    leaf = epoch_leaf(epoch, slot, signer_pk, chain_head)
    if leaf != claimed_leaf:
        logger.error("identity-bound leaf does not recompute")
        return False

    # (d) Merkle path from that leaf to the claimed fleet root
    try:
        path = [
            (bytes.fromhex(sib), bool(is_right))
            for sib, is_right in proof["path"]
        ]
    except Exception as exc:
        logger.error("malformed Merkle path: %s", exc)
        return False
    if not MerkleTree.verify_proof(leaf, path, claimed_root):
        logger.error("Merkle path does not reconstruct the claimed root")
        return False

    # (e) the claimed root must be an operator-signed epoch root, and the
    # epoch's roster must place this signer at this slot
    root = next(
        (
            r for r in cert.epoch_roots
            if r.epoch == epoch and r.fleet_root == claimed_root
        ),
        None,
    )
    if root is None:
        logger.error("fleet root not bound to any epoch root in the cert")
        return False
    if not verify_epoch_root(pk_o, root):
        logger.error("operator signature over the epoch root is invalid")
        return False
    rosters = [
        r for r in cert.rosters
        if roster_policy_digest(
            r.member_pks, r.cap_digests, r.t_quorum, r.caste_digests
        ) == root.roster_digest
        and verify_roster_signature(pk_o, r)
    ]
    if not rosters:
        logger.error("no signed roster matches the epoch root's policy")
        return False
    if not any(
        slot < len(r.member_pks) and r.member_pks[slot] == signer_pk
        for r in rosters
    ):
        logger.error("signer does not hold slot %d in the signed roster", slot)
        return False
    return True


def verify_from_files(
    cert_blob: bytes, proof_blob: bytes, operator_pk_hex: str
) -> bool:
    """Verify using only the exported material and the operator pubkey."""
    try:
        cert = ComplianceCertificate.from_json(cert_blob)
    except Exception as exc:
        logger.error("malformed certificate: %s", exc)
        return False

    # the caller-supplied operator pk must match the certificate's pk
    if cert.operator_pk.hex() != operator_pk_hex.strip():
        logger.error("operator pk mismatch")
        return False
    try:
        G1Element.from_bytes(cert.operator_pk)
    except Exception:
        logger.error("operator pk malformed")
        return False

    # epoch roots, roster signatures, roster-policy binding
    if not regulator_verify(cert):
        logger.error("regulator_verify failed")
        return False

    # per-event offline proof: signature, chain replay, leaf, path, root
    return verify_event_proof(proof_blob, cert)


# --------------------------------------------------------------------------
# tamper tests
# --------------------------------------------------------------------------

def _expect_reject(
    label: str, cert_blob: bytes, proof_blob: bytes, pk_hex: str
) -> None:
    if verify_from_files(cert_blob, proof_blob, pk_hex):
        raise SystemExit(f"FAIL: {label} verified (should reject)")
    logger.info("%s correctly rejected: PASS", label)


def tamper_tests(cert_blob: bytes, proof_blob: bytes, pk_hex: str) -> None:
    """Each mutation below must make verification fail."""
    # 1. flipped byte in a signed epoch root's fleet_root
    doc = json.loads(cert_blob)
    fr = bytearray.fromhex(doc["epochs"][0]["fleet_root"])
    fr[0] ^= 0x01
    doc["epochs"][0]["fleet_root"] = fr.hex()
    _expect_reject(
        "tampered fleet root",
        json.dumps(doc, separators=(",", ":")).encode(), proof_blob, pk_hex,
    )

    # 2. inserted roster with a mismatched policy
    doc = json.loads(cert_blob)
    if doc.get("rosters"):
        bad = dict(doc["rosters"][0])
        pks = list(bad["member_pks"])
        pks[0] = "00" * 48                      # off-roster fake key
        bad["member_pks"] = pks
        doc["rosters"].append(bad)
        _expect_reject(
            "inserted rogue roster",
            json.dumps(doc, separators=(",", ":")).encode(), proof_blob,
            pk_hex,
        )

    # 2b. omitted interior epoch. Every remaining root still carries a
    #     valid operator signature; only the root chain shows the gap.
    doc = json.loads(cert_blob)
    if len(doc["epochs"]) >= 3:
        gapped = dict(doc)
        gapped["epochs"] = [
            e for i, e in enumerate(doc["epochs"]) if i != 1
        ]
        _expect_reject(
            "omitted interior epoch",
            json.dumps(gapped, separators=(",", ":")).encode(), proof_blob,
            pk_hex,
        )

    # 2c. omitted leading epoch: the surviving first root no longer cites
    #     the genesis link value
    doc = json.loads(cert_blob)
    if len(doc["epochs"]) >= 2:
        trimmed = dict(doc)
        trimmed["epochs"] = doc["epochs"][1:]
        _expect_reject(
            "omitted leading epoch",
            json.dumps(trimmed, separators=(",", ":")).encode(), proof_blob,
            pk_hex,
        )

    # 3. flipped event payload: the signature no longer covers the body,
    #    and the event hash no longer chains to the head
    p = json.loads(proof_blob)
    payload = bytearray.fromhex(p["event"]["payload"])
    payload[0] ^= 0x01
    p["event"]["payload"] = payload.hex()
    _expect_reject(
        "flipped event payload", cert_blob,
        json.dumps(p, separators=(",", ":")).encode(), pk_hex,
    )

    # 4. swapped signer key: another genuine roster member is named as
    #    the signer, so the roster check passes but the signature and the
    #    identity-bound leaf do not
    p = json.loads(proof_blob)
    roster = json.loads(cert_blob)["rosters"][0]["member_pks"]
    other = next(pk for pk in roster if pk != p["signer_pk"])
    p["signer_pk"] = other
    _expect_reject(
        "swapped signer key", cert_blob,
        json.dumps(p, separators=(",", ":")).encode(), pk_hex,
    )

    # 5. wrong slot, leaf left alone: the leaf binds the slot index, so
    #    it no longer recomputes
    p = json.loads(proof_blob)
    p["slot"] = p["slot"] + 1
    _expect_reject(
        "wrong slot index", cert_blob,
        json.dumps(p, separators=(",", ":")).encode(), pk_hex,
    )

    # 5b. wrong slot with the leaf recomputed to match, which is what a
    #     prover trying to relocate an event would actually submit. The
    #     leaf check now passes and the Merkle path is what rejects it.
    p = json.loads(proof_blob)
    p["slot"] = p["slot"] + 1
    p["leaf"] = epoch_leaf(
        int(p["epoch"]), int(p["slot"]),
        bytes.fromhex(p["signer_pk"]), bytes.fromhex(p["chain_head"]),
    ).hex()
    _expect_reject(
        "relocated event (slot + matching leaf)", cert_blob,
        json.dumps(p, separators=(",", ":")).encode(), pk_hex,
    )

    # 6. truncated chain segment: the remaining events no longer reach
    #    the committed head
    p = json.loads(proof_blob)
    if p["successors"]:
        p["successors"] = p["successors"][:-1]
        _expect_reject(
            "truncated chain segment", cert_blob,
            json.dumps(p, separators=(",", ":")).encode(), pk_hex,
        )

    # 7. stripped event signature: the proof is no longer an attested
    #    record even though the body is intact
    p = json.loads(proof_blob)
    p["event"]["signature"] = "00" * 96
    _expect_reject(
        "stripped event signature", cert_blob,
        json.dumps(p, separators=(",", ":")).encode(), pk_hex,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the committed compliance certificate and event proof "
            "offline. Regenerates nothing unless --export is given."
        )
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help=(
            "regenerate data/verify_cert/ before verifying; needed after a "
            "change to the signed formats"
        ),
    )
    args = parser.parse_args(argv)

    if args.export:
        export_phase()
    else:
        missing = [
            str(p) for p in (CERT_PATH, PROOF_PATH, PUBKEY_PATH)
            if not p.exists()
        ]
        if missing:
            raise SystemExit(
                "FAIL: no committed artifact to verify (missing "
                + ", ".join(missing)
                + "); re-run with --export to generate it"
            )
        logger.info("verifying the committed artifact under %s", OUT_DIR)

    # VERIFY PHASE: load only the exported files + operator pubkey.
    cert_blob = CERT_PATH.read_bytes()
    proof_blob = PROOF_PATH.read_text().encode()
    operator_pk_hex = PUBKEY_PATH.read_text()

    if not verify_from_files(cert_blob, proof_blob, operator_pk_hex):
        raise SystemExit("FAIL: committed certificate did not verify")
    logger.info("committed certificate + event proof verification: PASS")

    tamper_tests(cert_blob, proof_blob, operator_pk_hex)
    logger.info("external verifier: ALL CHECKS PASS")


if __name__ == "__main__":
    main()
