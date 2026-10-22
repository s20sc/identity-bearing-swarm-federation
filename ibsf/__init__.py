"""Identity-bearing swarm federation: reference implementation."""

from .audit_chain import (
    AuditChain,
    AuditEvent,
    EpochClose,
    MerkleTree,
    close_epoch,
    epoch_leaf,
)
from .certificate import (
    ComplianceCertificate,
    CoverageAttestation,
    export_certificate,
    regulator_verify,
)
from .federation import (
    Fleet,
    Robot,
    canonical,
    run_decision,
    submit_decision_tuple,
)
from .manifest import CasteCertificate, IdentityManifest
from .network import NetworkModel
from .operator import (
    CommissioningStats,
    Operator,
    roster_policy_digest,
    verify_epoch_root,
    verify_manifest,
)

__all__ = [
    "AuditChain",
    "AuditEvent",
    "CasteCertificate",
    "CommissioningStats",
    "ComplianceCertificate",
    "CoverageAttestation",
    "EpochClose",
    "Fleet",
    "IdentityManifest",
    "MerkleTree",
    "NetworkModel",
    "Operator",
    "Robot",
    "canonical",
    "close_epoch",
    "epoch_leaf",
    "export_certificate",
    "roster_policy_digest",
    "verify_epoch_root",
    "verify_manifest",
    "regulator_verify",
    "run_decision",
    "submit_decision_tuple",
]

__version__ = "0.1.0"
