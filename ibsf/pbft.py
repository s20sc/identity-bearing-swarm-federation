"""PBFT baseline over the same simulated network substrate (Section VIII).

A fair latency model of three-phase PBFT (pre-prepare, prepare, commit)
with MAC authenticators, as in Castro-Liskov practical BFT:
  - 3 sequential communication rounds; prepare and commit are
    all-to-all. No round is gated on the slowest of its links (the
    100th percentile); every round completes at a quorum order
    statistic, the same accounting the BLS construction receives.
    Pre-prepare is a leader broadcast, and a replica enters the prepare
    phase on its own receipt rather than waiting for its peers, so that
    round completes at the (2f+1)-th fastest of the (N-1) leader links.
    Prepare and commit complete when 2f+1 replicas have each collected
    2f+1 matching messages: per receiving replica, its round-complete
    time is the (2f+1)-th smallest of its (N-1) incoming one-way delays,
    and the round completes at the (2f+1)-th smallest of those
    per-replica quorum times.
  - per-round processing at the bottleneck replica: 2f+1 quorum-MAC
    verifications per message batch, measured wall-clock on real
    HMAC-SHA256 operations;
  - message complexity 2N(N-1) + (N-1) versus 2N for the BLS
    construction.

View changes and batching pathologies are omitted, so the model is a
conservative (optimistic-for-PBFT) baseline.
"""

from __future__ import annotations

import hmac
import hashlib
import time
from dataclasses import dataclass

import numpy as np

from .network import NetworkModel


@dataclass(frozen=True)
class PBFTResult:
    total_ms: float
    transport_ms: float
    mac_ms: float
    messages: int


def _mac_batch_ms(key: bytes, msg: bytes, count: int) -> float:
    t0 = time.perf_counter()
    for i in range(count):
        hmac.new(key, msg + i.to_bytes(4, "big"), hashlib.sha256).digest()
    return (time.perf_counter() - t0) * 1000.0


def _broadcast_quorum_ms(
    net: NetworkModel, rng: np.random.Generator, n: int, quorum: int
) -> float:
    """Leader-to-all broadcast, timed to quorum receipt rather than to all.

    A replica enters the next phase the moment the pre-prepare reaches
    it, and the protocol advances once a quorum of replicas has done so.
    The LEADER is itself one of those replicas and holds the pre-prepare
    at time zero, so only quorum-1 follower receipts are waited for; an
    earlier revision waited for the quorum-th follower, overstating this
    round by one order statistic. Charging the slowest of the (N-1)
    links would be worse still: the maximum of N log-normal draws grows
    with N while a fixed-fraction order statistic does not, so a
    max-gate manufactures a divergent PBFT tail at large N. Both
    baselines are timed at quorum order statistics throughout.
    """
    if n <= 1:
        return 0.0
    followers_needed = quorum - 1        # leader already has the message
    if followers_needed <= 0:
        return 0.0
    delays = net.one_way(rng, n - 1)
    delays.sort()
    idx = min(followers_needed, n - 1) - 1
    return float(delays[idx])


def _quorum_round_ms(
    net: NetworkModel, rng: np.random.Generator, n: int, quorum: int
) -> float:
    """Latency of one all-to-all quorum round via order statistics.

    Each of the N receiving replicas draws (N-1) incoming one-way delays.
    A replica reaches quorum once it has collected `quorum` matching
    messages, i.e. at the `quorum`-th smallest of its (N-1) delays. The
    round completes when `quorum` replicas have reached that state, i.e.
    at the `quorum`-th smallest of the per-replica quorum times. This is
    the time-to-quorum, not the slowest-link (100th percentile) time.
    """
    if n <= 1:
        return 0.0
    incoming = quorum - 1  # messages needed beyond the replica's own
    incoming = min(max(incoming, 1), n - 1)  # bounded by (N-1) senders
    # (N x (N-1)) matrix of one-way delays, sorted along each row
    delays = net.one_way(rng, n * (n - 1)).reshape(n, n - 1)
    delays.sort(axis=1)
    # each replica reaches quorum at its `incoming`-th smallest delay
    per_replica_quorum = delays[:, incoming - 1]
    per_replica_quorum.sort()
    # round completes when `quorum` replicas have their quorum
    idx = min(quorum, n) - 1
    return float(per_replica_quorum[idx])


def run_pbft_decision(
    n: int, net: NetworkModel, rng: np.random.Generator, seed: int
) -> PBFTResult:
    """One PBFT consensus instance over n replicas."""
    f = (n - 1) // 3
    quorum = 2 * f + 1
    key = hashlib.sha256(f"pbft/{seed}".encode()).digest()
    msg = f"decision/{seed}".encode()

    # Round 1: pre-prepare, leader -> all (n-1 links), parallel broadcast,
    # gated at quorum receipt (not the slowest link) like rounds 2 and 3.
    r1 = _broadcast_quorum_ms(net, rng, n, quorum)
    # Round 2: prepare, all-to-all; completes at the quorum-time order
    # statistic (2f+1 replicas each holding 2f+1 matching prepares).
    r2 = _quorum_round_ms(net, rng, n, quorum)
    # Round 3: commit, all-to-all, same shape.
    r3 = _quorum_round_ms(net, rng, n, quorum)

    # MAC processing at the bottleneck replica: verify quorum messages
    # in prepare and commit rounds plus the pre-prepare.
    mac_ms = _mac_batch_ms(key, msg, 2 * quorum + 1)

    messages = (n - 1) + 2 * n * (n - 1)
    total = r1 + r2 + r3 + mac_ms
    return PBFTResult(total, r1 + r2 + r3, mac_ms, messages)
