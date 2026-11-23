"""Timing and message accounting for a three-round PBFT schedule.

The model draws delays for a leader broadcast and two all-to-all
rounds. Each quorum includes the local replica's message. It charges
for a batch of HMAC-SHA256 tag generation at one endpoint; tag
comparison and a replica state machine are outside the model. Its
message tally is (N - 1) + 2N(N - 1), and its timing parameters are
not a consensus-safety configuration.
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
    """Leader-to-all broadcast timed to quorum receipt.

    The leader has the pre-prepare at time zero and counts toward quorum.
    Only quorum-1 follower receipts are needed. Waiting for all N-1 links
    would measure the maximum link delay rather than quorum progression;
    waiting for quorum follower receipts would count one extra replica.
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

    Each receiving replica draws N - 1 incoming delays. Its own message
    counts, so it advances after quorum - 1 arrivals from peers. The
    round ends at the quorum-th local completion time.
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
    """Time the modelled three-round schedule for n replicas."""
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

    # Charge tag generation for the prepare and commit quorums
    # and the pre-prepare at one endpoint.
    mac_ms = _mac_batch_ms(key, msg, 2 * quorum + 1)

    messages = (n - 1) + 2 * n * (n - 1)
    total = r1 + r2 + r3 + mac_ms
    return PBFTResult(total, r1 + r2 + r3, mac_ms, messages)
