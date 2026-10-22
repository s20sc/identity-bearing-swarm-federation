"""Simulated network and operator-overhead latency models (Section VIII-A).

Transport latency is log-normal, fit to wired-LAN measurements with
median 0.4 ms and p99 4.5 ms. Operator scheduling overhead is
log-normal with median 5 ms. Crypto times are NOT simulated: they are
measured wall-clock on real blst operations by the callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Z99 = 2.3263478740408408  # Phi^{-1}(0.99)


def _lognorm_params(median_ms: float, p99_ms: float) -> tuple[float, float]:
    mu = math.log(median_ms)
    sigma = (math.log(p99_ms) - mu) / Z99
    return mu, sigma


@dataclass
class NetworkModel:
    """Seed-driven log-normal latency model."""

    median_ms: float = 0.4
    p99_ms: float = 4.5
    operator_overhead_median_ms: float = 5.0
    operator_overhead_p99_ms: float = 12.0

    def __post_init__(self) -> None:
        self._mu, self._sigma = _lognorm_params(self.median_ms, self.p99_ms)
        self._omu, self._osigma = _lognorm_params(
            self.operator_overhead_median_ms, self.operator_overhead_p99_ms
        )

    def one_way(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """n independent one-way transport delays, ms."""
        return rng.lognormal(self._mu, self._sigma, size=n)

    # A max-of-n "fanout" helper was deliberately removed. The maximum of
    # n log-normal draws grows with n while a fixed-fraction order
    # statistic does not, so timing any broadcast round by its slowest
    # link manufactures a divergent tail at large N. Every round in both
    # the federation path and the PBFT baseline is timed at a quorum
    # order statistic instead (see ibsf/pbft.py).

    def operator_overhead(self, rng: np.random.Generator) -> float:
        """Operator-side scheduling/processing overhead per round, ms."""
        return float(rng.lognormal(self._omu, self._osigma))
