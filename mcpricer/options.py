"""Option contracts. Each maps simulated paths to undiscounted payoffs."""

from dataclasses import dataclass

import numpy as np

KINDS = ("call", "put")


def _vanilla(x: np.ndarray, K: float, kind: str) -> np.ndarray:
    return np.maximum(x - K, 0.0) if kind == "call" else np.maximum(K - x, 0.0)


@dataclass(frozen=True)
class _Option:
    K: float
    T: float
    kind: str = "call"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.T <= 0:
            raise ValueError("T must be positive")

    # Number of time steps the contract needs at minimum (path-independent = 1).
    path_dependent = False
    early_exercise = False

    def payoff(self, paths: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class EuropeanOption(_Option):
    def payoff(self, paths):
        return _vanilla(paths[:, -1], self.K, self.kind)


@dataclass(frozen=True)
class AsianOption(_Option):
    """Arithmetic- or geometric-average-price option over the monitoring dates."""

    average: str = "arithmetic"
    path_dependent = True

    def payoff(self, paths):
        monitored = paths[:, 1:]
        if self.average == "arithmetic":
            avg = monitored.mean(axis=1)
        elif self.average == "geometric":
            avg = np.exp(np.log(monitored).mean(axis=1))
        else:
            raise ValueError("average must be 'arithmetic' or 'geometric'")
        return _vanilla(avg, self.K, self.kind)


@dataclass(frozen=True)
class BarrierOption(_Option):
    """Discretely monitored knock-in / knock-out barrier option.

    barrier_type is one of 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'.
    """

    barrier: float = 0.0
    barrier_type: str = "up-and-out"
    rebate: float = 0.0
    path_dependent = True

    def payoff(self, paths):
        direction, _, knock = self.barrier_type.partition("-and-")
        if direction not in ("up", "down") or knock not in ("in", "out"):
            raise ValueError(f"invalid barrier_type {self.barrier_type!r}")
        if direction == "up":
            hit = (paths >= self.barrier).any(axis=1)
        else:
            hit = (paths <= self.barrier).any(axis=1)
        active = ~hit if knock == "out" else hit
        vanilla = _vanilla(paths[:, -1], self.K, self.kind)
        return np.where(active, vanilla, self.rebate)


@dataclass(frozen=True)
class LookbackOption(_Option):
    """Lookback option. floating=True gives the floating-strike variant (K ignored)."""

    floating: bool = False
    path_dependent = True

    def payoff(self, paths):
        hi, lo, last = paths.max(axis=1), paths.min(axis=1), paths[:, -1]
        if self.floating:
            return last - lo if self.kind == "call" else hi - last
        return _vanilla(hi if self.kind == "call" else lo, self.K, self.kind)


@dataclass(frozen=True)
class AmericanOption(_Option):
    """American vanilla option, priced by Longstaff-Schwartz regression."""

    path_dependent = True
    early_exercise = True

    def exercise_value(self, spots: np.ndarray) -> np.ndarray:
        return _vanilla(spots, self.K, self.kind)

    def payoff(self, paths):
        # Used only for the exercise-at-expiry lower bound.
        return _vanilla(paths[:, -1], self.K, self.kind)
