"""Underlying asset dynamics."""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class GBM:
    """Geometric Brownian motion under the risk-neutral measure.

    dS = (r - q) S dt + sigma S dW
    """

    S0: float
    r: float
    sigma: float
    q: float = 0.0

    def __post_init__(self):
        if self.S0 <= 0:
            raise ValueError("S0 must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

    def bump(self, **changes) -> "GBM":
        return replace(self, **changes)

    def simulate(self, T: float, n_steps: int, normals: np.ndarray) -> np.ndarray:
        """Exact log-Euler simulation from pre-drawn standard normals.

        ``normals`` has shape (n_paths, n_steps). Returns an array of shape
        (n_paths, n_steps + 1) whose first column is S0.
        """
        dt = T / n_steps
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        log_paths = np.cumsum(drift + diffusion * normals, axis=1)
        paths = np.empty((normals.shape[0], n_steps + 1))
        paths[:, 0] = self.S0
        paths[:, 1:] = self.S0 * np.exp(log_paths)
        return paths
