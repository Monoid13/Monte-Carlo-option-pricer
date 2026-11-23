"""Monte Carlo pricing engine."""

import copy
from dataclasses import dataclass
from math import exp, sqrt

import numpy as np

from .analytic import geometric_asian_price
from .models import GBM
from .options import AmericanOption, AsianOption, _vanilla

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class PricingResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def conf_int(self) -> tuple:
        """95% confidence interval for the price."""
        half = Z_95 * self.std_error
        return (self.price - half, self.price + half)

    def __str__(self) -> str:
        lo, hi = self.conf_int
        return (
            f"price = {self.price:.6f}  (s.e. {self.std_error:.6f}, "
            f"95% CI [{lo:.6f}, {hi:.6f}], {self.n_paths:,} paths)"
        )


class MonteCarloPricer:
    """Prices options on a GBM underlying by Monte Carlo simulation.

    Parameters
    ----------
    model : GBM
        Underlying dynamics.
    n_paths : int
        Number of simulated paths (rounded up to even when antithetic=True).
    n_steps : int
        Time steps per path. Path-independent options use a single step
        regardless, since GBM is simulated exactly.
    antithetic : bool
        Pair each normal draw Z with -Z.
    control_variate : bool
        Use a control variate with known expectation: the geometric Asian
        price for arithmetic Asians, otherwise the discounted terminal spot.
    seed : int or None
        Seed for reproducibility. A fixed seed also gives common random
        numbers across re-pricings, which is what the Greeks rely on.
    batch_size : int
        Maximum paths held in memory at once (not used for American options,
        whose regression needs every path together).
    """

    def __init__(
        self,
        model: GBM,
        n_paths: int = 100_000,
        n_steps: int = 252,
        antithetic: bool = True,
        control_variate: bool = False,
        seed=None,
        batch_size: int = 50_000,
        lsm_degree: int = 3,
    ):
        if n_paths < 2:
            raise ValueError("n_paths must be at least 2")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        self.model = model
        self.n_paths = n_paths + (n_paths % 2 if antithetic else 0)
        self.n_steps = n_steps
        self.antithetic = antithetic
        self.control_variate = control_variate
        self.seed = seed
        self.batch_size = max(2, batch_size - batch_size % 2)
        self.lsm_degree = lsm_degree

    def with_model(self, model: GBM) -> "MonteCarloPricer":
        new = copy.copy(self)
        new.model = model
        return new

    # ------------------------------------------------------------------ paths
    def _normals(self, rng, n, steps):
        if self.antithetic:
            half = rng.standard_normal((n // 2, steps))
            # Interleave so rows 2i and 2i+1 are an antithetic pair.
            return np.stack([half, -half], axis=1).reshape(n, steps)
        return rng.standard_normal((n, steps))

    def _batches(self, steps, T):
        rng = np.random.default_rng(self.seed)
        remaining = self.n_paths
        while remaining > 0:
            n = min(remaining, self.batch_size)
            yield self.model.simulate(T, steps, self._normals(rng, n, steps))
            remaining -= n

    # ---------------------------------------------------------------- pricing
    def price(self, option) -> PricingResult:
        steps = self.n_steps if option.path_dependent else 1
        disc = exp(-self.model.r * option.T)

        if isinstance(option, AmericanOption):
            rng = np.random.default_rng(self.seed)
            paths = self.model.simulate(
                option.T, steps, self._normals(rng, self.n_paths, steps)
            )
            y = self._longstaff_schwartz(option, paths)
            x = disc * paths[:, -1] if self.control_variate else None
        else:
            ys, xs = [], []
            for paths in self._batches(steps, option.T):
                ys.append(disc * option.payoff(paths))
                if self.control_variate:
                    xs.append(disc * self._control(option, paths))
            y = np.concatenate(ys)
            x = np.concatenate(xs) if self.control_variate else None

        if self.control_variate:
            y = self._apply_control(y, x, self._control_mean(option, steps))
        return self._summarise(y)

    def _summarise(self, y):
        if self.antithetic:
            y = 0.5 * (y[0::2] + y[1::2])  # average pairs -> i.i.d. samples
        n = y.size
        return PricingResult(
            price=float(y.mean()),
            std_error=float(y.std(ddof=1) / sqrt(n)),
            n_paths=self.n_paths,
        )

    # -------------------------------------------------------- control variate
    @staticmethod
    def _is_arith_asian(option):
        return isinstance(option, AsianOption) and option.average == "arithmetic"

    def _control(self, option, paths):
        if self._is_arith_asian(option):
            geo = np.exp(np.log(paths[:, 1:]).mean(axis=1))
            return _vanilla(geo, option.K, option.kind)
        return paths[:, -1]

    def _control_mean(self, option, steps):
        m = self.model
        if self._is_arith_asian(option):
            return geometric_asian_price(m.S0, option.K, option.T, m.r, m.sigma, steps, m.q, option.kind)
        return m.S0 * exp(-m.q * option.T)

    @staticmethod
    def _apply_control(y, x, x_mean):
        cov = np.cov(y, x)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        return y - beta * (x - x_mean)

    # ------------------------------------------------------ Longstaff-Schwartz
    def _longstaff_schwartz(self, option, paths):
        """Discounted cash flow per path under the LSM exercise policy."""
        n_steps = paths.shape[1] - 1
        dt = option.T / n_steps
        step_disc = exp(-self.model.r * dt)
        cash = option.exercise_value(paths[:, -1])
        for t in range(n_steps - 1, 0, -1):
            cash *= step_disc
            spot = paths[:, t]
            exercise = option.exercise_value(spot)
            itm = exercise > 0
            if itm.sum() <= self.lsm_degree + 1:
                continue
            x = spot[itm] / option.K
            basis = np.vander(x, self.lsm_degree + 1)
            coef, *_ = np.linalg.lstsq(basis, cash[itm], rcond=None)
            continuation = basis @ coef
            ex_now = exercise[itm] > continuation
            idx = np.flatnonzero(itm)[ex_now]
            cash[idx] = exercise[idx]
        return cash * step_disc
