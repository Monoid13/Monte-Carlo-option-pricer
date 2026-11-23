"""Closed-form Black-Scholes results, used as benchmarks and control variates."""

from math import erf, exp, log, pi, sqrt


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _d1_d2(S, K, T, r, sigma, q):
    vol_sqrt_t = sigma * sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def black_scholes_price(S, K, T, r, sigma, q=0.0, kind="call"):
    """Black-Scholes-Merton price of a European call or put."""
    if T <= 0 or sigma <= 0:
        fwd_intrinsic = S * exp(-q * T) - K * exp(-r * T)
        return max(fwd_intrinsic, 0.0) if kind == "call" else max(-fwd_intrinsic, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if kind == "call":
        return S * exp(-q * T) * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    if kind == "put":
        return K * exp(-r * T) * norm_cdf(-d2) - S * exp(-q * T) * norm_cdf(-d1)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def black_scholes_greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    """Delta, gamma, vega, theta (per year) and rho of a European option."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    dq, dr = exp(-q * T), exp(-r * T)
    gamma = dq * norm_pdf(d1) / (S * sigma * sqrt(T))
    vega = S * dq * norm_pdf(d1) * sqrt(T)
    common_theta = -S * dq * norm_pdf(d1) * sigma / (2 * sqrt(T))
    if kind == "call":
        delta = dq * norm_cdf(d1)
        theta = common_theta - r * K * dr * norm_cdf(d2) + q * S * dq * norm_cdf(d1)
        rho = K * T * dr * norm_cdf(d2)
    elif kind == "put":
        delta = -dq * norm_cdf(-d1)
        theta = common_theta + r * K * dr * norm_cdf(-d2) - q * S * dq * norm_cdf(-d1)
        rho = -K * T * dr * norm_cdf(-d2)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def geometric_asian_price(S, K, T, r, sigma, n_steps, q=0.0, kind="call"):
    """Exact price of a discretely monitored geometric-average Asian option.

    Averaging is over the n_steps equally spaced dates T/n, 2T/n, ..., T.
    Used as a control variate for the arithmetic Asian option.
    """
    n, dt = n_steps, T / n_steps
    mean_t = dt * (n + 1) / 2.0
    var = sigma**2 * dt * (n + 1) * (2 * n + 1) / (6.0 * n)
    mu = log(S) + (r - q - 0.5 * sigma**2) * mean_t
    sd = sqrt(var)
    d2 = (mu - log(K)) / sd
    d1 = d2 + sd
    fwd = exp(mu + 0.5 * var)
    if kind == "call":
        return exp(-r * T) * (fwd * norm_cdf(d1) - K * norm_cdf(d2))
    if kind == "put":
        return exp(-r * T) * (K * norm_cdf(-d2) - fwd * norm_cdf(-d1))
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
