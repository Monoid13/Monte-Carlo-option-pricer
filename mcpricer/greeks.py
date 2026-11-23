"""Monte Carlo Greeks by bump-and-reprice with common random numbers."""

from dataclasses import replace

from .pricer import MonteCarloPricer

ONE_DAY = 1.0 / 365.0


def mc_greeks(pricer: MonteCarloPricer, option, rel_spot_bump=0.01, vol_bump=0.01, rate_bump=0.0001):
    """Central finite-difference Greeks.

    The pricer must have a fixed seed so every re-pricing reuses the same
    random numbers; otherwise the differences are swamped by noise.
    Theta is per year (the 1-day forward difference, annualised), vega is per
    unit of volatility and rho per unit of rate, matching black_scholes_greeks.
    """
    if pricer.seed is None:
        raise ValueError("mc_greeks needs a pricer with a fixed seed (common random numbers)")
    m = pricer.model

    def px(model=m, opt=option):
        return pricer.with_model(model).price(opt).price

    base = px()
    h = rel_spot_bump * m.S0
    up, down = px(m.bump(S0=m.S0 + h)), px(m.bump(S0=m.S0 - h))
    greeks = {
        "price": base,
        "delta": (up - down) / (2 * h),
        "gamma": (up - 2 * base + down) / h**2,
        "vega": (px(m.bump(sigma=m.sigma + vol_bump)) - px(m.bump(sigma=m.sigma - vol_bump))) / (2 * vol_bump),
        "rho": (px(m.bump(r=m.r + rate_bump)) - px(m.bump(r=m.r - rate_bump))) / (2 * rate_bump),
    }
    if option.T > ONE_DAY:
        greeks["theta"] = (px(opt=replace(option, T=option.T - ONE_DAY)) - base) / ONE_DAY
    return greeks
