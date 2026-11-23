import numpy as np
import pytest

from mcpricer import (
    GBM, AmericanOption, AsianOption, BarrierOption, EuropeanOption, LookbackOption,
    MonteCarloPricer, black_scholes_greeks, black_scholes_price, mc_greeks,
)
from mcpricer.__main__ import main
from mcpricer.analytic import geometric_asian_price

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2
MODEL = GBM(S0, R, SIGMA)


def within(result, target, n_se=4):
    return abs(result.price - target) < n_se * result.std_error


# ----------------------------------------------------------------- analytic
def test_put_call_parity():
    for q in (0.0, 0.03):
        c = black_scholes_price(S0, 110, T, R, SIGMA, q, "call")
        p = black_scholes_price(S0, 110, T, R, SIGMA, q, "put")
        assert c - p == pytest.approx(S0 * np.exp(-q * T) - 110 * np.exp(-R * T))


def test_black_scholes_reference_value():
    assert black_scholes_price(S0, K, T, R, SIGMA) == pytest.approx(10.450583572, abs=1e-8)


# ----------------------------------------------------------------- european
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("antithetic", [False, True])
@pytest.mark.parametrize("cv", [False, True])
def test_european_matches_black_scholes(kind, antithetic, cv):
    pricer = MonteCarloPricer(MODEL, 100_000, antithetic=antithetic, control_variate=cv, seed=7)
    res = pricer.price(EuropeanOption(K, T, kind))
    assert within(res, black_scholes_price(S0, K, T, R, SIGMA, kind=kind))


def test_dividend_yield():
    model = GBM(S0, R, SIGMA, q=0.03)
    res = MonteCarloPricer(model, 100_000, seed=1).price(EuropeanOption(105, T))
    assert within(res, black_scholes_price(S0, 105, T, R, SIGMA, 0.03))


def test_variance_reduction_lowers_std_error():
    opt = EuropeanOption(K, T)
    plain = MonteCarloPricer(MODEL, 100_000, antithetic=False, seed=3).price(opt)
    both = MonteCarloPricer(MODEL, 100_000, antithetic=True, control_variate=True, seed=3).price(opt)
    assert both.std_error < 0.8 * plain.std_error


def test_reproducible_with_seed_and_batching_invariant():
    opt = AsianOption(K, T)
    a = MonteCarloPricer(MODEL, 20_000, 12, seed=11).price(opt)
    b = MonteCarloPricer(MODEL, 20_000, 12, seed=11).price(opt)
    assert a.price == b.price
    c = MonteCarloPricer(MODEL, 20_000, 12, seed=11, batch_size=3_000).price(opt)
    assert np.isfinite(c.price) and within(c, a.price, n_se=6)


# -------------------------------------------------------------------- asian
def test_geometric_asian_matches_closed_form():
    res = MonteCarloPricer(MODEL, 100_000, 12, seed=2).price(AsianOption(K, T, average="geometric"))
    assert within(res, geometric_asian_price(S0, K, T, R, SIGMA, 12))


def test_arithmetic_asian_control_variate():
    opt = AsianOption(K, T)
    plain = MonteCarloPricer(MODEL, 100_000, 12, seed=4).price(opt)
    cv = MonteCarloPricer(MODEL, 100_000, 12, seed=4, control_variate=True).price(opt)
    assert cv.std_error < plain.std_error / 10
    assert abs(cv.price - plain.price) < 4 * plain.std_error
    # Arithmetic mean >= geometric mean, so the call is worth at least the geometric one.
    assert cv.price > geometric_asian_price(S0, K, T, R, SIGMA, 12)


# ------------------------------------------------------------------ barrier
def test_in_out_parity():
    pricer = MonteCarloPricer(MODEL, 50_000, 50, seed=5)
    out = pricer.price(BarrierOption(K, T, barrier=120, barrier_type="up-and-out"))
    knock_in = pricer.price(BarrierOption(K, T, barrier=120, barrier_type="up-and-in"))
    vanilla = MonteCarloPricer(MODEL, 50_000, 50, seed=5).price(EuropeanOption(K, T))
    # Same seed but different step counts, so compare within noise.
    assert abs(out.price + knock_in.price - vanilla.price) < 4 * vanilla.std_error


def test_barrier_far_away_equals_vanilla():
    res = MonteCarloPricer(MODEL, 50_000, 20, seed=6).price(
        BarrierOption(K, T, barrier=1e6, barrier_type="up-and-out"))
    assert within(res, black_scholes_price(S0, K, T, R, SIGMA))


def test_barrier_rejects_bad_type():
    with pytest.raises(ValueError):
        MonteCarloPricer(MODEL, 100, 2, seed=0).price(BarrierOption(K, T, barrier=120, barrier_type="sideways"))


# ----------------------------------------------------------------- lookback
def test_lookback_dominates_european():
    pricer = MonteCarloPricer(MODEL, 50_000, 50, seed=8)
    euro = black_scholes_price(S0, K, T, R, SIGMA)
    assert pricer.price(LookbackOption(K, T)).price > euro
    assert pricer.price(LookbackOption(K, T, floating=True)).price > euro


# ----------------------------------------------------------------- american
def test_american_put_reference():
    # Longstaff & Schwartz (2001) table 1: S=36, K=40, r=6%, sigma=20%, T=1 -> 4.478
    model = GBM(36, 0.06, 0.2)
    res = MonteCarloPricer(model, 100_000, 50, seed=9).price(AmericanOption(40, 1.0, "put"))
    assert res.price == pytest.approx(4.478, abs=0.03)
    assert res.price > black_scholes_price(36, 40, 1.0, 0.06, 0.2, kind="put")


def test_american_call_no_dividend_equals_european():
    res = MonteCarloPricer(MODEL, 50_000, 50, seed=10).price(AmericanOption(K, T, "call"))
    assert within(res, black_scholes_price(S0, K, T, R, SIGMA), n_se=5)


# ------------------------------------------------------------------- greeks
@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_black_scholes(kind):
    pricer = MonteCarloPricer(MODEL, 200_000, 1, seed=12)
    mc = mc_greeks(pricer, EuropeanOption(K, T, kind))
    bs = black_scholes_greeks(S0, K, T, R, SIGMA, kind=kind)
    assert mc["delta"] == pytest.approx(bs["delta"], abs=0.01)
    assert mc["gamma"] == pytest.approx(bs["gamma"], rel=0.1)
    assert mc["vega"] == pytest.approx(bs["vega"], rel=0.02)
    assert mc["rho"] == pytest.approx(bs["rho"], rel=0.02)
    assert mc["theta"] == pytest.approx(bs["theta"], rel=0.05)


def test_greeks_require_seed():
    with pytest.raises(ValueError):
        mc_greeks(MonteCarloPricer(MODEL, 1000), EuropeanOption(K, T))


# --------------------------------------------------------------- validation
def test_input_validation():
    with pytest.raises(ValueError):
        GBM(-1, R, SIGMA)
    with pytest.raises(ValueError):
        EuropeanOption(K, T, "straddle")
    with pytest.raises(ValueError):
        EuropeanOption(K, 0)


def test_cli_runs(capsys):
    main(["european", "--paths", "20000", "--greeks"])
    out = capsys.readouterr().out
    assert "Monte Carlo" in out and "Black-Scholes" in out and "delta" in out
