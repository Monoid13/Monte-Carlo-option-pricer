# Monte Carlo Option Pricer

A small, vectorised Monte Carlo engine (NumPy only) for pricing equity options
under geometric Brownian motion, with variance reduction, Greeks, and
closed-form benchmarks to check the results against.

## Features

| Contract | Notes |
|---|---|
| European call/put | Checked against Black–Scholes–Merton |
| Asian (arithmetic / geometric) | Discretely monitored; geometric has an exact closed form |
| Barrier (up/down, in/out) | Discretely monitored, optional rebate |
| Lookback (fixed / floating strike) | |
| American call/put | Longstaff–Schwartz least-squares regression |

- **Exact GBM simulation** (log-Euler, no discretisation bias), with an optional dividend yield `q`.
- **Variance reduction**: antithetic variates, and control variates. Arithmetic
  Asians use the closed-form geometric Asian as the control, which typically cuts
  the standard error by more than 10×. All other contracts use the discounted terminal spot.
- **Greeks** (delta, gamma, vega, theta, rho) by central bump-and-reprice using
  common random numbers.
- **Error reporting**: every price comes with its standard error and a 95% confidence interval.
- **Batched simulation**, so memory use stays bounded when you run many paths.

## Install

```bash
pip install -e ".[test]"
```

## Usage

### Python

```python
from mcpricer import GBM, MonteCarloPricer, EuropeanOption, AsianOption, AmericanOption, mc_greeks

model = GBM(S0=100, r=0.05, sigma=0.2, q=0.0)
pricer = MonteCarloPricer(model, n_paths=200_000, n_steps=252,
                          antithetic=True, control_variate=True, seed=42)

print(pricer.price(EuropeanOption(K=100, T=1.0, kind="call")))
print(pricer.price(AsianOption(K=100, T=1.0, average="arithmetic")))
print(pricer.price(AmericanOption(K=100, T=1.0, kind="put")))

print(mc_greeks(pricer, EuropeanOption(K=100, T=1.0)))
```

### Command line

```bash
python -m mcpricer european --greeks --control-variate
```

```bash
python -m mcpricer american --kind put --S0 36 --K 40 --r 0.06 --steps 50
```

```bash
python -m mcpricer barrier --barrier 130 --barrier-type up-and-out --steps 252
```

Run `python -m mcpricer --help` for every option.

Example output:

```
Monte Carlo: price = 10.460195  (s.e. 0.024887, 95% CI [10.411417, 10.508973], 100,000 paths)
Black-Scholes: price = 10.450584

Greek       Monte Carlo   Black-Scholes
delta          0.637494        0.636831
gamma          0.018282        0.018762
vega          37.539357       37.524035
theta         -6.422012       -6.414028
rho           53.298548       53.232482
```

## Method notes

- **Antithetic pairs** are averaged before the standard error is computed, so the
  reported error treats each pair as one independent sample.
- **Control variate**: `Y* = Y − β(X − E[X])`, with β estimated from the sample
  covariance.
- **Longstaff–Schwartz**: working backwards through the time steps, the discounted
  future cash flows of in-the-money paths are regressed on a cubic polynomial in
  `S/K`. A path exercises early when its immediate payoff beats the fitted
  continuation value. The estimate is in-sample, so it carries a small upward bias.
- **Greeks** need a fixed `seed`, so that every bumped re-pricing reuses the same
  random numbers. Theta is quoted per year and vega per unit of volatility. Gamma
  is the noisiest of them; for contracts with discontinuous payoffs (barriers),
  increase `rel_spot_bump`.

## Tests

```bash
pytest
```

The tests check the pricer against Black–Scholes prices and Greeks, the
geometric Asian closed form, put–call parity, barrier in/out parity, and the
Longstaff–Schwartz (2001) benchmark American put (S=36, K=40 → 4.478).
