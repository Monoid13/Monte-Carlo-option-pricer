"""Command-line interface: python -m mcpricer --help"""

import argparse
import time

from .analytic import black_scholes_greeks, black_scholes_price, geometric_asian_price
from .greeks import mc_greeks
from .models import GBM
from .options import AmericanOption, AsianOption, BarrierOption, EuropeanOption, LookbackOption
from .pricer import MonteCarloPricer


def build_parser():
    p = argparse.ArgumentParser(prog="mcpricer", description="Monte Carlo option pricer (GBM).")
    p.add_argument("style", choices=["european", "asian", "barrier", "lookback", "american"])
    p.add_argument("--kind", choices=["call", "put"], default="call")
    p.add_argument("--S0", type=float, default=100.0, help="spot price")
    p.add_argument("--K", type=float, default=100.0, help="strike")
    p.add_argument("--T", type=float, default=1.0, help="maturity in years")
    p.add_argument("--r", type=float, default=0.05, help="risk-free rate (cont. comp.)")
    p.add_argument("--sigma", type=float, default=0.2, help="volatility")
    p.add_argument("--q", type=float, default=0.0, help="dividend yield")
    p.add_argument("--paths", type=int, default=100_000)
    p.add_argument("--steps", type=int, default=252)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-antithetic", action="store_true")
    p.add_argument("--control-variate", action="store_true")
    p.add_argument("--greeks", action="store_true", help="also estimate Greeks")
    p.add_argument("--average", choices=["arithmetic", "geometric"], default="arithmetic")
    p.add_argument("--barrier", type=float, default=120.0)
    p.add_argument("--barrier-type", default="up-and-out",
                   choices=["up-and-out", "down-and-out", "up-and-in", "down-and-in"])
    p.add_argument("--rebate", type=float, default=0.0)
    p.add_argument("--floating", action="store_true", help="floating-strike lookback")
    return p


def make_option(a):
    common = dict(K=a.K, T=a.T, kind=a.kind)
    return {
        "european": lambda: EuropeanOption(**common),
        "asian": lambda: AsianOption(**common, average=a.average),
        "barrier": lambda: BarrierOption(**common, barrier=a.barrier,
                                         barrier_type=a.barrier_type, rebate=a.rebate),
        "lookback": lambda: LookbackOption(**common, floating=a.floating),
        "american": lambda: AmericanOption(**common),
    }[a.style]()


def main(argv=None):
    a = build_parser().parse_args(argv)
    model = GBM(S0=a.S0, r=a.r, sigma=a.sigma, q=a.q)
    option = make_option(a)
    pricer = MonteCarloPricer(model, n_paths=a.paths, n_steps=a.steps,
                              antithetic=not a.no_antithetic,
                              control_variate=a.control_variate, seed=a.seed)

    t0 = time.perf_counter()
    result = pricer.price(option)
    print(f"{option}\n{model}\n")
    print(f"Monte Carlo: {result}  [{time.perf_counter() - t0:.2f}s]")

    if a.style == "european":
        print(f"Black-Scholes: price = {black_scholes_price(a.S0, a.K, a.T, a.r, a.sigma, a.q, a.kind):.6f}")
    elif a.style == "asian" and a.average == "geometric":
        exact = geometric_asian_price(a.S0, a.K, a.T, a.r, a.sigma, a.steps, a.q, a.kind)
        print(f"Closed form:   price = {exact:.6f}")
    elif a.style == "american":
        eu = black_scholes_price(a.S0, a.K, a.T, a.r, a.sigma, a.q, a.kind)
        print(f"European equivalent (BS): {eu:.6f}  -> early-exercise premium ~ {result.price - eu:.6f}")

    if a.greeks:
        g = mc_greeks(pricer, option)
        bs = black_scholes_greeks(a.S0, a.K, a.T, a.r, a.sigma, a.q, a.kind) if a.style == "european" else {}
        print("\nGreek       Monte Carlo" + ("   Black-Scholes" if bs else ""))
        for name in ("delta", "gamma", "vega", "theta", "rho"):
            if name in g:
                line = f"{name:<10}{g[name]:>13.6f}"
                if name in bs:
                    line += f"{bs[name]:>16.6f}"
                print(line)


if __name__ == "__main__":
    main()
