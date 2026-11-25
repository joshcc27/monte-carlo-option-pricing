"""Focused tests for antithetic-variates implementation details.

This file verifies that antithetic Monte Carlo reports uncertainty using
pair means (the correct effective sampling unit) rather than raw path count.
"""

import numpy as np

from mc_pricer.payoffs import european_payoff
from mc_pricer.variance_reduction import (
    mc_price_european_antithetic,
    simulate_gbm_paths_antithetic,
)


def test_antithetic_european_stderr_uses_pair_means(european_market, make_rng):
    # Build one antithetic result through the public wrapper.
    params = dict(european_market)
    steps = 1
    n_paths = 100_000

    rng_for_wrapper = make_rng(123)
    result = mc_price_european_antithetic(
        **params, steps=steps, n_paths=n_paths, rng=rng_for_wrapper, option_type="call"
    )

    # Recreate the same antithetic sample manually and compute paired stderr
    # directly to confirm implementation-level correctness.
    rng_for_manual = make_rng(123)
    paths = simulate_gbm_paths_antithetic(
        S0=params["S0"],
        T=params["T"],
        r=params["r"],
        sigma=params["sigma"],
        steps=steps,
        n_paths=n_paths,
        rng=rng_for_manual,
    )
    ST = paths[:, -1]
    discounted = np.exp(-params["r"] * params["T"]) * european_payoff(ST, params["K"], "call")
    n_pairs = n_paths // 2
    pair_means = 0.5 * (discounted[:n_pairs] + discounted[n_pairs:])
    expected_stderr = pair_means.std(ddof=1) / np.sqrt(n_pairs)

    # Wrapper stderr should match manual paired estimate exactly.
    assert np.isclose(result["stderr"], expected_stderr, rtol=0, atol=1e-12)


def test_antithetic_reduces_stderr_for_european_call(european_market, make_rng):
    # The call payoff is monotone in Z, so X(Z) and X(-Z) are negatively
    # correlated and the pair-mean variance is below half the path variance.
    from mc_pricer.gbm import simulate_gbm_paths
    from mc_pricer.mc import mc_price_european

    p = european_market
    n_paths = 100_000
    paths = simulate_gbm_paths(
        S0=p["S0"], T=p["T"], r=p["r"], sigma=p["sigma"], steps=1, n_paths=n_paths, rng=make_rng(1)
    )
    plain = mc_price_european(paths[:, -1], K=p["K"], T=p["T"], r=p["r"], option_type="call")
    anti = mc_price_european_antithetic(
        **p, steps=1, n_paths=n_paths, rng=make_rng(2), option_type="call"
    )
    assert anti["stderr"] < plain["stderr"]
