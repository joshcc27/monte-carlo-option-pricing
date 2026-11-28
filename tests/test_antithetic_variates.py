"""Tests that antithetic sampling takes its standard error over pair means."""

import numpy as np

from mc_pricer.gbm import simulate_gbm_paths
from mc_pricer.mc import mc_price_european
from mc_pricer.payoffs import european_payoff
from mc_pricer.rng import RNG
from mc_pricer.variance_reduction import (
    mc_price_european_antithetic,
    simulate_gbm_paths_antithetic,
)


def test_antithetic_european_stderr_uses_pair_means(european_market):
    p = european_market
    n_paths = 100_000
    result = mc_price_european_antithetic(
        **p, steps=1, n_paths=n_paths, rng=RNG(seed=123), option_type="call"
    )

    paths = simulate_gbm_paths_antithetic(
        p["S0"], p["T"], p["r"], p["sigma"], steps=1, n_paths=n_paths, rng=RNG(seed=123)
    )
    discounted = np.exp(-p["r"] * p["T"]) * european_payoff(paths[:, -1], p["K"], "call")
    n_pairs = n_paths // 2
    pair_means = 0.5 * (discounted[:n_pairs] + discounted[n_pairs:])
    expected_stderr = pair_means.std(ddof=1) / np.sqrt(n_pairs)

    assert np.isclose(result["stderr"], expected_stderr, rtol=0, atol=1e-12)


def test_antithetic_reduces_stderr_for_european_call(european_market):
    # The call payoff is monotone in Z, so X(Z) and X(-Z) are negatively
    # correlated and the pair-mean variance is below half the path variance.
    p = european_market
    n_paths = 100_000
    paths = simulate_gbm_paths(
        p["S0"], p["T"], p["r"], p["sigma"], steps=1, n_paths=n_paths, rng=RNG(seed=1)
    )
    plain = mc_price_european(paths[:, -1], K=p["K"], T=p["T"], r=p["r"], option_type="call")
    anti = mc_price_european_antithetic(
        **p, steps=1, n_paths=n_paths, rng=RNG(seed=2), option_type="call"
    )
    assert anti["stderr"] < plain["stderr"]
