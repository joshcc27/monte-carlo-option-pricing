"""Tests for the plain Monte Carlo pricers: input checks, the result dictionary, agreement
with Black-Scholes and the 1/sqrt(N) error rate."""

import numpy as np
import pytest

from mc_pricer.bs_analytics import bs_price
from mc_pricer.gbm import simulate_gbm_paths
from mc_pricer.mc import mc_price_asian_arithmetic, mc_price_european
from mc_pricer.rng import RNG
from mc_pricer.variance_reduction import mc_price_european_antithetic

RESULT_KEYS = {"price", "stderr", "ci_low", "ci_high", "n_paths", "extra"}


def terminal_prices(m, n_paths, seed):
    # One step is exact for S_T, since GBM is simulated in log space.
    paths = simulate_gbm_paths(
        m["S0"], m["T"], m["r"], m["sigma"], steps=1, n_paths=n_paths, rng=RNG(seed=seed)
    )
    return paths[:, -1]


def price(m, ST, option_type="call"):
    return mc_price_european(ST, K=m["K"], T=m["T"], r=m["r"], option_type=option_type)


def test_mc_price_european_rejects_non_1d_input():
    with pytest.raises(ValueError, match="ST must be a 1D array-like"):
        mc_price_european(np.array([[100.0, 101.0]]), K=100.0, T=1.0, r=0.02, option_type="call")


def test_mc_price_asian_rejects_non_2d_input():
    with pytest.raises(ValueError, match="paths must be a 2D array-like"):
        mc_price_asian_arithmetic(
            np.array([100.0, 101.0, 102.0]), K=100.0, T=1.0, r=0.02, option_type="call"
        )


def test_plain_and_antithetic_results_have_the_same_keys(european_market):
    m = european_market
    n_paths = 20_000
    plain = price(m, terminal_prices(m, n_paths, seed=11))
    anti = mc_price_european_antithetic(
        **m, steps=1, n_paths=n_paths, rng=RNG(seed=11), option_type="call"
    )
    assert set(plain) == RESULT_KEYS
    assert set(anti) == RESULT_KEYS
    assert anti["extra"]["variance_reduction"] == "antithetic"
    assert anti["extra"]["effective_samples"] == n_paths // 2


def test_mc_price_within_confidence_interval(european_market):
    m = european_market
    result = price(m, terminal_prices(m, 200_000, seed=1234))
    bs_call = bs_price(**m, option_type="call")
    assert abs(result["price"] - bs_call) <= 3 * result["stderr"] + 1e-3


def test_mc_error_and_stderr_scale_with_inverse_sqrt_n(european_market):
    # Separate seeds for the two path counts, so the errors are independent.
    m = european_market
    n_small, n_large = 10_000, 160_000
    expected_ratio = np.sqrt(n_large / n_small)
    bs_call = bs_price(**m, option_type="call")
    small = [price(m, terminal_prices(m, n_small, seed)) for seed in range(10, 16)]
    large = [price(m, terminal_prices(m, n_large, seed)) for seed in range(210, 216)]

    def rmse(results):
        return np.sqrt(np.mean([(res["price"] - bs_call) ** 2 for res in results]))

    rmse_ratio = rmse(small) / rmse(large)
    assert rmse_ratio > 2.0
    assert 0.5 * expected_ratio <= rmse_ratio <= 2.0 * expected_ratio

    se_ratio = np.mean([res["stderr"] for res in small]) / np.mean([res["stderr"] for res in large])
    assert 0.9 * expected_ratio <= se_ratio <= 1.1 * expected_ratio


def test_mc_put_call_parity(european_market):
    # Call and put are priced from the same terminal prices.
    m = european_market
    ST = terminal_prices(m, 200_000, seed=909)
    call, put = price(m, ST, "call"), price(m, ST, "put")
    parity = m["S0"] - m["K"] * np.exp(-m["r"] * m["T"])
    combined_stderr = np.sqrt(call["stderr"] ** 2 + put["stderr"] ** 2)
    assert abs(call["price"] - put["price"] - parity) <= 4 * combined_stderr + 5e-3
