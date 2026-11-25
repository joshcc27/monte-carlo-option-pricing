"""Tests for the geometric Asian closed form and the control-variate pricer."""

import numpy as np
import pytest

from mc_pricer.asian_geometric import geometric_parameters, monitoring_times, price_geometric_asian
from mc_pricer.bs_analytics import bs_price
from mc_pricer.gbm import simulate_gbm_paths
from mc_pricer.mc import mc_price_asian_arithmetic
from mc_pricer.mc_results import build_result_iid, discount_payoffs
from mc_pricer.payoffs import asian_arithmetic_payoff, asian_geometric_payoff
from mc_pricer.variance_reduction import control_variate, mc_price_asian_arithmetic_cv

# ---------------------------------------------------------------------------
# Geometric Asian closed form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_geometric_asian_with_one_fixing_is_black_scholes(option_type):
    # One monitoring date at T: the "average" is S_T, so the price is Black-Scholes.
    geo = price_geometric_asian(
        S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, steps=1, option_type=option_type
    )
    bs = bs_price(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type=option_type)
    assert geo == pytest.approx(bs, abs=1e-10)
    if option_type == "call":
        assert geo == pytest.approx(10.4506, abs=1e-4)


def test_geometric_parameters_match_brute_force_double_sum():
    times = np.array([0.1, 0.35, 0.5, 0.9, 1.2])
    r, sigma = 0.03, 0.3
    mu_g, sigma_g = geometric_parameters(times, r, sigma)
    n = times.size
    expected_var = sigma**2 / n**2 * np.minimum.outer(times, times).sum()
    assert mu_g == pytest.approx((r - 0.5 * sigma**2) * times.mean())
    assert sigma_g**2 == pytest.approx(expected_var)


def test_geometric_asian_approaches_continuous_average_limit():
    # Continuous geometric average (Kemna-Vorst): log G has variance sigma^2 T / 3
    # and mean (r - sigma^2/2) T / 2.
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    var_c = sigma**2 * T / 3
    mean_c = np.log(S0) + 0.5 * (r - 0.5 * sigma**2) * T
    from scipy.stats import norm

    d1 = (mean_c - np.log(K) + var_c) / np.sqrt(var_c)
    d2 = d1 - np.sqrt(var_c)
    kv = np.exp(-r * T) * (np.exp(mean_c + 0.5 * var_c) * norm.cdf(d1) - K * norm.cdf(d2))
    price = price_geometric_asian(S0, K, T, r, sigma, steps=20_000, option_type="call")
    assert price == pytest.approx(kv, abs=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_geometric_closed_form_matches_monte_carlo(asian_market, make_rng, option_type):
    m = asian_market
    steps, n_paths = 12, 400_000
    paths = simulate_gbm_paths(
        m["S0"], m["T"], m["r"], m["sigma"], steps=steps, n_paths=n_paths, rng=make_rng(5)
    )
    discounted = discount_payoffs(
        asian_geometric_payoff(paths, m["K"], option_type), T=m["T"], r=m["r"]
    )
    mc = build_result_iid(discounted)
    closed = price_geometric_asian(**m, steps=steps, option_type=option_type)
    assert abs(mc["price"] - closed) < 3 * mc["stderr"]


def test_monitoring_times_exclude_zero():
    np.testing.assert_allclose(monitoring_times(1.0, 4), [0.25, 0.5, 0.75, 1.0])


# ---------------------------------------------------------------------------
# Control variate
# ---------------------------------------------------------------------------


def test_control_variate_helper_matches_manual_formula():
    rng = np.random.default_rng(0)
    y = rng.normal(size=1000)
    x = 2.0 * y + rng.normal(scale=0.1, size=1000)
    est, stderr, b = control_variate(x, y, 0.0)
    b_manual = np.cov(x, y, ddof=1)[0, 1] / np.var(y, ddof=1)
    adjusted = x - b_manual * y
    assert b == pytest.approx(b_manual)
    assert est == pytest.approx(adjusted.mean())
    assert stderr == pytest.approx(adjusted.std(ddof=1) / np.sqrt(1000))


def test_control_variate_rejects_empty_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        control_variate([], [], 0.0)


def test_control_variate_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        control_variate([1.0, 2.0], [1.0], 0.0)


def test_cv_result_schema(asian_market, make_rng):
    m = asian_market
    result = mc_price_asian_arithmetic_cv(
        **m, steps=32, n_paths=10_000, rng=make_rng(1), option_type="call"
    )
    assert set(result) == {"price", "stderr", "ci_low", "ci_high", "n_paths", "extra"}
    assert result["n_paths"] == 10_000
    assert result["extra"]["variance_reduction"] == "control_variate"
    assert result["extra"]["geo_asian_price"] == pytest.approx(
        price_geometric_asian(**m, steps=32, option_type="call")
    )
    assert result["ci_low"] < result["price"] < result["ci_high"]


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_cv_price_agrees_with_plain_mc_and_reduces_stderr(asian_market, make_rng, option_type):
    # Independent seeds, so the two estimates are independent and their
    # difference has variance stderr_plain^2 + stderr_cv^2.
    m = asian_market
    steps, n_paths = 64, 200_000
    paths = simulate_gbm_paths(
        S0=m["S0"],
        T=m["T"],
        r=m["r"],
        sigma=m["sigma"],
        steps=steps,
        n_paths=n_paths,
        rng=make_rng(77),
    )
    plain = mc_price_asian_arithmetic(
        paths, K=m["K"], T=m["T"], r=m["r"], option_type=option_type
    )
    cv = mc_price_asian_arithmetic_cv(
        **m, steps=steps, n_paths=n_paths, rng=make_rng(78), option_type=option_type
    )
    combined = np.hypot(plain["stderr"], cv["stderr"])
    assert abs(cv["price"] - plain["price"]) < 3 * combined
    assert plain["stderr"] / cv["stderr"] > 5


def test_cv_same_paths_gives_large_variance_reduction(asian_market, make_rng):
    # On the same paths the arithmetic and geometric payoffs are almost
    # perfectly correlated for this contract.
    m = asian_market
    paths = simulate_gbm_paths(
        S0=m["S0"], T=m["T"], r=m["r"], sigma=m["sigma"], steps=64, n_paths=100_000, rng=make_rng(3)
    )
    x = discount_payoffs(asian_arithmetic_payoff(paths, m["K"], "call"), T=m["T"], r=m["r"])
    y = discount_payoffs(asian_geometric_payoff(paths, m["K"], "call"), T=m["T"], r=m["r"])
    ey = price_geometric_asian(**m, steps=64, option_type="call")
    _, stderr_cv, b = control_variate(x, y, ey)
    plain_stderr = x.std(ddof=1) / np.sqrt(x.size)
    assert 0.9 < b < 1.2
    assert (plain_stderr / stderr_cv) ** 2 > 100
