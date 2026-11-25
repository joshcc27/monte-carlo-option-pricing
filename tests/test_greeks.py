"""Tests for the Monte Carlo European Greeks.

European payoffs depend on S_T only and the log-space GBM step is exact, so
``steps=1`` gives the same estimator as a fine grid at a fraction of the cost.
"""

import numpy as np
import pytest

from mc_pricer.greeks import mc_european_greeks
from mc_pricer.rng import RNG

N_PATHS = 300_000


def _greeks(seed, n_paths=N_PATHS, steps=1, **params):
    return mc_european_greeks(steps=steps, n_paths=n_paths, rng=RNG(seed=seed), **params)


def test_pathwise_delta_close_to_analytic():
    params = dict(S0=100.0, K=105.0, r=0.01, T=1.5, sigma=0.3, option_type="call")
    result = _greeks(777, **params)["delta"]
    assert abs(result["pathwise"] - result["analytic"]) < 4 * result["pathwise_stderr"]


def test_pathwise_vega_close_to_analytic():
    params = dict(S0=95.0, K=90.0, r=0.02, T=1.0, sigma=0.2, option_type="call")
    result = _greeks(31415, **params)["vega"]
    assert abs(result["pathwise"] - result["analytic"]) < 4 * result["pathwise_stderr"]
    assert abs(result["pathwise"] - result["analytic"]) / result["analytic"] < 1e-2


def test_finite_difference_delta_close_to_analytic():
    params = dict(S0=100.0, K=95.0, r=0.01, T=1.0, sigma=0.2, option_type="call")
    result = _greeks(123, **params)["delta"]
    assert abs(result["finite_difference"] - result["analytic"]) < 2e-2


def test_finite_difference_vega_close_to_analytic():
    params = dict(S0=100.0, K=100.0, r=0.02, T=1.0, sigma=0.25, option_type="call")
    result = _greeks(456, **params)["vega"]
    rel_error = abs(result["finite_difference"] - result["analytic"]) / result["analytic"]
    assert rel_error < 1e-2


def test_put_branch_pathwise_and_finite_difference():
    params = dict(S0=90.0, K=100.0, r=0.01, T=1.0, sigma=0.22, option_type="put")
    result = _greeks(789, **params)
    delta, vega = result["delta"], result["vega"]
    assert abs(delta["pathwise"] - delta["analytic"]) < 2e-2
    assert abs(delta["finite_difference"] - delta["analytic"]) < 2e-2
    assert abs(vega["pathwise"] - vega["analytic"]) / vega["analytic"] < 1e-2
    assert abs(vega["finite_difference"] - vega["analytic"]) / vega["analytic"] < 1.5e-2


def test_pathwise_vega_does_not_depend_on_step_count():
    # The pathwise vega uses sum(Z) * sqrt(dt), which has the same law for any
    # number of steps. Estimates on 1 and 32 steps agree within noise.
    params = dict(S0=100.0, K=100.0, r=0.02, T=1.0, sigma=0.2, option_type="call")
    one = _greeks(5, n_paths=100_000, steps=1, **params)["vega"]
    many = _greeks(6, n_paths=100_000, steps=32, **params)["vega"]
    combined = np.hypot(one["pathwise_stderr"], many["pathwise_stderr"])
    assert abs(one["pathwise"] - many["pathwise"]) < 4 * combined


def test_crn_finite_difference_delta_tracks_pathwise_on_same_paths():
    # With common random numbers and a small bump, the FD delta differs from
    # the pathwise delta only on paths that finish within h of the strike.
    params = dict(S0=100.0, K=100.0, r=0.02, T=1.0, sigma=0.2, option_type="call")
    delta = _greeks(8, n_paths=100_000, **params)["delta"]
    assert abs(delta["pathwise"] - delta["finite_difference"]) < 0.25 * delta["pathwise_stderr"]


def test_fd_gamma_stderr_grows_as_bump_shrinks():
    # The second difference of a kinked payoff is nonzero only for paths
    # within h of the strike, where it is of size 1/h. Its variance is then
    # O(1/h), so a 10x smaller bump inflates the stderr by about sqrt(10).
    params = dict(S0=100.0, K=100.0, r=0.02, T=1.0, sigma=0.2, option_type="call")
    wide = _greeks(9, n_paths=100_000, h_S=1.0, **params)["gamma"]
    narrow = _greeks(9, n_paths=100_000, h_S=0.1, **params)["gamma"]
    ratio = narrow["finite_difference_stderr"] / wide["finite_difference_stderr"]
    assert 2.0 < ratio < 5.0


def test_gamma_fd_close_to_analytic():
    params = dict(S0=100.0, K=100.0, r=0.02, T=1.0, sigma=0.2, option_type="call")
    result = _greeks(11, **params)["gamma"]
    assert abs(result["finite_difference"] - result["analytic"]) / result["analytic"] < 0.05


def test_theta_fd_close_to_analytic():
    params = dict(S0=100.0, K=100.0, r=0.05, T=1.0, sigma=0.2, option_type="call")
    result = _greeks(22, **params)["theta"]
    assert abs(result["finite_difference"] - result["analytic"]) / abs(result["analytic"]) < 0.05


def test_rho_fd_close_to_analytic():
    params = dict(S0=100.0, K=100.0, r=0.05, T=1.0, sigma=0.2, option_type="call")
    result = _greeks(33, **params)["rho"]
    assert abs(result["finite_difference"] - result["analytic"]) / abs(result["analytic"]) < 0.05


def test_fd_gamma_identical_for_call_and_put_under_crn():
    # Call minus put payoff is S_T - K, which is linear in S0 path by path.
    # Its central second difference is zero, so with common random numbers the
    # call and put gamma estimates must match to rounding error.
    params = dict(S0=100.0, K=100.0, r=0.02, T=1.0, sigma=0.2)
    call = _greeks(0, n_paths=20_000, option_type="call", **params)["gamma"]
    put = _greeks(0, n_paths=20_000, option_type="put", **params)["gamma"]
    assert call["finite_difference"] == pytest.approx(put["finite_difference"], abs=1e-8)


def test_greeks_output_contains_stderr_fields():
    params = dict(S0=100.0, K=100.0, r=0.01, T=1.0, sigma=0.2, option_type="call")
    result = _greeks(42, n_paths=20_000, **params)
    for greek in ("delta", "vega"):
        assert result[greek]["pathwise_stderr"] > 0
        assert result[greek]["finite_difference_stderr"] > 0
    for greek in ("gamma", "theta", "rho"):
        assert {"finite_difference", "finite_difference_stderr", "analytic"} <= set(result[greek])
        assert result[greek]["finite_difference_stderr"] > 0


def test_bump_size_upper_bounds_raise_clear_errors():
    params = dict(S0=100.0, K=100.0, r=0.01, T=1.0, sigma=0.2, option_type="call")
    with pytest.raises(ValueError, match=r"0 < h_S < S0"):
        _greeks(1, n_paths=1_000, h_S=100.0, **params)
    with pytest.raises(ValueError, match=r"0 < h_sigma < sigma"):
        _greeks(1, n_paths=1_000, h_sigma=0.2, **params)
    with pytest.raises(ValueError, match="h_T"):
        _greeks(1, n_paths=1_000, h_T=1.5, **params)


@pytest.mark.parametrize("bump", ["h_S", "h_sigma", "h_T", "h_r"])
@pytest.mark.parametrize("value", [0.0, -0.01])
def test_non_positive_bumps_are_rejected(bump, value):
    params = dict(S0=100.0, K=100.0, r=0.01, T=1.0, sigma=0.2, option_type="call")
    with pytest.raises(ValueError, match=f"{bump} must be positive"):
        _greeks(1, n_paths=1_000, **{bump: value}, **params)
