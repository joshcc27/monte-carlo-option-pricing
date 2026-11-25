"""Tests for the Sobol generator and the Brownian-bridge path construction."""

import numpy as np
import pytest

from mc_pricer.bs_analytics import bs_price
from mc_pricer.gbm import simulate_gbm_paths
from mc_pricer.mc import mc_price_asian_arithmetic, mc_price_european
from mc_pricer.rng import RNG, QMCGenerator, brownian_bridge_increments
from mc_pricer.variance_reduction import mc_price_asian_arithmetic_cv


def test_qmc_generator_output_shape():
    qmc = QMCGenerator(dim=8, seed=0)
    out = qmc.normal(size=(16, 8))
    assert out.shape == (16, 8)


def test_qmc_generator_approximately_standard_normal():
    qmc = QMCGenerator(dim=64, seed=1)
    out = qmc.normal(size=(4096, 64))
    assert abs(out.mean()) < 0.05
    assert abs(out.std() - 1.0) < 0.05


def test_qmc_generator_rejects_wrong_dim():
    qmc = QMCGenerator(dim=8, seed=0)
    with pytest.raises(ValueError, match="dim=8"):
        qmc.normal(size=(16, 16))


def test_qmc_reduces_pricing_error_vs_mc():
    # RMSE over 15 scrambles vs RMSE over 15 pseudo-random seeds, same N.
    S0, K, r, T, sigma = 100.0, 100.0, 0.02, 1.0, 0.2
    steps = 64
    n_paths = 1024  # power of two
    bs_call = bs_price(S0, K, T, r, sigma, "call")

    mc_errors = []
    for seed in range(15):
        paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, rng=RNG(seed=seed))
        mc_errors.append(mc_price_european(paths[:, -1], K, r, T, "call")["price"] - bs_call)

    qmc_errors = []
    for seed in range(15):
        qmc = QMCGenerator(dim=steps, seed=seed)
        paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, rng=qmc)
        qmc_errors.append(mc_price_european(paths[:, -1], K, r, T, "call")["price"] - bs_call)

    mc_rmse = np.sqrt(np.mean(np.square(mc_errors)))
    qmc_rmse = np.sqrt(np.mean(np.square(qmc_errors)))
    assert qmc_rmse < mc_rmse


def test_brownian_bridge_terminal_value_is_first_coordinate():
    # Increments times sqrt(dt) sum to W(1), which the bridge takes from z[:, 0].
    z = np.random.default_rng(0).normal(size=(50, 16))
    increments = brownian_bridge_increments(z)
    np.testing.assert_allclose(increments.sum(axis=1) * np.sqrt(1 / 16), z[:, 0])


@pytest.mark.parametrize("steps", [8, 12])
def test_brownian_bridge_preserves_iid_normal_increments(steps):
    # Also valid when steps is not a power of two: output covariance is the identity.
    z = np.random.default_rng(1).normal(size=(200_000, steps))
    cov = np.cov(brownian_bridge_increments(z), rowvar=False)
    np.testing.assert_allclose(cov, np.eye(steps), atol=0.015)


def test_brownian_bridge_qmc_beats_time_ordered_qmc_for_asian():
    # With 64 time steps, time-ordered Sobol coordinates spread the path
    # variance over all dimensions; the bridge concentrates it in the first few.
    S0, K, r, T, sigma, steps, n = 100.0, 100.0, 0.02, 1.0, 0.2, 64, 2048
    reference = mc_price_asian_arithmetic_cv(
        S0, K, r, T, sigma, steps, 400_000, RNG(seed=0), "call"
    )["price"]

    def rmse(bridge):
        errors = []
        for seed in range(12):
            qmc = QMCGenerator(dim=steps, seed=seed, brownian_bridge=bridge)
            paths = simulate_gbm_paths(S0, r, sigma, T, steps, n, rng=qmc)
            errors.append(mc_price_asian_arithmetic(paths, K, r, T, "call")["price"] - reference)
        return np.sqrt(np.mean(np.square(errors)))

    assert rmse(True) < rmse(False)
