"""Tests for GBM path simulation: shape, input checks and terminal moments."""

import numpy as np
import pytest

from mc_pricer.gbm import simulate_gbm_paths
from mc_pricer.rng import RNG


def test_simulate_gbm_paths_shape_and_initial_level():
    S0, T, r, sigma, steps, n_paths = 100.0, 1.0, 0.02, 0.2, 8, 16
    paths = simulate_gbm_paths(S0, T, r, sigma, steps=steps, n_paths=n_paths, rng=RNG(seed=7))

    assert paths.shape == (n_paths, steps + 1)
    assert np.allclose(paths[:, 0], S0)


def test_simulate_gbm_paths_with_explicit_shocks_matches_formula():
    # With zero shocks every path is the drift alone.
    S0, T, r, sigma, steps, n_paths = 100.0, 1.0, 0.05, 0.2, 4, 3
    shocks = np.zeros((n_paths, steps))

    paths = simulate_gbm_paths(S0, T, r, sigma, steps=steps, n_paths=n_paths, shocks=shocks)

    dt = T / steps
    drift = (r - 0.5 * sigma * sigma) * dt
    expected = S0 * np.exp(np.arange(steps + 1) * drift)

    assert np.allclose(paths, expected[None, :])


def test_simulate_gbm_paths_rejects_float_steps():
    with pytest.raises(TypeError, match="steps must be an integer"):
        simulate_gbm_paths(
            S0=100.0, T=1.0, r=0.05, sigma=0.2, steps=10.0, n_paths=100, rng=RNG(seed=1)
        )


def test_simulate_gbm_paths_rejects_float_n_paths():
    with pytest.raises(TypeError, match="n_paths must be an integer"):
        simulate_gbm_paths(
            S0=100.0, T=1.0, r=0.05, sigma=0.2, steps=10, n_paths=100.0, rng=RNG(seed=1)
        )


def test_simulate_gbm_paths_rejects_invalid_shocks_shape():
    with pytest.raises(ValueError, match=r"shocks must have shape"):
        simulate_gbm_paths(
            S0=100.0,
            r=0.03,
            sigma=0.25,
            T=1.0,
            steps=5,
            n_paths=10,
            shocks=np.zeros((10, 4)),
        )


def test_simulate_gbm_paths_terminal_moments_match_gbm_theory():
    # E[S_T] = S0 e^(rT) and Var[S_T] = S0^2 e^(2rT) (e^(sigma^2 T) - 1).
    S0, T, r, sigma = 100.0, 1.0, 0.03, 0.2
    paths = simulate_gbm_paths(S0, T, r, sigma, steps=16, n_paths=200_000, rng=RNG(seed=123))
    ST = paths[:, -1]

    mean = S0 * np.exp(r * T)
    var = S0**2 * np.exp(2 * r * T) * (np.exp(sigma * sigma * T) - 1.0)
    assert abs(ST.mean() - mean) / mean < 0.01
    assert abs(ST.var() - var) / var < 0.03
