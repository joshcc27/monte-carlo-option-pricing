"""Monte Carlo Greeks for European options.

Delta and vega are estimated two ways, by pathwise derivatives and by central finite
differences with common random numbers. Gamma, theta and rho use finite differences only.
Every estimate comes with its standard error and the Black-Scholes value.

The terminal price depends only on the sum of the shocks, so ``steps=1`` gives the same
distribution as any finer grid and is the cheapest choice.
"""

import numpy as np

from .bs_analytics import bs_delta, bs_gamma, bs_rho, bs_theta, bs_vega
from .gbm import simulate_gbm_paths
from .mc_results import standard_error
from .payoffs import european_payoff
from .validation import normalise_option_type, validate_positive, validate_positive_int


def _estimate(samples, method):
    return {method: samples.mean(), f"{method}_stderr": standard_error(samples)}


def mc_european_greeks(
    S0,
    K,
    T,
    r,
    sigma,
    *,
    option_type,
    steps,
    n_paths,
    rng,
    h_S=None,
    h_sigma=None,
    h_T=None,
    h_r=None,
):
    """Estimate the delta, vega, gamma, theta and rho of a European option.

    Parameters
    ----------
    steps : int
        Number of time steps per path.
    n_paths : int
        Number of paths.
    rng : RNG or QMCGenerator
        Source of the shocks. One draw is reused for the base run and every bumped run.
    h_S, h_sigma, h_T, h_r : float, optional
        Finite-difference bumps in spot (delta and gamma), volatility (vega), maturity
        (theta) and rate (rho). Each must be positive, and ``h_S``, ``h_sigma`` and ``h_T``
        must be smaller than ``S0``, ``sigma`` and ``T``.

    Returns
    -------
    dict
        Keys ``delta``, ``vega``, ``gamma``, ``theta`` and ``rho``. Each holds
        ``finite_difference``, ``finite_difference_stderr``, ``analytic`` and its bump size.
        Delta and vega also hold ``pathwise`` and ``pathwise_stderr``.
    """
    validate_positive(S0, "S0")
    validate_positive(K, "K")
    validate_positive(T, "T")
    validate_positive(sigma, "sigma")
    validate_positive_int(steps, "steps")
    validate_positive_int(n_paths, "n_paths")
    if rng is None:
        raise ValueError("rng is required")
    option_type = normalise_option_type(option_type)

    h_S = h_S if h_S is not None else 0.01 * S0
    h_sigma = h_sigma if h_sigma is not None else min(0.001, 0.5 * sigma)
    h_T = h_T if h_T is not None else min(1.0 / 252, T * 0.1)
    h_r = h_r if h_r is not None else 1e-3
    for name, h in (("h_S", h_S), ("h_sigma", h_sigma), ("h_T", h_T), ("h_r", h_r)):
        if not h > 0:
            raise ValueError(f"{name} must be positive, got {h}")
    if h_S >= S0:
        raise ValueError("h_S must satisfy 0 < h_S < S0")
    if h_sigma >= sigma:
        raise ValueError("h_sigma must satisfy 0 < h_sigma < sigma")
    if h_T >= T:
        raise ValueError("h_T must satisfy 0 < h_T < T")

    # The base run and every bumped run reuse these shocks (common random numbers).
    shocks = rng.normal(size=(n_paths, steps))
    sim = {"steps": steps, "n_paths": n_paths, "shocks": shocks}

    def payoffs(spot, maturity, rate, vol):
        paths = simulate_gbm_paths(spot, maturity, rate, vol, **sim)
        return np.exp(-rate * maturity) * european_payoff(paths[:, -1], K, option_type)

    ST = simulate_gbm_paths(S0, T, r, sigma, **sim)[:, -1]
    discount = np.exp(-r * T)
    base = discount * european_payoff(ST, K, option_type)

    # Pathwise: the payoff's slope in S_T times dS_T/dS0 = S_T / S0 for delta, and times
    # dS_T/dsigma = S_T (-sigma T + sqrt(dt) * sum of the shocks) for vega.
    if option_type == "call":
        slope = (ST > K).astype(float)
    else:
        slope = -(ST < K).astype(float)
    delta_pw = discount * (slope * (ST / S0))
    dlog_dsigma = -sigma * T + np.sqrt(T / steps) * np.sum(shocks, axis=1)
    vega_pw = discount * (slope * (ST * dlog_dsigma))

    up, down = payoffs(S0 + h_S, T, r, sigma), payoffs(S0 - h_S, T, r, sigma)
    delta_fd = (up - down) / (2 * h_S)
    # The second difference reuses the delta bumps, so gamma needs no extra simulation.
    gamma_fd = (up - 2 * base + down) / h_S**2
    vol_up, vol_down = payoffs(S0, T, r, sigma + h_sigma), payoffs(S0, T, r, sigma - h_sigma)
    vega_fd = (vol_up - vol_down) / (2 * h_sigma)
    # Theta is the derivative in calendar time, so it has the opposite sign to a bump in T.
    theta_fd = -(payoffs(S0, T + h_T, r, sigma) - payoffs(S0, T - h_T, r, sigma)) / (2 * h_T)
    # The rate bump moves both the drift and the discount factor.
    rho_fd = (payoffs(S0, T, r + h_r, sigma) - payoffs(S0, T, r - h_r, sigma)) / (2 * h_r)

    return {
        "delta": {
            **_estimate(delta_pw, "pathwise"),
            **_estimate(delta_fd, "finite_difference"),
            "analytic": bs_delta(S0, K, T, r, sigma, option_type=option_type),
            "h_S": h_S,
        },
        "vega": {
            **_estimate(vega_pw, "pathwise"),
            **_estimate(vega_fd, "finite_difference"),
            "analytic": bs_vega(S0, K, T, r, sigma),
            "h_sigma": h_sigma,
        },
        "gamma": {
            **_estimate(gamma_fd, "finite_difference"),
            "analytic": bs_gamma(S0, K, T, r, sigma),
            "h_S": h_S,
        },
        "theta": {
            **_estimate(theta_fd, "finite_difference"),
            "analytic": bs_theta(S0, K, T, r, sigma, option_type=option_type),
            "h_T": h_T,
        },
        "rho": {
            **_estimate(rho_fd, "finite_difference"),
            "analytic": bs_rho(S0, K, T, r, sigma, option_type=option_type),
            "h_r": h_r,
        },
    }
