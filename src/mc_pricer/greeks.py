"""Monte Carlo Greeks for European options.

Delta and vega are estimated two ways: pathwise derivatives and central
finite differences with common random numbers (CRN). Gamma, theta and rho use
finite differences only. Every estimate comes with its standard error and
the Black-Scholes value.

Only European payoffs are supported. The terminal price depends on the sum of
the shocks only, so ``steps=1`` gives the same distribution as any finer grid
and is the cheapest choice.
"""

import numpy as np

from .bs_analytics import bs_delta, bs_gamma, bs_rho, bs_theta, bs_vega
from .gbm import simulate_gbm_paths
from .payoffs import european_payoff
from .validation import normalise_option_type, validate_positive, validate_positive_int


def _pathwise_delta(ST, S0, K, option_type, discount):
    """Discounted pathwise delta samples. Under GBM, dST/dS0 = ST/S0 on every path."""
    dST_dS0 = ST / S0
    if option_type == "call":
        samples = (ST > K).astype(float) * dST_dS0
    else:
        samples = -(ST < K).astype(float) * dST_dS0
    return discount * samples


def _pathwise_vega(ST, sigma, T, steps, shocks, K, option_type, discount):
    """Discounted pathwise vega samples, from the shocks that generated ``ST``."""
    dt = T / steps
    # In log space, d log ST / d sigma = -sigma*T + sqrt(dt) * sum_t Z_t.
    dlog_dsigma = -sigma * T + np.sqrt(dt) * np.sum(shocks, axis=1)
    dST_dsigma = ST * dlog_dsigma
    if option_type == "call":
        samples = (ST > K).astype(float) * dST_dsigma
    else:
        samples = -(ST < K).astype(float) * dST_dsigma
    return discount * samples


def _stderr(samples):
    """Return Monte Carlo standard error of a sample mean estimator."""
    n = samples.size
    return samples.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0


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
    """Estimate European Greeks with pathwise and finite-difference MC.

    Delta and vega are estimated by both pathwise derivatives and central
    finite differences with common random numbers (CRN). Gamma, theta, and
    rho use central finite differences with CRN only (pathwise estimators for
    these are either undefined or require score-function methods).

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (annualised, decimal).
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.
    steps : int
        Number of simulation time increments.
    n_paths : int
        Number of Monte Carlo paths.
    rng : object
        Random source exposing ``normal(size=...)``.
    h_S : float, optional
        Spot bump size for delta and gamma FD. Must satisfy ``0 < h_S < S0``.
    h_sigma : float, optional
        Vol bump size for vega FD. Must satisfy ``0 < h_sigma < sigma``.
    h_T : float, optional
        Maturity bump size for theta FD. Must satisfy ``0 < h_T < T``.
    h_r : float, optional
        Rate bump size for rho FD. Must be positive.

    Bump sizes that are zero or negative raise ``ValueError``; they are not
    silently replaced.

    Returns
    -------
    dict
        Nested result dictionary with keys ``delta``, ``vega``, ``gamma``,
        ``theta``, ``rho``. Delta and vega contain ``pathwise``,
        ``pathwise_stderr``, ``finite_difference``, ``finite_difference_stderr``,
        ``analytic``, and the bump size. Gamma, theta, rho contain
        ``finite_difference``, ``finite_difference_stderr``, ``analytic``,
        and the bump size.

    Raises
    ------
    ValueError
        If core inputs are invalid, ``rng`` is missing, or bump sizes violate
        their required ranges.
    """

    validate_positive(S0, "S0")
    validate_positive(K, "K")
    validate_positive(T, "T")
    validate_positive(sigma, "sigma")
    validate_positive_int(steps, "steps")
    validate_positive_int(n_paths, "n_paths")
    if rng is None:
        raise ValueError("rng is required")

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

    shocks = rng.normal(size=(n_paths, steps))
    # Every bumped run reuses these shocks (common random numbers).
    sim = {"steps": steps, "n_paths": n_paths, "shocks": shocks}
    base_paths = simulate_gbm_paths(S0, T, r, sigma, **sim)
    ST = base_paths[:, -1]
    discount = np.exp(-r * T)

    option_type = normalise_option_type(option_type)

    # --- Delta and Gamma (share the same S0 bumped paths) ---
    delta_samples = _pathwise_delta(ST, S0, K, option_type, discount)
    delta_pw = delta_samples.mean()
    delta_pw_stderr = _stderr(delta_samples)

    paths_up = simulate_gbm_paths(S0 + h_S, T, r, sigma, **sim)
    paths_down = simulate_gbm_paths(S0 - h_S, T, r, sigma, **sim)
    payoffs_up = discount * european_payoff(paths_up[:, -1], K, option_type)
    payoffs_down = discount * european_payoff(paths_down[:, -1], K, option_type)
    payoffs_base = discount * european_payoff(ST, K, option_type)

    delta_fd_samples = (payoffs_up - payoffs_down) / (2 * h_S)
    delta_fd = delta_fd_samples.mean()
    delta_fd_stderr = _stderr(delta_fd_samples)

    # Second central difference reuses the same up/down payoffs, so no extra simulation.
    gamma_fd_samples = (payoffs_up - 2 * payoffs_base + payoffs_down) / (h_S ** 2)
    gamma_fd = gamma_fd_samples.mean()
    gamma_fd_stderr = _stderr(gamma_fd_samples)

    # --- Vega ---
    vega_samples = _pathwise_vega(ST, sigma, T, steps, shocks, K, option_type, discount)
    vega_pw = vega_samples.mean()
    vega_pw_stderr = _stderr(vega_samples)

    paths_sigma_up = simulate_gbm_paths(S0, T, r, sigma + h_sigma, **sim)
    paths_sigma_down = simulate_gbm_paths(S0, T, r, sigma - h_sigma, **sim)
    payoffs_sigma_up = discount * european_payoff(paths_sigma_up[:, -1], K, option_type)
    payoffs_sigma_down = discount * european_payoff(paths_sigma_down[:, -1], K, option_type)
    vega_fd_samples = (payoffs_sigma_up - payoffs_sigma_down) / (2 * h_sigma)
    vega_fd = vega_fd_samples.mean()
    vega_fd_stderr = _stderr(vega_fd_samples)

    # --- Theta: theta = dV/dt = -dV/dT ---
    paths_T_up = simulate_gbm_paths(S0, T + h_T, r, sigma, **sim)
    paths_T_down = simulate_gbm_paths(S0, T - h_T, r, sigma, **sim)
    payoffs_T_up = np.exp(-r * (T + h_T)) * european_payoff(paths_T_up[:, -1], K, option_type)
    payoffs_T_down = np.exp(-r * (T - h_T)) * european_payoff(paths_T_down[:, -1], K, option_type)
    theta_fd_samples = -(payoffs_T_up - payoffs_T_down) / (2 * h_T)
    theta_fd = theta_fd_samples.mean()
    theta_fd_stderr = _stderr(theta_fd_samples)

    # --- Rho: rate bump affects both GBM drift and discount factor ---
    paths_r_up = simulate_gbm_paths(S0, T, r + h_r, sigma, **sim)
    paths_r_down = simulate_gbm_paths(S0, T, r - h_r, sigma, **sim)
    payoffs_r_up = np.exp(-(r + h_r) * T) * european_payoff(paths_r_up[:, -1], K, option_type)
    payoffs_r_down = np.exp(-(r - h_r) * T) * european_payoff(paths_r_down[:, -1], K, option_type)
    rho_fd_samples = (payoffs_r_up - payoffs_r_down) / (2 * h_r)
    rho_fd = rho_fd_samples.mean()
    rho_fd_stderr = _stderr(rho_fd_samples)

    return {
        "delta": {
            "pathwise": delta_pw,
            "pathwise_stderr": delta_pw_stderr,
            "finite_difference": delta_fd,
            "finite_difference_stderr": delta_fd_stderr,
            "analytic": bs_delta(S0, K, T, r, sigma, option_type=option_type),
            "h_S": h_S,
        },
        "vega": {
            "pathwise": vega_pw,
            "pathwise_stderr": vega_pw_stderr,
            "finite_difference": vega_fd,
            "finite_difference_stderr": vega_fd_stderr,
            "analytic": bs_vega(S0, K, T, r, sigma),
            "h_sigma": h_sigma,
        },
        "gamma": {
            "finite_difference": gamma_fd,
            "finite_difference_stderr": gamma_fd_stderr,
            "analytic": bs_gamma(S0, K, T, r, sigma),
            "h_S": h_S,
        },
        "theta": {
            "finite_difference": theta_fd,
            "finite_difference_stderr": theta_fd_stderr,
            "analytic": bs_theta(S0, K, T, r, sigma, option_type=option_type),
            "h_T": h_T,
        },
        "rho": {
            "finite_difference": rho_fd,
            "finite_difference_stderr": rho_fd_stderr,
            "analytic": bs_rho(S0, K, T, r, sigma, option_type=option_type),
            "h_r": h_r,
        },
    }
