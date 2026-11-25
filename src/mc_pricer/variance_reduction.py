"""Variance reduction: antithetic variates and control variates.

- Antithetic variates pair each shock row ``Z`` with ``-Z``. The two paths in
  a pair are dependent, so the standard error is computed from the
  ``n_paths / 2`` pair means (``mc_results.build_result_antithetic``), not
  from the raw paths.
- Control variates adjust a target sample ``X`` with a correlated sample
  ``Y`` of known mean: ``X - b (Y - E[Y])`` with ``b = Cov(X, Y) / Var(Y)``.
  For the arithmetic Asian the control is the geometric Asian payoff, whose
  price is known in closed form.
"""

import numpy as np

from .asian_geometric import price_geometric_asian
from .gbm import simulate_gbm_paths
from .mc import barrier_level_for_simulation
from .mc_results import build_result_antithetic, build_result_iid, discount_payoffs
from .payoffs import (
    asian_arithmetic_payoff,
    asian_geometric_payoff,
    barrier_payoff,
    european_payoff,
)
from .validation import normalise_option_type


def _antithetic_shocks(rng, n_paths, steps):
    """Return shocks of shape ``(n_paths, steps)`` arranged as ``[Z, -Z]``.

    Raises ``ValueError`` if ``n_paths`` is odd.
    """
    if n_paths % 2 != 0:
        raise ValueError("Antithetic variates require an even number of paths")
    half = n_paths // 2
    z_half = rng.normal(size=(half, steps))
    return np.concatenate([z_half, -z_half], axis=0)


def simulate_gbm_paths_antithetic(S0, T, r, sigma, *, steps, n_paths, rng):
    """Simulate GBM paths from antithetic shocks.

    Returns a matrix of shape ``(n_paths, steps + 1)``. Row ``i`` and row
    ``i + n_paths // 2`` are an antithetic pair. ``n_paths`` must be even.
    """
    shocks = _antithetic_shocks(rng, n_paths, steps)
    return simulate_gbm_paths(S0, T, r, sigma, steps=steps, n_paths=n_paths, shocks=shocks)


def mc_price_european_antithetic(S0, K, T, r, sigma, *, steps, n_paths, rng, option_type):
    """Price a European option with antithetic sampling.

    ``n_paths`` counts both members of each pair and must be even. The
    standard error is computed from pair means.
    """
    paths = simulate_gbm_paths_antithetic(S0, T, r, sigma, steps=steps, n_paths=n_paths, rng=rng)
    payoffs = european_payoff(paths[:, -1], K, option_type)
    discounted = discount_payoffs(payoffs, T, r)
    return build_result_antithetic(discounted, extra={"variance_reduction": "antithetic"})


def mc_price_asian_arithmetic_antithetic(
    S0, K, T, r, sigma, *, steps, n_paths, rng, option_type
):
    """Price an arithmetic-average Asian option with antithetic sampling.

    ``n_paths`` counts both members of each pair and must be even. The
    standard error is computed from pair means.
    """
    paths = simulate_gbm_paths_antithetic(S0, T, r, sigma, steps=steps, n_paths=n_paths, rng=rng)
    payoffs = asian_arithmetic_payoff(paths, K, option_type)
    discounted = discount_payoffs(payoffs, T, r)
    return build_result_antithetic(discounted, extra={"variance_reduction": "antithetic"})


def mc_price_barrier_antithetic(
    S0,
    K,
    barrier,
    T,
    r,
    sigma,
    *,
    barrier_type,
    steps,
    n_paths,
    rng,
    option_type,
    continuity_correction=False,
):
    """Price a discretely monitored barrier option with antithetic sampling.

    Monitoring, validation and the optional continuity correction follow
    ``mc.mc_price_barrier``. The barrier indicator is not monotone in the
    shocks, so the gain from antithetic pairing is usually smaller than for a
    vanilla option.
    """
    level = barrier_level_for_simulation(
        S0,
        barrier,
        T,
        sigma,
        barrier_type=barrier_type,
        steps=steps,
        continuity_correction=continuity_correction,
    )
    paths = simulate_gbm_paths_antithetic(S0, T, r, sigma, steps=steps, n_paths=n_paths, rng=rng)
    payoffs = barrier_payoff(paths, K, level, barrier_type, option_type)
    discounted = discount_payoffs(payoffs, T, r)
    return build_result_antithetic(
        discounted, extra={"variance_reduction": "antithetic", "barrier_used": float(level)}
    )


def _control_variate_samples(X, Y, EY):
    """Return ``(X - b (Y - EY), b)`` with ``b`` the sample-optimal coefficient."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    if X.ndim != 1:
        raise ValueError("X and Y must be 1D arrays")
    if X.size == 0:
        raise ValueError("X and Y must be non-empty")

    if X.size > 1:
        centred_Y = Y - Y.mean()
        var_Y = np.sum(centred_Y**2) / (X.size - 1)
        cov = np.sum((X - X.mean()) * centred_Y) / (X.size - 1)
    else:
        var_Y = 0.0
        cov = 0.0
    b = cov / var_Y if var_Y > 0 else 0.0
    return X - b * (Y - EY), b


def control_variate(X, Y, EY):
    """Apply a control-variate adjustment to paired samples.

    Parameters
    ----------
    X : array-like
        Target samples (for example discounted arithmetic Asian payoffs).
    Y : array-like
        Control samples, same shape as ``X``.
    EY : float
        Known expectation of ``Y``.

    Returns
    -------
    tuple[float, float, float]
        ``(estimate, stderr, b)`` for the adjusted samples
        ``X - b (Y - EY)``, with ``b = Cov(X, Y) / Var(Y)`` estimated from the
        same samples. Estimating ``b`` in-sample adds a bias of order
        ``1 / n``, which is negligible at the path counts used here.

    Raises
    ------
    ValueError
        If ``X`` and ``Y`` differ in shape, are not 1D, or are empty.
    """
    adjusted, b = _control_variate_samples(X, Y, EY)
    stderr = adjusted.std(ddof=1) / np.sqrt(adjusted.size) if adjusted.size > 1 else 0.0
    return float(adjusted.mean()), float(stderr), float(b)


def mc_price_asian_arithmetic_cv(S0, K, T, r, sigma, *, steps, n_paths, rng, option_type):
    """Price an arithmetic Asian option with the geometric Asian as control variate.

    The geometric and arithmetic averages of the same path are highly
    correlated, and the geometric Asian has a closed-form price
    (``asian_geometric.price_geometric_asian``), so it serves as ``E[Y]``.

    Returns
    -------
    dict
        Standard result dictionary built from the adjusted samples. ``extra``
        holds ``variance_reduction="control_variate"``, ``cv_beta`` (the
        fitted ``b``) and ``geo_asian_price`` (the closed-form control mean).
    """
    option_type = normalise_option_type(option_type)
    paths = simulate_gbm_paths(S0, T, r, sigma, steps=steps, n_paths=n_paths, rng=rng)

    target = discount_payoffs(asian_arithmetic_payoff(paths, K, option_type), T, r)
    control = discount_payoffs(asian_geometric_payoff(paths, K, option_type), T, r)
    geo_price = price_geometric_asian(S0, K, T, r, sigma, steps=steps, option_type=option_type)

    adjusted, b = _control_variate_samples(target, control, geo_price)
    return build_result_iid(
        adjusted,
        extra={
            "variance_reduction": "control_variate",
            "cv_beta": float(b),
            "geo_asian_price": geo_price,
        },
    )
