"""Antithetic variates and control variates.

Antithetic sampling pairs each shock row ``Z`` with ``-Z``. The two paths in a pair are
dependent, so the standard error comes from the ``n_paths / 2`` pair means.

A control variate adjusts target samples ``X`` with correlated samples ``Y`` of known mean,
``X - b (Y - E[Y])`` with ``b = Cov(X, Y) / Var(Y)``. For the arithmetic Asian the control is
the geometric Asian payoff, whose price is known in closed form.
"""

import numpy as np

from .asian_geometric import price_geometric_asian
from .gbm import simulate_gbm_paths
from .mc_results import (
    build_result_antithetic,
    build_result_iid,
    discount_payoffs,
    standard_error,
)
from .payoffs import (
    asian_arithmetic_payoff,
    asian_geometric_payoff,
    barrier_level_for_simulation,
    barrier_payoff,
    european_payoff,
)
from .validation import normalise_option_type


def simulate_gbm_paths_antithetic(S0, T, r, sigma, *, steps, n_paths, rng):
    """Simulate GBM paths from shocks arranged as ``[Z, -Z]``.

    Rows ``i`` and ``i + n_paths // 2`` are an antithetic pair, so ``n_paths`` must be even.
    """
    if n_paths % 2 != 0:
        raise ValueError("Antithetic variates require an even number of paths")
    z = rng.normal(size=(n_paths // 2, steps))
    shocks = np.concatenate([z, -z], axis=0)
    return simulate_gbm_paths(S0, T, r, sigma, steps=steps, n_paths=n_paths, shocks=shocks)


def mc_price_european_antithetic(S0, K, T, r, sigma, *, steps, n_paths, rng, option_type):
    """Price a European option with antithetic sampling. ``n_paths`` must be even."""
    paths = simulate_gbm_paths_antithetic(S0, T, r, sigma, steps=steps, n_paths=n_paths, rng=rng)
    payoffs = european_payoff(paths[:, -1], K, option_type)
    discounted = discount_payoffs(payoffs, T, r)
    return build_result_antithetic(discounted, extra={"variance_reduction": "antithetic"})


def mc_price_asian_arithmetic_antithetic(S0, K, T, r, sigma, *, steps, n_paths, rng, option_type):
    """Price an arithmetic Asian option with antithetic sampling. ``n_paths`` must be even."""
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
    ``mc.mc_price_barrier``. The barrier indicator is not monotone in the shocks, so the gain
    from pairing is usually smaller than for a vanilla option.
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
    """Return ``(X - b (Y - EY), b)``, with ``b`` fitted on the same samples."""
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
        Target samples, for example discounted arithmetic Asian payoffs.
    Y : array-like
        Control samples, the same shape as ``X``.
    EY : float
        Known expectation of ``Y``.

    Returns
    -------
    tuple[float, float, float]
        ``(estimate, stderr, b)`` for the adjusted samples ``X - b (Y - EY)``, with
        ``b = Cov(X, Y) / Var(Y)`` estimated from the same samples. Fitting ``b`` in-sample
        adds a bias of order ``1 / n``, which is negligible at the path counts used here.
    """
    adjusted, b = _control_variate_samples(X, Y, EY)
    return float(adjusted.mean()), float(standard_error(adjusted)), float(b)


def mc_price_asian_arithmetic_cv(S0, K, T, r, sigma, *, steps, n_paths, rng, option_type):
    """Price an arithmetic Asian option with the geometric Asian as control variate.

    The geometric and arithmetic averages of the same path are highly correlated, and the
    geometric Asian has a closed-form price (``asian_geometric.price_geometric_asian``) to use
    as ``E[Y]``.

    Returns
    -------
    dict
        The result dictionary for the adjusted samples. ``extra`` holds
        ``variance_reduction="control_variate"``, ``cv_beta`` (the fitted ``b``) and
        ``geo_asian_price`` (the closed-form control mean).
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
