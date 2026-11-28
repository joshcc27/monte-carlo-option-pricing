"""Plain Monte Carlo pricers for European, arithmetic Asian and barrier options.

Each pricer turns simulated prices into discounted payoffs and returns the result dictionary
from ``mc_results``. The standard error assumes independent paths, so it does not apply to
Sobol paths from ``rng.QMCGenerator``. For those, price with several independent scrambles
and use the spread of the estimates.
"""

import numpy as np

from .mc_results import build_result_iid, discount_payoffs
from .payoffs import (
    asian_arithmetic_payoff,
    barrier_level_for_simulation,
    barrier_payoff,
    european_payoff,
)


def _path_matrix(paths):
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2:
        raise ValueError("paths must be a 2D array-like with shape (n_paths, n_times)")
    if paths.shape[1] < 2:
        raise ValueError("paths must include at least one monitoring date beyond S0")
    return paths


def mc_price_european(ST, K, T, r, *, option_type):
    """Price a European option from a 1D array of simulated terminal prices ``ST``."""
    ST = np.asarray(ST, dtype=float)
    if ST.ndim != 1:
        raise ValueError("ST must be a 1D array-like of terminal prices")
    payoffs = european_payoff(ST, K, option_type)
    return build_result_iid(discount_payoffs(payoffs, T, r))


def mc_price_asian_arithmetic(paths, K, T, r, *, option_type):
    """Price an arithmetic-average Asian option from paths of shape ``(n_paths, steps + 1)``.

    The average is over the monitoring dates, columns ``1:``, and leaves out ``S0``.
    """
    paths = _path_matrix(paths)
    payoffs = asian_arithmetic_payoff(paths, K, option_type)
    return build_result_iid(discount_payoffs(payoffs, T, r))


def mc_price_barrier(
    paths,
    K,
    barrier,
    T,
    r,
    *,
    barrier_type,
    option_type,
    sigma=None,
    continuity_correction=False,
):
    """Price a knock-in or knock-out option from simulated paths.

    The barrier is checked only on the simulated dates, so by default this is the price of a
    discretely monitored contract. Pass ``sigma`` and ``continuity_correction=True`` to shift
    the barrier (``payoffs.bgk_adjusted_barrier``) and approximate continuous monitoring.

    Parameters
    ----------
    paths : array-like
        Path matrix of shape ``(n_paths, steps + 1)``, with ``S0`` in column 0.
    barrier : float
        Barrier level. Up barriers must be above ``S0`` and down barriers below.
    barrier_type : str
        ``"up_out"``, ``"up_in"``, ``"down_out"`` or ``"down_in"``.
    sigma : float, optional
        Volatility, needed only for the continuity correction.
    continuity_correction : bool, optional
        Apply the Broadie-Glasserman-Kou barrier shift.

    Returns
    -------
    dict
        The result dictionary. ``extra["barrier_used"]`` is the level applied to the paths.
    """
    paths = _path_matrix(paths)
    level = barrier_level_for_simulation(
        paths[0, 0],
        barrier,
        T,
        sigma,
        barrier_type=barrier_type,
        steps=paths.shape[1] - 1,
        continuity_correction=continuity_correction,
    )
    payoffs = barrier_payoff(paths, K, level, barrier_type, option_type)
    return build_result_iid(discount_payoffs(payoffs, T, r), extra={"barrier_used": float(level)})
