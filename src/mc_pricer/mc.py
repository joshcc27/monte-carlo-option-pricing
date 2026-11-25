"""Plain Monte Carlo pricers.

Each pricer maps simulated prices to payoffs, discounts them, and returns the
standard result dictionary from ``mc_pricer.mc_results``.

The reported ``stderr`` and confidence interval assume i.i.d. samples. They
are not valid when the paths come from a Sobol sequence
(``rng.QMCGenerator``): QMC points are deliberately dependent. For QMC, repeat
the run over several independent scrambles and use the spread of the
estimates (randomised QMC).
"""

import numpy as np

from .mc_results import build_result_iid, discount_payoffs
from .payoffs import (
    asian_arithmetic_payoff,
    barrier_payoff,
    bgk_adjusted_barrier,
    european_payoff,
    validate_barrier,
)


def _mc_price(data, payoff_fn, K, r, T, option_type):
    """Generic Monte Carlo pricing routine parameterised by payoff function.

    Parameters
    ----------
    data : array-like
        Simulation output consumed by ``payoff_fn``.
        For European options this is terminal prices ``ST``; for arithmetic
        Asian options this is the full path matrix.
    payoff_fn : callable
        Function with signature ``payoff_fn(data, K, option_type)`` that
        returns pathwise intrinsic payoffs.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity in years.
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    dict
        Standard Monte Carlo result dictionary from
        ``mc_pricer.mc_results.build_result_iid``.
    """
    # Ensure numeric array input before passing to payoff routines.
    arr = np.asarray(data, dtype=float)
    # Compute intrinsic payoff path by path using selected payoff function.
    payoffs = payoff_fn(arr, K, option_type)
    # Convert maturity payoffs to present values.
    discounted = discount_payoffs(payoffs, r, T)
    # Build unified pricing and uncertainty summary.
    return build_result_iid(discounted)


def mc_price_european(ST, K, r, T, option_type):
    """Price a European option from simulated terminal prices.

    Parameters
    ----------
    ST : array-like
        Terminal prices at maturity for all simulated paths.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity in years.
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    dict
        Standard Monte Carlo result dictionary containing ``price``,
        ``stderr``, confidence interval bounds, sample count, and ``extra``.

    Raises
    ------
    ValueError
        If ``ST`` is not a one-dimensional array-like.
    """
    # Enforce 1D terminal-price convention for European payoffs.
    st_arr = np.asarray(ST, dtype=float)
    if st_arr.ndim != 1:
        raise ValueError("ST must be a 1D array-like of terminal prices")

    return _mc_price(st_arr, european_payoff, K, r, T, option_type)


def barrier_level_for_simulation(
    S0, barrier, barrier_type, T, steps, sigma, continuity_correction
):
    """Validate a barrier contract and return the level to apply to simulated paths.

    Without correction this is ``barrier``. With correction it is the
    Broadie-Glasserman-Kou shifted level for monitoring interval ``T / steps``.

    Raises
    ------
    ValueError
        If the barrier is on the wrong side of ``S0``, ``sigma`` is missing
        when the correction is requested, or the shifted barrier crosses
        ``S0`` (too few monitoring dates for the correction to make sense).
    """
    validate_barrier(barrier_type, barrier, S0)
    if not continuity_correction:
        return barrier
    if sigma is None:
        raise ValueError("sigma is required when continuity_correction=True")
    level = bgk_adjusted_barrier(barrier, sigma, T / steps, barrier_type)
    if (barrier_type.lower().startswith("up") and level <= S0) or (
        barrier_type.lower().startswith("down") and level >= S0
    ):
        raise ValueError(
            f"continuity-corrected barrier {level:.4f} crosses S0={S0}; "
            "increase steps or disable the correction"
        )
    return level


def mc_price_barrier(
    paths, K, r, T, barrier, barrier_type, option_type, *, sigma=None, continuity_correction=False
):
    """Price a knock-in or knock-out option from full simulated paths.

    The barrier is monitored only at the simulated dates. With the default
    ``continuity_correction=False`` the result is the price of a *discretely*
    monitored contract with ``n_times - 1`` equally spaced observations. Set
    ``continuity_correction=True`` (and pass ``sigma``) to approximate the
    continuously monitored contract by shifting the barrier
    (``payoffs.bgk_adjusted_barrier``).

    Parameters
    ----------
    paths : array-like
        Path matrix of shape ``(n_paths, n_times)``; column 0 is ``S0``.
    K : float
        Strike.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Maturity in years.
    barrier : float
        Barrier level. Up barriers must be above ``S0``, down barriers below.
    barrier_type : str
        ``"up_out"``, ``"up_in"``, ``"down_out"`` or ``"down_in"``.
    option_type : str
        ``"call"`` or ``"put"``.
    sigma : float, optional
        Volatility, needed only for the continuity correction.
    continuity_correction : bool, optional
        Apply the Broadie-Glasserman-Kou barrier shift.

    Returns
    -------
    dict
        Standard Monte Carlo result dictionary. ``extra["barrier_used"]`` is
        the level actually applied to the paths.
    """
    path_arr = np.asarray(paths, dtype=float)
    if path_arr.ndim != 2:
        raise ValueError("paths must be a 2D array-like with shape (n_paths, n_times)")
    if path_arr.shape[1] < 2:
        raise ValueError("paths must include at least one monitoring date beyond S0")

    steps = path_arr.shape[1] - 1
    level = barrier_level_for_simulation(
        path_arr[0, 0], barrier, barrier_type, T, steps, sigma, continuity_correction
    )

    payoffs = barrier_payoff(path_arr, K, level, barrier_type, option_type)
    discounted = discount_payoffs(payoffs, r, T)
    return build_result_iid(discounted, extra={"barrier_used": float(level)})


def mc_price_asian_arithmetic(paths, K, r, T, option_type):
    """Price an arithmetic-average Asian option from full simulated paths.

    Parameters
    ----------
    paths : array-like
        Simulated path matrix where rows are paths and columns are times.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity in years.
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    dict
        Standard Monte Carlo result dictionary containing ``price``,
        ``stderr``, confidence interval bounds, sample count, and ``extra``.

    Raises
    ------
    ValueError
        If ``paths`` is not a two-dimensional array-like with at least two
        time columns (initial value + at least one monitoring point).
    """
    # Enforce matrix input: rows are paths, columns are observation times.
    path_arr = np.asarray(paths, dtype=float)
    if path_arr.ndim != 2:
        raise ValueError("paths must be a 2D array-like with shape (n_paths, n_times)")
    # Require at least initial value and one monitored value.
    if path_arr.shape[1] < 2:
        raise ValueError("paths must include at least one monitoring date beyond S0")

    return _mc_price(path_arr, asian_arithmetic_payoff, K, r, T, option_type)
