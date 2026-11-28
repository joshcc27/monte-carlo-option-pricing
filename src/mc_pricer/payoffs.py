"""Payoffs of simulated prices, before discounting, and the barrier checks they rely on.

Path-dependent payoffs take a path matrix of shape ``(n_paths, steps + 1)`` with ``S0`` in
column 0 and the monitoring dates in columns ``1:``.
"""

import numpy as np

from .validation import normalise_option_type

# Broadie-Glasserman-Kou constant: -zeta(1/2) / sqrt(2 * pi).
BGK_BETA = 0.5826

BARRIER_TYPES = ("up_out", "up_in", "down_out", "down_in")


def european_payoff(ST, K, option_type):
    """Return ``max(ST - K, 0)`` for a call or ``max(K - ST, 0)`` for a put."""
    t = normalise_option_type(option_type)
    if t == "call":
        return np.maximum(ST - K, 0.0)
    return np.maximum(K - ST, 0.0)


def asian_arithmetic_payoff(paths, K, option_type):
    """Return the payoff on the arithmetic average of columns ``1:``, which leaves out ``S0``."""
    if paths.shape[1] < 2:
        raise ValueError("paths must include at least one monitoring date beyond S0")
    avg_prices = paths[:, 1:].mean(axis=1)
    return european_payoff(avg_prices, K, option_type)


def asian_geometric_payoff(paths, K, option_type):
    """Return the payoff on the geometric average of columns ``1:``.

    Its expectation has a closed form (``asian_geometric.price_geometric_asian``), which makes
    it the control variate for the arithmetic Asian.
    """
    if paths.shape[1] < 2:
        raise ValueError("paths must include at least one monitoring date beyond S0")
    geo_avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
    return european_payoff(geo_avg, K, option_type)


def _check_barrier_type(barrier_type):
    bt = barrier_type.lower()
    if bt not in BARRIER_TYPES:
        raise ValueError(
            f"barrier_type must be one of {sorted(BARRIER_TYPES)}, got '{barrier_type}'"
        )
    return bt


def validate_barrier(barrier_type, barrier, S0):
    """Check the barrier type and that the barrier is on the right side of spot.

    An up barrier must be above ``S0`` and a down barrier below it. Otherwise the contract is
    knocked in or out at inception, which is almost always an input mistake. Returns the
    lower-cased barrier type.
    """
    bt = _check_barrier_type(barrier_type)
    if bt.startswith("up") and not barrier > S0:
        raise ValueError(f"up barrier must be above S0 (barrier={barrier}, S0={S0})")
    if bt.startswith("down") and not barrier < S0:
        raise ValueError(f"down barrier must be below S0 (barrier={barrier}, S0={S0})")
    return bt


def bgk_adjusted_barrier(barrier, sigma, dt, barrier_type):
    """Return the barrier to simulate so that discrete monitoring mimics continuous.

    Broadie, Glasserman and Kou (1997) show that a barrier ``H`` checked every ``dt`` prices
    like a continuously monitored barrier moved away from spot by
    ``exp(0.5826 * sigma * sqrt(dt))``. To approximate the continuous contract, the simulated
    barrier is moved towards spot by the same factor:

    - up barrier:   ``H * exp(-0.5826 * sigma * sqrt(dt))``
    - down barrier: ``H * exp(+0.5826 * sigma * sqrt(dt))``

    The error after correction is ``o(sqrt(dt))`` instead of ``O(sqrt(dt))``.
    """
    bt = _check_barrier_type(barrier_type)
    shift = BGK_BETA * sigma * np.sqrt(dt)
    if bt.startswith("up"):
        return barrier * np.exp(-shift)
    return barrier * np.exp(shift)


def barrier_level_for_simulation(
    S0, barrier, T, sigma, *, barrier_type, steps, continuity_correction
):
    """Validate a barrier contract and return the level to check on the simulated paths.

    That level is ``barrier`` itself, or with ``continuity_correction`` the BGK-shifted
    barrier for the monitoring interval ``T / steps``.

    Raises
    ------
    ValueError
        If the barrier is on the wrong side of ``S0``, ``sigma`` is missing when the correction
        is requested, or the shifted barrier crosses ``S0`` (too few monitoring dates for the
        correction to make sense).
    """
    bt = validate_barrier(barrier_type, barrier, S0)
    if not continuity_correction:
        return barrier
    if sigma is None:
        raise ValueError("sigma is required when continuity_correction=True")
    level = bgk_adjusted_barrier(barrier, sigma, T / steps, bt)
    if (bt.startswith("up") and level <= S0) or (bt.startswith("down") and level >= S0):
        raise ValueError(
            f"continuity-corrected barrier {level:.4f} crosses S0={S0}; "
            "increase steps or disable the correction"
        )
    return level


def barrier_payoff(paths, K, barrier, barrier_type, option_type):
    """Return knock-in or knock-out payoffs from simulated paths.

    The barrier is checked only on the simulated dates (columns ``1:``). A path that crosses
    the barrier and comes back between two dates is missed, so a discretely monitored
    knock-out is worth more than a continuously monitored one, and a knock-in less. Use
    ``bgk_adjusted_barrier`` to approximate continuous monitoring.

    Parameters
    ----------
    paths : numpy.ndarray
        Path matrix of shape ``(n_paths, steps + 1)``, with ``S0`` in column 0.
    barrier : float
        Barrier level. Up barriers must be above ``S0`` and down barriers below.
    barrier_type : str
        ``"up_out"``, ``"up_in"``, ``"down_out"`` or ``"down_in"``.
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("paths must include at least one monitoring date beyond S0")

    bt = validate_barrier(barrier_type, barrier, paths[0, 0])

    monitoring = paths[:, 1:]
    if bt.startswith("up"):
        crossed = np.any(monitoring >= barrier, axis=1)
    else:
        crossed = np.any(monitoring <= barrier, axis=1)

    vanilla = european_payoff(paths[:, -1], K, option_type)
    if bt.endswith("out"):
        return np.where(crossed, 0.0, vanilla)
    return np.where(crossed, vanilla, 0.0)
