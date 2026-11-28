"""Sources of standard normals with a common ``normal(size)`` method.

``RNG`` draws pseudo-random normals and ``QMCGenerator`` maps scrambled Sobol points to normals,
optionally through a Brownian bridge. Either can be passed as ``rng`` to ``simulate_gbm_paths``.
"""

from collections import deque

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import Sobol


class RNG:
    """Pseudo-random standard normals from ``numpy.random.default_rng(seed)``."""

    def __init__(self, seed=None):
        self._generator = np.random.default_rng(seed)

    def normal(self, size):
        return self._generator.normal(size=size)


def brownian_bridge_increments(z):
    """Map standard normals to Brownian increments with a Brownian bridge.

    Column 0 of ``z`` sets the terminal value ``W(1)``, column 1 the midpoint, columns 2 and 3
    the quarter points, and so on (breadth-first bisection). The output has the same
    distribution as i.i.d. N(0, 1) increments, so it can be passed to ``simulate_gbm_paths``
    as ``shocks``.

    With pseudo-random input this changes nothing statistically. With Sobol input it puts the
    best-distributed coordinates on the directions that explain most of the path variance,
    which is what makes QMC effective for path-dependent payoffs with many time steps.

    Parameters
    ----------
    z : numpy.ndarray
        Standard normals of shape ``(n, steps)``.

    Returns
    -------
    numpy.ndarray
        Increments ``(W(t_k) - W(t_{k-1})) / sqrt(dt)`` of shape ``(n, steps)``.
    """
    z = np.asarray(z, dtype=float)
    n, m = z.shape
    dt = 1.0 / m
    w = np.zeros((n, m + 1))
    w[:, m] = z[:, 0]  # W(1) ~ N(0, 1)
    col = 1
    intervals = deque([(0, m)])
    while intervals:
        left, right = intervals.popleft()
        if right - left < 2:
            continue
        mid = (left + right) // 2
        weight = (mid - left) / (right - left)
        mean = (1.0 - weight) * w[:, left] + weight * w[:, right]
        std = np.sqrt((mid - left) * (right - mid) / (right - left) * dt)
        w[:, mid] = mean + std * z[:, col]
        col += 1
        intervals.append((left, mid))
        intervals.append((mid, right))
    return np.diff(w, axis=1) / np.sqrt(dt)


class QMCGenerator:
    """Scrambled Sobol points mapped to standard normals by the inverse normal CDF.

    Parameters
    ----------
    dim : int
        Dimension of each point. Must equal the number of time steps when used with
        ``simulate_gbm_paths``.
    seed : int, optional
        Scramble seed. Different seeds give independent randomisations of the same sequence.
    brownian_bridge : bool, optional
        Map each point to path increments with ``brownian_bridge_increments`` instead of
        using its coordinates as increments in time order.

    Notes
    -----
    Use powers of two for ``n_paths``. Otherwise scipy warns, because the balance properties
    of the Sobol sequence are lost. Successive ``normal`` calls continue the sequence.

    The i.i.d. standard error that the pricers report does not apply to QMC output. For an
    error estimate, price with several scrambles (different ``seed``) and use the spread of
    the estimates.
    """

    def __init__(self, dim, seed=None, brownian_bridge=False):
        self._engine = Sobol(d=dim, scramble=True, seed=seed)
        self._dim = dim
        self._brownian_bridge = brownian_bridge

    def normal(self, size):
        """Return quasi-random standard normals of shape ``size = (n, dim)``."""
        if isinstance(size, int):
            n, d = size, 1
        else:
            n = size[0]
            d = size[1] if len(size) > 1 else 1
        if d != self._dim:
            raise ValueError(
                f"QMCGenerator has dim={self._dim} but size requests {d} columns. "
                "Initialise QMCGenerator(dim=steps) to match the path step count."
            )
        # Clamp away from 0 and 1, where the inverse CDF is infinite.
        uniform = np.clip(self._engine.random(n), 1e-10, 1 - 1e-10)
        z = norm.ppf(uniform)
        if self._brownian_bridge:
            z = brownian_bridge_increments(z)
        return z
