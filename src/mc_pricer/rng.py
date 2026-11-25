"""Random number sources with a common ``normal(size)`` interface.

- ``RNG``: pseudo-random wrapper around ``numpy.random.default_rng``.
- ``QMCGenerator``: scrambled Sobol points mapped to normals, optionally
  through a Brownian bridge.

Either can be passed as ``rng`` to ``simulate_gbm_paths``.
"""

from collections import deque

import numpy as np


class RNG:
    """Lightweight wrapper around ``numpy.random.default_rng``.

    Parameters
    ----------
    seed : int | None, optional
        Seed for reproducible random streams. If ``None``, NumPy entropy is
        used to initialise the generator.
    """

    def __init__(self, seed=None):
        self.seed(seed)

    def seed(self, seed=None):
        """Reset the underlying generator with a new seed.

        Parameters
        ----------
        seed : int | None, optional
            Seed value passed to ``numpy.random.default_rng``.
        """
        self._generator = np.random.default_rng(seed)

    def normal(self, size, mean=0.0, std=1.0):
        """Draw normal random variates.

        Parameters
        ----------
        size : int | tuple[int, ...]
            Output shape of the draw.
        mean : float, optional
            Mean of the normal distribution (default ``0.0``).
        std : float, optional
            Standard deviation of the normal distribution (default ``1.0``).

        Returns
        -------
        numpy.ndarray
            Array of normal random variates with requested shape.
        """
        return self._generator.normal(loc=mean, scale=std, size=size)

    def uniform(self, size=None):
        """Draw uniform random variates on ``[0, 1)``.

        Parameters
        ----------
        size : int | tuple[int, ...] | None, optional
            Output shape. If ``None``, returns a scalar float.

        Returns
        -------
        float | numpy.ndarray
            Uniform draw(s) in the half-open interval ``[0, 1)``.
        """
        return self._generator.random(size=size)


def brownian_bridge_increments(z):
    """Map standard normals to Brownian increments using a Brownian bridge.

    Column 0 of ``z`` sets the terminal value ``W(1)``, column 1 the midpoint,
    columns 2-3 the quarter points, and so on (breadth-first bisection). The
    output has the same distribution as i.i.d. N(0, 1) increments, so it can
    be passed straight to ``simulate_gbm_paths`` as ``shocks``.

    With pseudo-random input this changes nothing statistically. With Sobol
    input it puts the best-distributed coordinates on the directions that
    explain most of the path variance, which is what makes QMC effective for
    path-dependent payoffs with many time steps.

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
    """Scrambled Sobol generator with the same ``normal(size)`` interface as ``RNG``.

    Parameters
    ----------
    dim : int
        Dimension of each point. Must equal the number of time steps when used
        with ``simulate_gbm_paths``.
    seed : int | None, optional
        Scramble seed. Different seeds give independent randomisations of the
        same sequence.
    brownian_bridge : bool, optional
        If True, map each Sobol point to path increments with
        ``brownian_bridge_increments`` instead of using coordinates as
        increments in time order. Default False.

    Notes
    -----
    - Use powers of two for ``n_paths``; scipy warns otherwise because the
      balance properties of the Sobol sequence are lost.
    - The generator is stateful: successive ``normal`` calls continue the
      sequence.
    - The i.i.d. standard error reported by the pricers is not valid for QMC
      output. To get an error estimate, price with several independent
      scrambles (different ``seed``) and use the spread of the estimates.
    """

    def __init__(self, dim, seed=None, brownian_bridge=False):
        from scipy.stats.qmc import Sobol

        self._engine = Sobol(d=dim, scramble=True, seed=seed)
        self._dim = dim
        self._brownian_bridge = brownian_bridge

    def normal(self, size):
        """Draw quasi-random standard normals by inverse-CDF transform.

        Parameters
        ----------
        size : int | tuple[int, int]
            Output shape ``(n, dim)``.

        Raises
        ------
        ValueError
            If the requested number of columns does not match ``dim``.
        """
        from scipy.stats import norm as scipy_norm

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
        uniform = self._engine.random(n)
        # Clamp away from 0/1 to avoid +/-inf from the inverse CDF.
        uniform = np.clip(uniform, 1e-10, 1 - 1e-10)
        z = scipy_norm.ppf(uniform)
        if self._brownian_bridge:
            z = brownian_bridge_increments(z)
        return z
