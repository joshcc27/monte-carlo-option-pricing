"""Path simulation for risk-neutral geometric Brownian motion, ``dS_t / S_t = r dt + sigma dW_t``.

Paths are built in log space, so the prices on the time grid are exact for any step size.
"""

import numpy as np

from .validation import validate_positive, validate_positive_int


def simulate_gbm_paths(S0, T, r, sigma, *, steps, n_paths, rng=None, shocks=None):
    """Simulate GBM prices on ``steps`` equally spaced dates up to ``T``.

    Parameters
    ----------
    steps : int
        Number of time steps.
    n_paths : int
        Number of paths.
    rng : RNG or QMCGenerator, optional
        Source of standard normals with a ``normal(size)`` method. Required unless ``shocks``
        is given.
    shocks : array-like, optional
        Standard normal shocks of shape ``(n_paths, steps)``. When given, ``rng`` is ignored.

    Returns
    -------
    numpy.ndarray
        Prices of shape ``(n_paths, steps + 1)``, with ``S0`` in column 0.
    """
    validate_positive(S0, "S0")
    validate_positive(sigma, "sigma")
    validate_positive(T, "T")
    validate_positive_int(steps, "steps")
    validate_positive_int(n_paths, "n_paths")

    if shocks is not None:
        shocks = np.asarray(shocks, dtype=float)
        if shocks.shape != (n_paths, steps):
            raise ValueError(f"shocks must have shape {(n_paths, steps)}")
    elif rng is None:
        raise ValueError("rng is required when shocks are not provided")
    else:
        shocks = rng.normal(size=(n_paths, steps))

    # Each step adds (r - sigma^2 / 2) dt + sigma sqrt(dt) Z to log S.
    dt = T / steps
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt)

    paths = np.empty((n_paths, steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(drift + diffusion * shocks, axis=1))
    return paths
