"""Discounting, standard errors and the result dictionary that every pricer returns.

The dictionary holds ``price``, ``stderr``, ``ci_low``, ``ci_high``, ``n_paths`` and ``extra``.
The confidence interval is the two-sided 95% normal interval.
"""

import numpy as np
from scipy.stats import norm

Z_95 = norm.ppf(0.975)


def discount_payoffs(payoffs, T, r):
    """Discount payoffs paid at ``T`` by ``exp(-r T)``."""
    return np.exp(-r * T) * np.asarray(payoffs, dtype=float)


def standard_error(samples):
    """Return the standard error of the mean of a 1D sample, or 0 for a single value."""
    n = samples.size
    return samples.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0


def _result(samples, n_paths, extra):
    price = samples.mean()
    stderr = standard_error(samples)
    half_width = Z_95 * stderr
    return {
        "price": float(price),
        "stderr": float(stderr),
        "ci_low": float(price - half_width),
        "ci_high": float(price + half_width),
        "n_paths": n_paths,
        "extra": extra,
    }


def build_result_iid(discounted_payoffs, extra=None):
    """Build the result dictionary from independent discounted payoffs."""
    samples = np.asarray(discounted_payoffs, dtype=float)
    if samples.shape[0] == 0:
        raise ValueError("No payoffs provided")
    return _result(samples, samples.shape[0], extra or {})


def build_result_antithetic(discounted_payoffs, extra=None):
    """Build the result dictionary from antithetic payoffs ordered as ``[X(Z), X(-Z)]``.

    The two paths in a pair are dependent, so the standard error is taken over the pair means
    ``(X(Z) + X(-Z)) / 2``. ``n_paths`` counts both paths of every pair, and
    ``extra["effective_samples"]`` is the number of pairs.
    """
    payoffs = np.asarray(discounted_payoffs, dtype=float)
    n_paths = payoffs.shape[0]
    if n_paths == 0:
        raise ValueError("No payoffs provided")
    if n_paths % 2 != 0:
        raise ValueError("Antithetic estimator requires an even number of paths")

    n_pairs = n_paths // 2
    pair_means = 0.5 * (payoffs[:n_pairs] + payoffs[n_pairs:])
    return _result(pair_means, n_paths, {**(extra or {}), "effective_samples": n_pairs})
