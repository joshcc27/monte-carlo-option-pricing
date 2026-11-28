"""Closed-form price of a discretely monitored geometric-average Asian option.

Under GBM the log of the geometric average of prices observed at ``t_1, ..., t_n`` is normal,
so the option has a Black-Scholes-type price. With ``G = (prod_i S(t_i))^(1/n)``:

    log G ~ N(log S0 + mu_g, sigma_g^2)
    mu_g      = (r - sigma^2 / 2) * mean(t_i)
    sigma_g^2 = sigma^2 / n^2 * sum_i sum_j min(t_i, t_j)

For ``n = 1`` and ``t_1 = T`` this is the Black-Scholes price of a European option. The
arithmetic Asian pricer uses this price as its control variate mean.
"""

import numpy as np
from scipy.stats import norm

from .validation import normalise_option_type, validate_positive


def monitoring_times(T, steps):
    """Return the ``steps`` equally spaced monitoring times ``T/steps, ..., T``.

    ``t = 0`` is left out, matching the average in ``payoffs.asian_arithmetic_payoff``
    (columns ``1:`` of the path matrix).
    """
    times = np.linspace(0.0, T, steps + 1)
    return times[1:]


def _sum_pairwise_min(times):
    """Return ``sum_i sum_j min(t_i, t_j)`` in O(n log n).

    After sorting, ``t_(k)`` is the minimum in ``2 * (n - k) + 1`` ordered pairs (k = 1..n),
    so the double sum collapses to a weighted single sum.
    """
    t = np.sort(times)
    n = t.size
    k = np.arange(1, n + 1)
    return float(np.sum(t * (2 * (n - k) + 1)))


def geometric_parameters(times, r, sigma):
    """Return ``(mu_g, sigma_g)`` with ``log(G) ~ N(log(S0) + mu_g, sigma_g^2)``.

    ``times`` are the monitoring times, a non-empty 1D array of positive values.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty 1D array")
    if np.any(times <= 0):
        raise ValueError("monitoring times must be positive")

    n = times.size
    mu_g = (r - 0.5 * sigma * sigma) * times.mean()
    var_g = sigma * sigma * _sum_pairwise_min(times) / (n * n)
    return mu_g, float(np.sqrt(var_g))


def price_geometric_asian(S0, K, T, r, sigma, *, steps, option_type):
    """Return the closed-form price of a geometric-average Asian call or put.

    The average is taken over the ``steps`` equally spaced dates ``T/steps, ..., T``, without
    ``S0``, and the payoff is paid at ``T``.
    """
    validate_positive(S0, "S0")
    validate_positive(K, "K")
    validate_positive(sigma, "sigma")
    validate_positive(T, "T")
    option_type = normalise_option_type(option_type)

    times = monitoring_times(T, steps)
    mu_g, sigma_g = geometric_parameters(times, r, sigma)

    mean_ln = np.log(S0) + mu_g
    forward_g = np.exp(mean_ln + 0.5 * sigma_g * sigma_g)  # E[G]
    d1 = (mean_ln - np.log(K) + sigma_g * sigma_g) / sigma_g
    d2 = d1 - sigma_g
    discount = np.exp(-r * T)

    if option_type == "call":
        return float(discount * (forward_g * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(discount * (K * norm.cdf(-d2) - forward_g * norm.cdf(-d1)))
