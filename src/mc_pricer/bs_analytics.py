"""Closed-form Black-Scholes prices and Greeks, and a continuously monitored down-and-out call.

The arguments are spot ``S0``, strike ``K``, maturity ``T`` in years, the continuously
compounded rate ``r`` and the volatility ``sigma``. Rate and volatility are constant, there
are no dividends, and exercise is European. Non-positive ``S0``, ``K``, ``T`` or ``sigma``
raise ``ValueError``.
"""

import math

from scipy.stats import norm

from .validation import normalise_option_type, validate_positive


def _d1_d2(S0, K, T, r, sigma):
    validate_positive(S0, "S0")
    validate_positive(K, "K")
    validate_positive(T, "T")
    validate_positive(sigma, "sigma")

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    return d1, d1 - sigma * sqrt_T


def bs_price(S0, K, T, r, sigma, *, option_type):
    """Return the Black-Scholes price of a European call or put."""
    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    disc = math.exp(-r * T)
    if normalise_option_type(option_type) == "call":
        return S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
    return K * disc * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def bs_delta(S0, K, T, r, sigma, *, option_type):
    """Return the Black-Scholes delta, the derivative of the price in ``S0``."""
    d1, _ = _d1_d2(S0, K, T, r, sigma)
    if normalise_option_type(option_type) == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def bs_vega(S0, K, T, r, sigma):
    """Return the Black-Scholes vega per unit of ``sigma`` (0.01 is one vol point)."""
    d1, _ = _d1_d2(S0, K, T, r, sigma)
    return S0 * math.sqrt(T) * norm.pdf(d1)


def bs_gamma(S0, K, T, r, sigma):
    """Return the Black-Scholes gamma, which is the same for calls and puts."""
    d1, _ = _d1_d2(S0, K, T, r, sigma)
    return norm.pdf(d1) / (S0 * sigma * math.sqrt(T))


def bs_theta(S0, K, T, r, sigma, *, option_type):
    """Return the Black-Scholes theta per year, ``dV/dt = -dV/dT``."""
    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    disc = math.exp(-r * T)
    option_type = normalise_option_type(option_type)
    decay = -(S0 * sigma * norm.pdf(d1)) / (2.0 * math.sqrt(T))
    if option_type == "call":
        return decay - r * K * disc * norm.cdf(d2)
    return decay + r * K * disc * norm.cdf(-d2)


def bs_rho(S0, K, T, r, sigma, *, option_type):
    """Return the Black-Scholes rho, per unit change in ``r``."""
    _, d2 = _d1_d2(S0, K, T, r, sigma)
    disc = math.exp(-r * T)
    if normalise_option_type(option_type) == "call":
        return K * T * disc * norm.cdf(d2)
    return -K * T * disc * norm.cdf(-d2)


def bs_down_and_out_call(S0, K, barrier, T, r, sigma):
    """Return the price of a continuously monitored down-and-out call.

    Standard closed form (Merton 1973; Reiner and Rubinstein 1991; Hull, *Options, Futures,
    and Other Derivatives*, ch. 26) with no rebate and no dividends. Requires
    ``barrier < S0``.

    With ``lam = (r + sigma^2 / 2) / sigma^2`` and ``H = barrier``:

    - ``H <= K``: ``c_do = c_bs - c_di`` where
      ``c_di = S0 (H/S0)^(2 lam) N(y) - K e^(-rT) (H/S0)^(2 lam - 2) N(y - sigma sqrt(T))``
      and ``y = ln(H^2 / (S0 K)) / (sigma sqrt(T)) + lam sigma sqrt(T)``.
    - ``H > K``: ``c_do = S0 N(x1) - K e^(-rT) N(x1 - sigma sqrt(T))
      - S0 (H/S0)^(2 lam) N(y1) + K e^(-rT) (H/S0)^(2 lam - 2) N(y1 - sigma sqrt(T))``
      with ``x1 = ln(S0/H) / (sigma sqrt(T)) + lam sigma sqrt(T)`` and
      ``y1 = ln(H/S0) / (sigma sqrt(T)) + lam sigma sqrt(T)``.

    Discretely monitored Monte Carlo prices sit above this value. See
    ``payoffs.bgk_adjusted_barrier`` for the continuity correction.
    """
    validate_positive(barrier, "barrier")
    if barrier >= S0:
        raise ValueError("down-and-out barrier must be below S0")
    validate_positive(K, "K")
    validate_positive(T, "T")
    validate_positive(sigma, "sigma")

    H = barrier
    sig_sqrt_t = sigma * math.sqrt(T)
    lam = (r + 0.5 * sigma * sigma) / (sigma * sigma)
    disc = math.exp(-r * T)
    ratio = H / S0

    if H <= K:
        y = math.log(H * H / (S0 * K)) / sig_sqrt_t + lam * sig_sqrt_t
        c_di = S0 * ratio ** (2 * lam) * norm.cdf(y) - K * disc * ratio ** (2 * lam - 2) * norm.cdf(
            y - sig_sqrt_t
        )
        return bs_price(S0, K, T, r, sigma, option_type="call") - c_di

    x1 = math.log(S0 / H) / sig_sqrt_t + lam * sig_sqrt_t
    y1 = math.log(H / S0) / sig_sqrt_t + lam * sig_sqrt_t
    return (
        S0 * norm.cdf(x1)
        - K * disc * norm.cdf(x1 - sig_sqrt_t)
        - S0 * ratio ** (2 * lam) * norm.cdf(y1)
        + K * disc * ratio ** (2 * lam - 2) * norm.cdf(y1 - sig_sqrt_t)
    )
