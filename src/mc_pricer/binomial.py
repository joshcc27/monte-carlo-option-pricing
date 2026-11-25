"""Cox-Ross-Rubinstein binomial tree for European options.

- Up factor ``u = exp(sigma * sqrt(dt))``, down factor ``d = 1 / u``.
- Risk-neutral probability ``p = (exp(r * dt) - d) / (u - d)``.
- Backward induction from terminal payoffs.

Only European exercise is implemented; there is no early-exercise check.
"""

import math

import numpy as np

from .validation import normalise_option_type, validate_positive, validate_positive_int


def price_binomial(S0, K, T, r, sigma, *, steps, option_type="call"):
    """Return the CRR binomial price of a European call or put.

    Parameters
    ----------
    S0, K : float
        Spot and strike.
    T : float
        Maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (annualised, decimal).
    steps : int
        Number of time steps.
    option_type : str
        ``"call"`` or ``"put"`` (case-insensitive).

    Raises
    ------
    ValueError
        If numeric inputs are not positive, ``option_type`` is invalid, or
        the tree admits arbitrage (``d < exp(r * dt) < u`` fails, so ``p`` is
        outside ``(0, 1)``). The last case happens when ``|r| * sqrt(dt)`` is
        large relative to ``sigma``; more steps fixes it.
    TypeError
        If ``steps`` is not an integer.
    """
    validate_positive(S0, "S0")
    validate_positive(K, "K")
    validate_positive(T, "T")
    validate_positive(sigma, "sigma")
    validate_positive_int(steps, "steps")
    option_type = normalise_option_type(option_type)

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp(r * dt)
    if not d < growth < u:
        raise ValueError(
            f"CRR tree is not arbitrage-free with steps={steps}: need "
            f"d < exp(r*dt) < u, got d={d:.6f}, exp(r*dt)={growth:.6f}, u={u:.6f}. "
            "Increase steps or sigma."
        )
    p = (growth - d) / (u - d)
    disc = 1.0 / growth

    # Terminal prices, ordered from lowest (all down moves) to highest.
    j = np.arange(steps, -1, -1)
    ST = S0 * (d**j) * (u ** (steps - j))

    if option_type == "call":
        values = np.maximum(ST - K, 0.0)
    else:
        values = np.maximum(K - ST, 0.0)

    for _ in range(steps):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])

    return float(values[0])
