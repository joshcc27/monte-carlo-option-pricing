"""Cox-Ross-Rubinstein binomial tree for European and American options.

- Up factor ``u = exp(sigma * sqrt(dt))``, down factor ``d = 1 / u``.
- Risk-neutral probability ``p = (exp(r * dt) - d) / (u - d)``.
- Backward induction from terminal payoffs. For American exercise, each node
  takes the larger of the continuation value and immediate exercise.
"""

import math

import numpy as np

from .validation import normalise_option_type, validate_positive, validate_positive_int


def price_binomial(S0, K, T, r, sigma, *, steps, option_type="call", exercise="european"):
    """Return the CRR binomial price of a European or American call or put.

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
    exercise : str
        ``"european"`` (the default) or ``"american"`` (case-insensitive).

    Raises
    ------
    ValueError
        If numeric inputs are not positive, ``option_type`` or ``exercise`` is
        invalid, or the tree admits arbitrage (``d < exp(r * dt) < u`` fails, so ``p`` is
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
    exercise = str(exercise).lower()
    if exercise not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}")

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

    sign = 1.0 if option_type == "call" else -1.0
    values = np.maximum(sign * (ST - K), 0.0)

    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])
        if exercise == "american":
            # Prices at level i, lowest first, matching the order of ``values``.
            S = S0 * d ** np.arange(i, -1, -1) * u ** np.arange(i + 1)
            values = np.maximum(values, sign * (S - K))

    return float(values[0])
