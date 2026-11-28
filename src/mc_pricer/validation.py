"""Input checks shared by the pricing modules."""

import numpy as np


def validate_positive(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_positive_int(value, name):
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def normalise_option_type(option_type):
    """Return ``"call"`` or ``"put"`` from a case-insensitive option type."""
    t = option_type.lower()
    if t not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return t
