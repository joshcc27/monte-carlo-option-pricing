"""Shared fixtures: standard market parameter sets and a seeded RNG factory."""

import pytest

from mc_pricer.rng import RNG


@pytest.fixture
def european_market():
    """European-option parameters reused across tests."""
    return {"S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.02, "sigma": 0.25}


@pytest.fixture
def asian_market():
    """Asian-option parameters reused across tests."""
    return {"S0": 100.0, "K": 95.0, "T": 1.0, "r": 0.015, "sigma": 0.25}


@pytest.fixture
def make_rng():
    """Return a factory for seeded RNG instances."""

    def _make(seed):
        return RNG(seed=seed)

    return _make
