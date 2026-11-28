"""Market parameter sets shared by several test files."""

import pytest


@pytest.fixture
def european_market():
    return {"S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.02, "sigma": 0.25}


@pytest.fixture
def asian_market():
    return {"S0": 100.0, "K": 95.0, "T": 1.0, "r": 0.015, "sigma": 0.25}
