"""Tests for the CRR binomial tree pricer."""

import math

import pytest

from mc_pricer.binomial import price_binomial

S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2

# Black-Scholes prices for these inputs.
BS_CALL = 10.450583572185565
BS_PUT = 5.573526022256971


def test_call_converges_to_bs_at_high_steps():
    price = price_binomial(S0, K, T, r, sigma, steps=2000, option_type="call")
    assert abs(price - BS_CALL) < 0.02


def test_put_converges_to_bs_at_high_steps():
    price = price_binomial(S0, K, T, r, sigma, steps=2000, option_type="put")
    assert abs(price - BS_PUT) < 0.02


def test_put_call_parity():
    # C - P = S0 - K exp(-rT)
    call = price_binomial(S0, K, T, r, sigma, steps=1000, option_type="call")
    put = price_binomial(S0, K, T, r, sigma, steps=1000, option_type="put")
    parity_rhs = S0 - K * math.exp(-r * T)
    assert abs((call - put) - parity_rhs) < 1e-4


def test_error_decreases_with_more_steps():
    step_counts = [10, 50, 200, 1000]
    errors = [abs(price_binomial(S0, K, T, r, sigma, steps=n) - BS_CALL) for n in step_counts]
    for i in range(len(errors) - 1):
        assert errors[i] >= errors[i + 1], (
            f"Error did not decrease: steps={step_counts[i]} err={errors[i]:.6f}, "
            f"steps={step_counts[i + 1]} err={errors[i + 1]:.6f}"
        )


def test_call_and_put_are_positive(european_market):
    m = european_market
    call = price_binomial(**m, steps=200, option_type="call")
    put = price_binomial(**m, steps=200, option_type="put")
    assert call > 0
    assert put > 0


def test_rejects_non_positive_spot():
    with pytest.raises(ValueError):
        price_binomial(0.0, K, T, r, sigma, steps=100)


def test_rejects_non_positive_strike():
    with pytest.raises(ValueError):
        price_binomial(S0, -1.0, T, r, sigma, steps=100)


def test_rejects_non_positive_maturity():
    with pytest.raises(ValueError):
        price_binomial(S0, K, 0.0, r, sigma, steps=100)


def test_rejects_non_positive_sigma():
    with pytest.raises(ValueError):
        price_binomial(S0, K, T, r, 0.0, steps=100)


def test_rejects_float_steps():
    with pytest.raises(TypeError):
        price_binomial(S0, K, T, r, sigma, steps=100.0)


def test_rejects_invalid_option_type():
    with pytest.raises(ValueError):
        price_binomial(S0, K, T, r, sigma, steps=100, option_type="fwd")


def test_rejects_tree_with_risk_neutral_probability_outside_unit_interval():
    # sigma * sqrt(dt) = 0.125 and r * dt = 0.5, so exp(r * dt) > u and p > 1.
    with pytest.raises(ValueError, match="arbitrage-free"):
        price_binomial(S0=100.0, K=100.0, T=1.0, r=0.25, sigma=0.01, steps=2)


def test_same_inputs_are_fine_with_enough_steps():
    # The condition is roughly r * sqrt(dt) < sigma, so more steps restore it.
    price = price_binomial(S0=100.0, K=100.0, T=1.0, r=0.25, sigma=0.01, steps=2000)
    assert price == pytest.approx(100.0 - 100.0 * math.exp(-0.25), abs=1e-3)


# American put for the parameters above, from the same tree at 10,000 steps.
AMERICAN_PUT_10000 = 6.0903


def test_american_call_equals_european_call():
    # Without dividends, exercising a call early is never optimal.
    european = price_binomial(S0, K, T, r, sigma, steps=500, option_type="call")
    american = price_binomial(
        S0, K, T, r, sigma, steps=500, option_type="call", exercise="american"
    )
    assert american == pytest.approx(european, abs=1e-12)


@pytest.mark.parametrize("spot", [80.0, 100.0, 120.0])
def test_american_put_is_worth_at_least_european_and_intrinsic(spot):
    european = price_binomial(spot, K, T, r, sigma, steps=500, option_type="put")
    american = price_binomial(
        spot, K, T, r, sigma, steps=500, option_type="put", exercise="american"
    )
    assert american > european
    assert american >= max(K - spot, 0.0)


def test_american_put_converges():
    price = price_binomial(S0, K, T, r, sigma, steps=2000, option_type="put", exercise="american")
    assert abs(price - AMERICAN_PUT_10000) < 0.002


def test_rejects_invalid_exercise():
    with pytest.raises(ValueError, match="exercise"):
        price_binomial(S0, K, T, r, sigma, steps=100, exercise="bermudan")
