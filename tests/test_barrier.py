"""Tests for barrier payoffs, the barrier pricers and the continuity correction."""

import numpy as np
import pytest

from mc_pricer.bs_analytics import bs_down_and_out_call, bs_price
from mc_pricer.gbm import simulate_gbm_paths
from mc_pricer.mc import mc_price_barrier, mc_price_european
from mc_pricer.mc_results import build_result_antithetic
from mc_pricer.payoffs import barrier_payoff, bgk_adjusted_barrier, european_payoff
from mc_pricer.rng import RNG
from mc_pricer.variance_reduction import (
    mc_price_barrier_antithetic,
    simulate_gbm_paths_antithetic,
)

# ---------------------------------------------------------------------------
# barrier_payoff unit tests
# ---------------------------------------------------------------------------


def test_down_out_knocked_path_gets_zero_payoff():
    # A path that dips below the barrier should receive zero payoff.
    # Shape: (2 paths, 3 times including S0)
    paths = np.array(
        [
            [100.0, 80.0, 110.0],  # crosses barrier=90 → knocked out
            [100.0, 95.0, 115.0],  # stays above barrier → survives
        ]
    )
    payoffs = barrier_payoff(
        paths, K=100.0, barrier=90.0, barrier_type="down_out", option_type="call"
    )
    assert payoffs[0] == 0.0  # knocked out
    assert payoffs[1] == pytest.approx(15.0)  # vanilla call payoff


def test_down_in_only_pays_when_crossed():
    paths = np.array(
        [
            [100.0, 80.0, 115.0],  # crosses barrier=90 → activated
            [100.0, 95.0, 115.0],  # never crosses → zero
        ]
    )
    payoffs = barrier_payoff(
        paths, K=100.0, barrier=90.0, barrier_type="down_in", option_type="call"
    )
    assert payoffs[0] == pytest.approx(15.0)
    assert payoffs[1] == 0.0


def test_up_out_knocked_path_gets_zero():
    paths = np.array(
        [
            [100.0, 130.0, 105.0],  # crosses barrier=120 → knocked out
            [100.0, 110.0, 108.0],  # stays below → survives
        ]
    )
    payoffs = barrier_payoff(
        paths, K=100.0, barrier=120.0, barrier_type="up_out", option_type="call"
    )
    assert payoffs[0] == 0.0
    assert payoffs[1] == pytest.approx(8.0)


def test_up_in_only_pays_when_crossed():
    paths = np.array(
        [
            [100.0, 130.0, 108.0],  # crosses barrier=120 → activated
            [100.0, 110.0, 108.0],  # never crosses → zero
        ]
    )
    payoffs = barrier_payoff(
        paths, K=100.0, barrier=120.0, barrier_type="up_in", option_type="call"
    )
    assert payoffs[0] == pytest.approx(8.0)
    assert payoffs[1] == 0.0


def test_knock_in_plus_knock_out_equals_vanilla():
    # In-out parity: knock-in payoff + knock-out payoff = vanilla payoff (same paths).
    rng = RNG(seed=42)
    paths = simulate_gbm_paths(100.0, 0.02, 0.2, 1.0, steps=64, n_paths=10_000, rng=rng)
    K, barrier = 100.0, 85.0
    out_payoffs = barrier_payoff(paths, K, barrier, "down_out", "call")
    in_payoffs = barrier_payoff(paths, K, barrier, "down_in", "call")
    vanilla = european_payoff(paths[:, -1], K, "call")
    np.testing.assert_allclose(out_payoffs + in_payoffs, vanilla)


def test_barrier_payoff_rejects_invalid_barrier_type():
    paths = np.ones((5, 3))
    with pytest.raises(ValueError, match="barrier_type"):
        barrier_payoff(
            paths, K=100.0, barrier=90.0, barrier_type="sideways_out", option_type="call"
        )


@pytest.mark.parametrize(
    ("barrier_type", "barrier"),
    [("up_out", 95.0), ("up_in", 100.0), ("down_out", 105.0), ("down_in", 100.0)],
)
def test_barrier_on_wrong_side_of_spot_is_rejected(barrier_type, barrier):
    paths = np.full((3, 4), 100.0)
    with pytest.raises(ValueError, match="barrier must be"):
        barrier_payoff(
            paths, K=100.0, barrier=barrier, barrier_type=barrier_type, option_type="call"
        )


def test_barrier_payoff_rejects_single_column_path():
    with pytest.raises(ValueError, match="monitoring date"):
        barrier_payoff(
            np.ones((5, 1)), K=100.0, barrier=90.0, barrier_type="down_out", option_type="call"
        )


# ---------------------------------------------------------------------------
# mc_price_barrier integration tests
# ---------------------------------------------------------------------------


def test_mc_price_barrier_result_schema(make_rng):
    rng = make_rng(1)
    paths = simulate_gbm_paths(100.0, 0.02, 0.2, 1.0, steps=32, n_paths=10_000, rng=rng)
    result = mc_price_barrier(
        paths, K=100.0, r=0.02, T=1.0, barrier=85.0, barrier_type="down_out", option_type="call"
    )
    assert set(result.keys()) == {"price", "stderr", "ci_low", "ci_high", "n_paths", "extra"}
    assert result["price"] >= 0
    assert result["ci_low"] < result["price"] < result["ci_high"]


def test_mc_price_barrier_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2D"):
        mc_price_barrier(
            np.array([100.0, 101.0]),
            K=100.0,
            r=0.02,
            T=1.0,
            barrier=85.0,
            barrier_type="down_out",
            option_type="call",
        )


def test_down_out_call_cheaper_than_vanilla(make_rng):
    # A down-and-out call is strictly less valuable than a vanilla call
    # (it can be knocked out, so it can only be worth less or equal).
    rng = make_rng(7)
    paths = simulate_gbm_paths(100.0, 0.02, 0.2, 1.0, steps=64, n_paths=50_000, rng=rng)
    vanilla = mc_price_european(paths[:, -1], K=100.0, r=0.02, T=1.0, option_type="call")
    barrier_result = mc_price_barrier(
        paths, K=100.0, r=0.02, T=1.0, barrier=80.0, barrier_type="down_out", option_type="call"
    )
    assert barrier_result["price"] < vanilla["price"]


def test_down_out_with_zero_barrier_matches_vanilla(make_rng):
    # A barrier of 0 can never be breached under GBM, so the knock-out price
    # should match the plain European price.
    rng = make_rng(99)
    paths = simulate_gbm_paths(100.0, 0.02, 0.2, 1.0, steps=16, n_paths=20_000, rng=rng)
    vanilla = mc_price_european(paths[:, -1], K=100.0, r=0.02, T=1.0, option_type="call")
    barrier_result = mc_price_barrier(
        paths, K=100.0, r=0.02, T=1.0, barrier=0.0, barrier_type="down_out", option_type="call"
    )
    assert abs(barrier_result["price"] - vanilla["price"]) < 1e-10


# ---------------------------------------------------------------------------
# Continuous closed form and the Broadie-Glasserman-Kou correction
# ---------------------------------------------------------------------------

S0, K, H, R, SIGMA, T = 100.0, 100.0, 90.0, 0.05, 0.2, 1.0


def test_down_and_out_closed_form_reference_value():
    assert bs_down_and_out_call(S0, K, T, R, SIGMA, H) == pytest.approx(8.6655, abs=1e-4)


def test_down_and_out_closed_form_limits():
    # Barrier far below spot: vanilla call. Continuous at H = K.
    vanilla = bs_price(S0, K, T, R, SIGMA, "call")
    assert bs_down_and_out_call(S0, K, T, R, SIGMA, 1e-6) == pytest.approx(vanilla, abs=1e-10)
    below = bs_down_and_out_call(S0, 95.0, T, R, SIGMA, 95.0 - 1e-9)
    above = bs_down_and_out_call(S0, 95.0, T, R, SIGMA, 95.0 + 1e-9)
    assert below == pytest.approx(above, abs=1e-6)


def test_bgk_shift_moves_barrier_towards_spot():
    dt = 1.0 / 50
    shift = np.exp(0.5826 * SIGMA * np.sqrt(dt))
    assert bgk_adjusted_barrier(90.0, SIGMA, dt, "down_out") == pytest.approx(90.0 * shift)
    assert bgk_adjusted_barrier(120.0, SIGMA, dt, "up_in") == pytest.approx(120.0 / shift)


def test_discrete_down_and_out_converges_from_above_and_correction_closes_gap():
    closed = bs_down_and_out_call(S0, K, T, R, SIGMA, H)
    uncorrected, corrected, stderrs = [], [], []
    for steps in (4, 16, 64):
        paths = simulate_gbm_paths(S0, R, SIGMA, T, steps, 100_000, rng=RNG(seed=steps))
        plain = mc_price_barrier(paths, K, R, T, H, "down_out", "call")
        bgk = mc_price_barrier(
            paths, K, R, T, H, "down_out", "call", sigma=SIGMA, continuity_correction=True
        )
        uncorrected.append(plain["price"])
        corrected.append(bgk["price"])
        stderrs.append(plain["stderr"])

    # Discrete monitoring misses crossings between dates, so the knock-out is
    # worth more than the continuous contract, and the gap shrinks with steps.
    for price, se in zip(uncorrected, stderrs, strict=True):
        assert price > closed + 3 * se
    assert uncorrected[0] > uncorrected[1] > uncorrected[2]

    # With the shifted barrier the 64-date price is within MC noise plus a
    # small residual bias of the continuous value, and far closer than without.
    assert abs(corrected[-1] - closed) < 3 * stderrs[-1] + 0.05
    assert abs(corrected[-1] - closed) < 0.25 * (uncorrected[-1] - closed)


def test_continuity_correction_requires_sigma():
    paths = simulate_gbm_paths(S0, R, SIGMA, T, 8, 100, rng=RNG(seed=0))
    with pytest.raises(ValueError, match="sigma is required"):
        mc_price_barrier(paths, K, R, T, H, "down_out", "call", continuity_correction=True)


def test_continuity_correction_rejects_shift_across_spot():
    paths = simulate_gbm_paths(S0, R, SIGMA, T, 2, 100, rng=RNG(seed=0))
    with pytest.raises(ValueError, match="crosses S0"):
        mc_price_barrier(
            paths, K, R, T, 99.0, "down_out", "call", sigma=SIGMA, continuity_correction=True
        )


# ---------------------------------------------------------------------------
# Antithetic barrier pricer
# ---------------------------------------------------------------------------


def test_barrier_antithetic_matches_manual_pair_mean_estimate():
    steps, n_paths = 32, 20_000
    result = mc_price_barrier_antithetic(
        S0, K, R, T, SIGMA, steps, n_paths, RNG(seed=4), H, "down_out", "call"
    )
    paths = simulate_gbm_paths_antithetic(S0, R, SIGMA, T, steps, n_paths, RNG(seed=4))
    discounted = np.exp(-R * T) * barrier_payoff(paths, K, H, "down_out", "call")
    manual = build_result_antithetic(discounted)
    assert result["price"] == pytest.approx(manual["price"], abs=1e-12)
    assert result["stderr"] == pytest.approx(manual["stderr"], abs=1e-12)
    assert result["extra"]["variance_reduction"] == "antithetic"


def test_barrier_antithetic_agrees_with_plain_mc():
    steps, n_paths = 32, 100_000
    paths = simulate_gbm_paths(S0, R, SIGMA, T, steps, n_paths, rng=RNG(seed=10))
    plain = mc_price_barrier(paths, K, R, T, H, "down_out", "call")
    anti = mc_price_barrier_antithetic(
        S0, K, R, T, SIGMA, steps, n_paths, RNG(seed=11), H, "down_out", "call"
    )
    assert abs(anti["price"] - plain["price"]) < 3 * np.hypot(plain["stderr"], anti["stderr"])
    assert anti["stderr"] < plain["stderr"]
