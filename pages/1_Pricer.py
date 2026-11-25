import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from mc_pricer.bs_analytics import bs_down_and_out_call, bs_price
from mc_pricer.gbm import simulate_gbm_paths
from mc_pricer.mc import (
    barrier_level_for_simulation,
    mc_price_asian_arithmetic,
    mc_price_barrier,
    mc_price_european,
)
from mc_pricer.payoffs import asian_arithmetic_payoff, barrier_payoff, european_payoff
from mc_pricer.rng import RNG
from mc_pricer.variance_reduction import (
    mc_price_asian_arithmetic_antithetic,
    mc_price_asian_arithmetic_cv,
    mc_price_barrier_antithetic,
    mc_price_european_antithetic,
)

st.set_page_config(page_title="Option Pricer", layout="wide")
st.title("Option Pricer")

# Default barrier as a fraction of spot, per barrier direction.
BARRIER_DEFAULT_RATIO = {"down": 0.90, "up": 1.15}

with st.sidebar:
    st.header("Parameters")

    st.subheader("Market")
    S0 = st.number_input("Spot S0", value=100.0, min_value=0.01, step=1.0)
    K = st.number_input("Strike K", value=100.0, min_value=0.01, step=1.0)
    r = st.number_input(
        "Rate r", value=0.02, min_value=-0.10, max_value=0.30, step=0.005, format="%.3f"
    )
    T = st.number_input("Maturity T", value=1.0, min_value=0.01, step=0.25)
    sigma = st.number_input(
        "Vol σ", value=0.20, min_value=0.01, max_value=2.0, step=0.01, format="%.2f"
    )

    st.subheader("Contract")
    contract = st.selectbox("Type", ["European", "Asian arithmetic", "Barrier"])
    option_type = st.radio("Side", ["call", "put"], horizontal=True)

    barrier = None
    barrier_type = None
    continuity_correction = False
    if contract == "Barrier":
        barrier_type = st.selectbox("Barrier type", ["down_out", "down_in", "up_out", "up_in"])
        direction = barrier_type.split("_")[0]
        barrier = st.number_input(
            "Barrier level",
            value=round(BARRIER_DEFAULT_RATIO[direction] * S0, 2),
            step=1.0,
            key=f"barrier_{direction}",
            help="Up barriers must be above spot, down barriers below.",
        )
        continuity_correction = st.checkbox(
            "Continuity correction (Broadie-Glasserman-Kou)",
            value=False,
            help="Shift the barrier by exp(±0.5826·σ·√Δt) so the discretely monitored "
            "simulation approximates a continuously monitored barrier.",
        )

    st.subheader("Simulation")
    n_paths = st.select_slider(
        "Paths",
        options=[1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 500_000],
        value=100_000,
    )
    steps = st.select_slider("Steps", options=[8, 16, 32, 64, 128, 252], value=64)
    seed = st.number_input("Seed", value=42, step=1)

    st.subheader("Variance reduction")
    vr_options = ["None", "Antithetic"]
    if contract == "Asian arithmetic":
        vr_options.append("Control variate")
    vr = st.selectbox("Method", vr_options)

if contract == "Barrier":
    st.info(
        f"The barrier is checked only on the {steps} simulated dates, so this prices a "
        "discretely monitored contract. A path can cross the barrier and come back "
        "between two dates without being caught, which makes a discrete knock-out worth "
        "more than a continuously monitored one. Tick the continuity correction to "
        "approximate continuous monitoring."
    )


def price_contract():
    rng = RNG(seed=int(seed))
    paths = None
    if contract == "European":
        if vr == "Antithetic":
            result = mc_price_european_antithetic(
                S0, K, T, r, sigma, steps=steps, n_paths=n_paths, rng=rng, option_type=option_type
            )
        else:
            paths = simulate_gbm_paths(S0, T, r, sigma, steps=steps, n_paths=n_paths, rng=rng)
            result = mc_price_european(paths[:, -1], K, T, r, option_type=option_type)
    elif contract == "Asian arithmetic":
        if vr == "Antithetic":
            result = mc_price_asian_arithmetic_antithetic(
                S0, K, T, r, sigma, steps=steps, n_paths=n_paths, rng=rng, option_type=option_type
            )
        elif vr == "Control variate":
            result = mc_price_asian_arithmetic_cv(
                S0, K, T, r, sigma, steps=steps, n_paths=n_paths, rng=rng, option_type=option_type
            )
        else:
            paths = simulate_gbm_paths(S0, T, r, sigma, steps=steps, n_paths=n_paths, rng=rng)
            result = mc_price_asian_arithmetic(paths, K, T, r, option_type=option_type)
    else:
        if vr == "Antithetic":
            result = mc_price_barrier_antithetic(
                S0,
                K,
                barrier,
                T,
                r,
                sigma,
                barrier_type=barrier_type,
                steps=steps,
                n_paths=n_paths,
                rng=rng,
                option_type=option_type,
                continuity_correction=continuity_correction,
            )
        else:
            paths = simulate_gbm_paths(S0, T, r, sigma, steps=steps, n_paths=n_paths, rng=rng)
            result = mc_price_barrier(
                paths,
                K,
                barrier,
                T,
                r,
                barrier_type=barrier_type,
                option_type=option_type,
                sigma=sigma,
                continuity_correction=continuity_correction,
            )
    return result, paths


if st.button("Price"):
    try:
        with st.spinner("Simulating..."):
            t0 = time.perf_counter()
            result, paths = price_contract()
            elapsed_ms = (time.perf_counter() - t0) * 1000
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Result")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Price", f"{result['price']:.4f}")
    col2.metric("Std error", f"{result['stderr']:.5f}")
    col3.metric("95% CI low", f"{result['ci_low']:.4f}")
    col4.metric("95% CI high", f"{result['ci_high']:.4f}")
    col5.metric("Runtime", f"{elapsed_ms:.0f} ms")

    if contract == "European":
        bs = bs_price(S0, K, T, r, sigma, option_type=option_type)
        st.caption(f"Black-Scholes: **{bs:.4f}**  |  Error: {abs(result['price'] - bs):.5f}")
    if contract == "Barrier":
        st.caption(f"Barrier applied to the simulated paths: {result['extra']['barrier_used']:.4f}")
        if barrier_type == "down_out" and option_type == "call":
            cf = bs_down_and_out_call(S0, K, barrier, T, r, sigma)
            st.caption(
                f"Continuously monitored down-and-out call (closed form): **{cf:.4f}**  |  "
                f"Difference: {result['price'] - cf:+.4f}"
            )

    if paths is not None:
        disc = np.exp(-r * T)
        if contract == "European":
            payoffs = disc * european_payoff(paths[:, -1], K, option_type)
        elif contract == "Asian arithmetic":
            payoffs = disc * asian_arithmetic_payoff(paths, K, option_type)
        else:
            level = barrier_level_for_simulation(
                S0,
                barrier,
                T,
                sigma,
                barrier_type=barrier_type,
                steps=steps,
                continuity_correction=continuity_correction,
            )
            payoffs = disc * barrier_payoff(paths, K, level, barrier_type, option_type)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(payoffs[payoffs > 0], bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.axvline(
            result["price"], color="crimson", linewidth=1.5, label=f"Mean = {result['price']:.3f}"
        )
        ax.set_xlabel("Discounted payoff")
        ax.set_ylabel("Frequency")
        ax.set_title("Payoff distribution (paths with positive payoff)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("The histogram is shown for plain Monte Carlo only.")
