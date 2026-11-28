import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_common import (
    fits_in_memory,
    market_inputs,
    show_chart,
    sidebar_hint,
    store_market,
    thousands,
)
from mc_pricer.bs_analytics import bs_down_and_out_call, bs_price
from mc_pricer.gbm import simulate_gbm_paths
from mc_pricer.mc import mc_price_asian_arithmetic, mc_price_barrier, mc_price_european
from mc_pricer.payoffs import (
    asian_arithmetic_payoff,
    barrier_level_for_simulation,
    barrier_payoff,
    european_payoff,
)
from mc_pricer.rng import RNG
from mc_pricer.variance_reduction import (
    mc_price_asian_arithmetic_antithetic,
    mc_price_asian_arithmetic_cv,
    mc_price_barrier_antithetic,
    mc_price_european_antithetic,
)

st.title("Option Pricer")
sidebar_hint()

# Default barrier as a fraction of spot, per barrier direction.
BARRIER_DEFAULT_RATIO = {"down": 0.90, "up": 1.15}
BARRIER_LABELS = {
    "down_out": "Down-and-out",
    "down_in": "Down-and-in",
    "up_out": "Up-and-out",
    "up_in": "Up-and-in",
}

with st.sidebar:
    contract = st.selectbox("Contract", ["European", "Asian arithmetic", "Barrier"])
    barrier_type = None
    if contract == "Barrier":
        barrier_type = st.selectbox(
            "Barrier type", list(BARRIER_LABELS), format_func=BARRIER_LABELS.get
        )

    with st.form("inputs"):
        st.subheader("Market")
        S0, K, T, r, sigma = market_inputs()

        st.subheader("Contract")
        option_type = st.radio("Side", ["call", "put"], horizontal=True)
        barrier = None
        continuity_correction = False
        if contract == "Barrier":
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
            format_func=thousands,
        )
        steps = st.select_slider("Steps", options=[8, 16, 32, 64, 128, 252], value=64)
        seed = st.number_input("Seed", value=42, step=1)
        vr_options = ["None", "Antithetic"]
        if contract == "Asian arithmetic":
            vr_options.append("Control variate")
        vr = st.selectbox("Variance reduction", vr_options)

        st.form_submit_button("Price", type="primary", on_click=store_market)

if contract == "Barrier":
    st.info(
        f"The barrier is checked only on the {steps} simulated dates, so this prices a "
        "discretely monitored contract. A path can cross the barrier and come back "
        "between two dates without being caught, which makes a discrete knock-out worth "
        "more than a continuously monitored one. Tick the continuity correction to "
        "approximate continuous monitoring."
    )


@st.cache_data(max_entries=8, show_spinner="Simulating...")
def price_contract(
    contract,
    option_type,
    S0,
    K,
    T,
    r,
    sigma,
    barrier,
    barrier_type,
    continuity_correction,
    n_paths,
    steps,
    seed,
    vr,
):
    """Price the contract. Returns the result, the discounted payoffs (plain MC only) and ms."""
    t0 = time.perf_counter()
    rng = RNG(seed=seed)
    common = {"steps": steps, "n_paths": n_paths, "rng": rng, "option_type": option_type}
    payoffs = None
    disc = np.exp(-r * T)

    if contract == "European":
        if vr == "Antithetic":
            result = mc_price_european_antithetic(S0, K, T, r, sigma, **common)
        else:
            paths = simulate_gbm_paths(S0, T, r, sigma, steps=steps, n_paths=n_paths, rng=rng)
            result = mc_price_european(paths[:, -1], K, T, r, option_type=option_type)
            payoffs = disc * european_payoff(paths[:, -1], K, option_type)
    elif contract == "Asian arithmetic":
        if vr == "Antithetic":
            result = mc_price_asian_arithmetic_antithetic(S0, K, T, r, sigma, **common)
        elif vr == "Control variate":
            result = mc_price_asian_arithmetic_cv(S0, K, T, r, sigma, **common)
        else:
            paths = simulate_gbm_paths(S0, T, r, sigma, steps=steps, n_paths=n_paths, rng=rng)
            result = mc_price_asian_arithmetic(paths, K, T, r, option_type=option_type)
            payoffs = disc * asian_arithmetic_payoff(paths, K, option_type)
    else:
        barrier_args = {
            "barrier_type": barrier_type,
            "continuity_correction": continuity_correction,
        }
        if vr == "Antithetic":
            result = mc_price_barrier_antithetic(
                S0, K, barrier, T, r, sigma, **barrier_args, **common
            )
        else:
            paths = simulate_gbm_paths(S0, T, r, sigma, steps=steps, n_paths=n_paths, rng=rng)
            result = mc_price_barrier(
                paths, K, barrier, T, r, option_type=option_type, sigma=sigma, **barrier_args
            )
            level = barrier_level_for_simulation(S0, barrier, T, sigma, steps=steps, **barrier_args)
            payoffs = disc * barrier_payoff(paths, K, level, barrier_type, option_type)

    return result, payoffs, (time.perf_counter() - t0) * 1000


if not fits_in_memory(n_paths, steps):
    st.stop()

try:
    result, payoffs, elapsed_ms = price_contract(
        contract,
        option_type,
        S0,
        K,
        T,
        r,
        sigma,
        barrier,
        barrier_type,
        continuity_correction,
        n_paths,
        steps,
        int(seed),
        vr,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

METHOD_LABELS = {
    "None": "plain Monte Carlo",
    "Antithetic": "antithetic sampling",
    "Control variate": "control variate",
}
if contract == "Barrier":
    priced = f"{BARRIER_LABELS[barrier_type]} {option_type}, barrier {barrier:g}"
else:
    priced = f"{contract} {option_type}"

st.subheader("Result")
st.caption(f"{priced} · {n_paths:,} paths · {steps} steps · {METHOD_LABELS[vr]}")
col1, col2, col3 = st.columns(3)
col1.metric("Price", f"{result['price']:.4f}")
col2.metric("Std error", f"{result['stderr']:.5f}")
col3.metric("Runtime", f"{elapsed_ms:.0f} ms")
st.caption(f"95% confidence interval: {result['ci_low']:.4f} to {result['ci_high']:.4f}")


def compare_caption(name, reference):
    error = result["price"] - reference
    in_se = f" ({error / result['stderr']:+.1f} standard errors)" if result["stderr"] > 0 else ""
    st.caption(f"{name}: **{reference:.4f}**  |  Error: {error:+.5f}{in_se}")


if contract == "European":
    compare_caption("Black-Scholes", bs_price(S0, K, T, r, sigma, option_type=option_type))
if contract == "Barrier":
    st.caption(f"Barrier applied to the simulated paths: {result['extra']['barrier_used']:.4f}")
    if option_type == "call" and barrier_type in ("down_out", "down_in"):
        down_and_out = bs_down_and_out_call(S0, K, barrier, T, r, sigma)
        if barrier_type == "down_out":
            compare_caption("Continuously monitored down-and-out call (closed form)", down_and_out)
        else:
            compare_caption(
                "Continuously monitored down-and-in call (Black-Scholes call minus down-and-out)",
                bs_price(S0, K, T, r, sigma, option_type="call") - down_and_out,
            )
        if not continuity_correction:
            st.caption(
                "The gap is mainly the bias from checking the barrier on discrete dates only. "
                "Tick the continuity correction to shrink it."
            )

positive = None if payoffs is None else payoffs[payoffs > 0]
if payoffs is None:
    st.info("The histogram is shown for plain Monte Carlo only.")
elif positive.size == 0:
    st.info("No simulated path pays anything, so there is no histogram to show.")
else:
    edge = float(np.quantile(positive, 0.995))
    counts, edges = np.histogram(positive[positive <= edge], bins=60)
    bins = pd.DataFrame({"start": edges[:-1], "end": edges[1:], "paths": counts})
    bars = (
        alt.Chart(bins)
        # Rectangles from zero: a bar mark with x and x2 would be drawn horizontally.
        .mark_rect(color="steelblue", stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X("start:Q", title="Discounted payoff"),
            x2="end:Q",
            y=alt.Y("paths:Q", title="Paths"),
            y2=alt.datum(0),
            tooltip=[
                alt.Tooltip("start:Q", title="From", format=".2f"),
                alt.Tooltip("end:Q", title="To", format=".2f"),
                alt.Tooltip("paths:Q", title="Paths", format=","),
            ],
        )
    )
    mean = (
        alt.Chart(pd.DataFrame({"price": [result["price"]]}))
        .mark_rule(color="crimson", strokeWidth=2)
        .encode(x="price:Q", tooltip=[alt.Tooltip("price:Q", title="Price (mean)", format=".4f")])
    )
    show_chart(
        (bars + mean).properties(
            title=alt.Title("Payoff distribution", subtitle="Paths with a positive payoff")
        )
    )
    st.caption(
        f"{np.mean(payoffs == 0):.0%} of paths pay nothing and are left out. The red line is "
        "the mean payoff, which is the price. The axis stops at the 99.5th percentile."
    )
