import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from mc_pricer.gbm import simulate_gbm_paths
from mc_pricer.mc import mc_price_asian_arithmetic
from mc_pricer.rng import RNG
from mc_pricer.variance_reduction import (
    mc_price_asian_arithmetic_antithetic,
    mc_price_asian_arithmetic_cv,
)

st.set_page_config(page_title="Variance Reduction", layout="wide")
st.title("Variance reduction: arithmetic Asian call")

with st.sidebar:
    st.header("Parameters")
    S0 = st.number_input("Spot S0", value=100.0, min_value=0.01, step=1.0)
    K = st.number_input("Strike K", value=95.0, min_value=0.01, step=1.0)
    r = st.number_input(
        "Rate r", value=0.015, min_value=-0.10, max_value=0.30, step=0.005, format="%.3f"
    )
    T = st.number_input("Maturity T", value=1.0, min_value=0.01, step=0.25)
    sigma = st.number_input(
        "Vol σ", value=0.25, min_value=0.01, max_value=2.0, step=0.01, format="%.2f"
    )
    n_paths = st.select_slider("Paths", options=[10_000, 50_000, 100_000, 200_000], value=100_000)
    steps = st.select_slider("Monitoring dates", options=[12, 32, 64, 128], value=64)
    seed = st.number_input("Seed", value=42, step=1)

st.caption(
    "Antithetic stderr is computed from pair means. The control variate is the "
    "geometric-average Asian payoff on the same paths, whose price is known in closed form."
)

if st.button("Run benchmark"):
    with st.spinner("Running three estimators..."):
        paths = simulate_gbm_paths(
            S0, T, r, sigma, steps=steps, n_paths=n_paths, rng=RNG(seed=int(seed))
        )
        plain = mc_price_asian_arithmetic(paths, K, T, r, option_type="call")
        anti = mc_price_asian_arithmetic_antithetic(
            S0,
            K,
            T,
            r,
            sigma,
            steps=steps,
            n_paths=n_paths,
            rng=RNG(seed=int(seed)),
            option_type="call",
        )
        cv = mc_price_asian_arithmetic_cv(
            S0,
            K,
            T,
            r,
            sigma,
            steps=steps,
            n_paths=n_paths,
            rng=RNG(seed=int(seed)),
            option_type="call",
        )

    methods = ["Plain MC", "Antithetic", "Control variate"]
    results = [plain, anti, cv]
    stderrs = [res["stderr"] for res in results]
    vrf = [(plain["stderr"] / s) ** 2 for s in stderrs]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(methods, stderrs, color=["steelblue", "seagreen", "coral"], width=0.5)
    for bar, factor in zip(bars, vrf, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stderrs) * 0.01,
            f"VRF {factor:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Standard error")
    ax.set_title(f"Arithmetic Asian call, {n_paths:,} paths")
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        pd.DataFrame(
            {
                "Method": methods,
                "Price": [f"{res['price']:.4f}" for res in results],
                "Std error": [f"{s:.6f}" for s in stderrs],
                "Variance reduction factor": [f"{f:.1f}" for f in vrf],
            }
        ).set_index("Method"),
        width="stretch",
    )
    st.caption(
        "Variance reduction factor = (plain stderr / method stderr)². It is the "
        "factor by which the method cuts the number of paths needed for a given "
        "accuracy, before accounting for run time."
    )
