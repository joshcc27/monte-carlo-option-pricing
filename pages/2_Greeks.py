import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mc_pricer.greeks import mc_european_greeks
from mc_pricer.rng import RNG

st.set_page_config(page_title="Greeks", layout="wide")
st.title("Greeks (European options)")

with st.sidebar:
    st.header("Parameters")
    S0 = st.number_input("Spot S0", value=100.0, min_value=0.01, step=1.0)
    K = st.number_input("Strike K", value=100.0, min_value=0.01, step=1.0)
    r = st.number_input(
        "Rate r", value=0.05, min_value=-0.10, max_value=0.30, step=0.005, format="%.3f"
    )
    T = st.number_input("Maturity T", value=1.0, min_value=0.01, step=0.25)
    sigma = st.number_input(
        "Vol σ", value=0.20, min_value=0.01, max_value=2.0, step=0.01, format="%.2f"
    )
    option_type = st.radio("Side", ["call", "put"], horizontal=True)
    n_paths = st.select_slider(
        "Paths", options=[10_000, 50_000, 100_000, 300_000, 1_000_000], value=300_000
    )
    steps = st.select_slider(
        "Steps",
        options=[1, 16, 64],
        value=1,
        help="A European payoff depends only on S_T and the GBM step is exact, "
        "so one step gives the same answer as many.",
    )
    seed = st.number_input("Seed", value=42, step=1)

st.caption(
    "Delta and vega: pathwise estimator. Gamma, theta and rho: central finite "
    "differences with common random numbers. Error bars are 95% intervals."
)

if st.button("Estimate Greeks"):
    with st.spinner("Running Monte Carlo..."):
        result = mc_european_greeks(
            S0,
            K,
            T,
            r,
            sigma,
            option_type=option_type,
            steps=steps,
            n_paths=n_paths,
            rng=RNG(seed=int(seed)),
        )

    greeks_order = ["delta", "vega", "gamma", "theta", "rho"]
    rows = []
    for g in greeks_order:
        block = result[g]
        if "pathwise" in block:
            method, mc_val, mc_se = "pathwise", block["pathwise"], block["pathwise_stderr"]
        else:
            method = "finite difference"
            mc_val, mc_se = block["finite_difference"], block["finite_difference_stderr"]
        an_val = block["analytic"]
        rows.append(
            {
                "Greek": g.capitalize(),
                "Method": method,
                "MC": mc_val,
                "Std error": mc_se,
                "Analytic": an_val,
                "Error / stderr": (mc_val - an_val) / mc_se if mc_se > 0 else np.nan,
            }
        )
    table = pd.DataFrame(rows).set_index("Greek")

    x = np.arange(len(table))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        x - width / 2,
        table["MC"],
        width,
        yerr=1.96 * table["Std error"],
        label="Monte Carlo",
        color="steelblue",
        capsize=4,
    )
    ax.bar(
        x + width / 2, table["Analytic"], width, label="Black-Scholes", color="coral", alpha=0.85
    )
    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    ax.set_ylabel("Value")
    ax.set_title("Monte Carlo Greeks vs Black-Scholes")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(table.round(5), width="stretch")

    st.subheader("Pathwise vs finite difference")
    compare = pd.DataFrame(
        {
            g.capitalize(): {
                "Pathwise": result[g]["pathwise"],
                "Pathwise stderr": result[g]["pathwise_stderr"],
                "Finite difference": result[g]["finite_difference"],
                "FD stderr": result[g]["finite_difference_stderr"],
            }
            for g in ("delta", "vega")
        }
    ).T
    st.dataframe(compare.round(5), width="stretch")
