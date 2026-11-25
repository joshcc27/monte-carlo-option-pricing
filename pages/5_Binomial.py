import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from mc_pricer.binomial import price_binomial
from mc_pricer.bs_analytics import bs_price

st.set_page_config(page_title="Binomial Tree", layout="wide")
st.title("CRR binomial tree (European exercise)")

with st.sidebar:
    st.header("Market")
    S0 = st.number_input("Spot S0", value=100.0, min_value=0.01, step=1.0)
    K = st.number_input("Strike K", value=100.0, min_value=0.01, step=1.0)
    r = st.number_input(
        "Rate r", value=0.05, min_value=-0.10, max_value=0.30, step=0.005, format="%.3f"
    )
    T = st.number_input("Maturity T", value=1.0, min_value=0.01, step=0.25)
    sigma = st.number_input(
        "Vol σ", value=0.20, min_value=0.01, max_value=2.0, step=0.01, format="%.2f"
    )

    st.subheader("Contract")
    option_type = st.radio("Option type", ["call", "put"], horizontal=True)
    steps = st.select_slider("Steps", options=[10, 50, 100, 200, 500, 1000], value=200)

st.caption("European exercise only. There is no early-exercise (American) check.")

if st.button("Price"):
    bs_ref = bs_price(S0, K, T, r, sigma, option_type=option_type)
    try:
        binom_price = price_binomial(S0, K, T, r, sigma, steps=steps, option_type=option_type)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Binomial price", f"{binom_price:.6f}")
    col2.metric("Black-Scholes", f"{bs_ref:.6f}")
    col3.metric("Abs error vs BS", f"{abs(binom_price - bs_ref):.6f}")

    st.subheader("Convergence to Black-Scholes")
    step_counts, prices, skipped = [], [], []
    for n in np.unique(np.logspace(1, 3, 30).astype(int)):
        try:
            prices.append(price_binomial(S0, K, T, r, sigma, steps=int(n), option_type=option_type))
            step_counts.append(int(n))
        except ValueError:
            skipped.append(int(n))
    if skipped:
        st.warning(
            f"Skipped step counts {skipped}: the tree is not arbitrage-free there "
            "(exp(r·Δt) falls outside [d, u])."
        )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(step_counts, prices, "o-", color="steelblue", markersize=4, label="Binomial")
    ax.axhline(
        bs_ref, color="tomato", linestyle="--", linewidth=1.5, label=f"Black-Scholes = {bs_ref:.4f}"
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Option price")
    ax.set_title(f"CRR {option_type}: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
