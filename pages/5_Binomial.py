import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_common import market_inputs, show_chart, sidebar_hint, store_market
from mc_pricer.binomial import price_binomial
from mc_pricer.bs_analytics import bs_price

st.title("CRR binomial tree")
sidebar_hint()

with st.sidebar.form("inputs"):
    st.header("Market")
    S0, K, T, r, sigma = market_inputs()

    st.subheader("Contract")
    option_type = st.radio("Option type", ["call", "put"], horizontal=True)
    exercise = st.radio("Exercise", ["European", "American"], horizontal=True)
    steps = st.select_slider("Steps", options=[10, 50, 100, 200, 500, 1000], value=200)
    st.form_submit_button("Price", type="primary", on_click=store_market)

st.caption(
    "American exercise takes the larger of continuing and exercising at every node. Without "
    "dividends an American call is never exercised early, so it equals the European call."
)


@st.cache_data(max_entries=16)
def tree_price(S0, K, T, r, sigma, steps, option_type, exercise):
    return price_binomial(
        S0, K, T, r, sigma, steps=steps, option_type=option_type, exercise=exercise
    )


try:
    european = tree_price(S0, K, T, r, sigma, steps, option_type, "european")
    american = tree_price(S0, K, T, r, sigma, steps, option_type, "american")
except ValueError as exc:
    st.error(str(exc))
    st.stop()
bs_ref = bs_price(S0, K, T, r, sigma, option_type=option_type)

col1, col2, col3, col4 = st.columns(4)
col1.metric("European (tree)", f"{european:.6f}")
col2.metric("American (tree)", f"{american:.6f}")
col3.metric("Early-exercise premium", f"{american - european:.6f}")
col4.metric("Black-Scholes (European)", f"{bs_ref:.6f}")
st.caption(f"European tree error against Black-Scholes: {european - bs_ref:+.6f}")

st.subheader(f"Convergence of the {exercise} {option_type}")
step_counts, prices, skipped = [], [], []
for n in np.unique(np.logspace(1, 3, 30).astype(int)):
    try:
        prices.append(tree_price(S0, K, T, r, sigma, int(n), option_type, exercise.lower()))
        step_counts.append(int(n))
    except ValueError:
        skipped.append(int(n))
if skipped:
    st.warning(
        f"Skipped step counts {skipped}: the tree is not arbitrage-free there "
        "(exp(r·Δt) falls outside [d, u])."
    )

tree_line = (
    alt.Chart(pd.DataFrame({"Steps": step_counts, "Price": prices}))
    .mark_line(point=True, color="steelblue")
    .encode(
        x=alt.X("Steps:Q", scale=alt.Scale(type="log"), title="Steps"),
        y=alt.Y("Price:Q", scale=alt.Scale(zero=False), title="Option price"),
        tooltip=["Steps:Q", alt.Tooltip("Price:Q", format=".6f")],
    )
)
bs_line = (
    alt.Chart(pd.DataFrame({"Black-Scholes": [bs_ref]}))
    .mark_rule(color="tomato", strokeDash=[6, 4], strokeWidth=2)
    .encode(y="Black-Scholes:Q", tooltip=[alt.Tooltip("Black-Scholes:Q", format=".6f")])
)
show_chart(
    (tree_line + bs_line).properties(
        title=alt.Title(
            f"{exercise} {option_type}, CRR tree",
            subtitle=f"Dashed line: Black-Scholes European price, {bs_ref:.4f}",
        )
    )
)
