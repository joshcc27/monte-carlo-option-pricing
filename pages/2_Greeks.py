import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_common import market_inputs, show_chart, sidebar_hint, store_market, thousands
from mc_pricer.greeks import mc_european_greeks
from mc_pricer.rng import RNG

st.title("Greeks (European options)")
sidebar_hint()

GREEK_LABELS = {
    "delta": "Delta",
    "vega": "Vega",
    "gamma": "Gamma",
    "theta": "Theta (per year)",
    "rho": "Rho",
}

with st.sidebar.form("inputs"):
    st.header("Parameters")
    S0, K, T, r, sigma = market_inputs()
    option_type = st.radio("Side", ["call", "put"], horizontal=True)
    n_paths = st.select_slider(
        "Paths",
        options=[10_000, 50_000, 100_000, 300_000, 1_000_000],
        value=300_000,
        format_func=thousands,
    )
    seed = st.number_input("Seed", value=42, step=1)
    st.form_submit_button("Estimate Greeks", type="primary", on_click=store_market)

st.caption(
    "Delta and vega: pathwise estimator. Gamma, theta and rho: central finite differences "
    "with common random numbers. A European payoff depends only on the final price and the "
    "GBM step is exact, so each path is simulated in one step. The chart shows how far each "
    "estimate is from Black-Scholes in standard errors. About 95% should fall inside the "
    "shaded band."
)


@st.cache_data(max_entries=8, show_spinner="Running Monte Carlo...")
def estimate_greeks(S0, K, T, r, sigma, option_type, n_paths, seed):
    return mc_european_greeks(
        S0,
        K,
        T,
        r,
        sigma,
        option_type=option_type,
        steps=1,
        n_paths=n_paths,
        rng=RNG(seed=seed),
    )


try:
    result = estimate_greeks(S0, K, T, r, sigma, option_type, n_paths, int(seed))
except ValueError as exc:
    st.error(str(exc))
    st.stop()

rows = []
for g, label in GREEK_LABELS.items():
    block = result[g]
    if "pathwise" in block:
        method, mc_val, mc_se = "pathwise", block["pathwise"], block["pathwise_stderr"]
    else:
        method = "finite difference"
        mc_val, mc_se = block["finite_difference"], block["finite_difference_stderr"]
    an_val = block["analytic"]
    rows.append(
        {
            "Greek": label,
            "Method": method,
            "MC": mc_val,
            "Std error": mc_se,
            "Analytic": an_val,
            "Error / stderr": (mc_val - an_val) / mc_se if mc_se > 0 else np.nan,
        }
    )
table = pd.DataFrame(rows).set_index("Greek")

# The chart uses the plain names, so the five labels fit across a phone screen.
points = pd.DataFrame(
    {
        "Greek": [g.capitalize() for g in GREEK_LABELS],
        "z": table["Error / stderr"].to_numpy(),
        "MC": table["MC"].to_numpy(),
        "Black-Scholes": table["Analytic"].to_numpy(),
    }
)
z_scale = alt.Scale(domain=[min(-3.0, points["z"].min() - 0.5), max(3.0, points["z"].max() + 0.5)])
z_title = "(MC − Black-Scholes) / std error"
band = (
    alt.Chart(pd.DataFrame({"low": [-1.96], "high": [1.96]}))
    .mark_rect(color="grey", opacity=0.15)
    .encode(y=alt.Y("low:Q", scale=z_scale, title=z_title), y2="high:Q")
)
zero = alt.Chart(pd.DataFrame({"z": [0.0]})).mark_rule(color="black").encode(y="z:Q")
dots = (
    alt.Chart(points)
    .mark_circle(size=150, color="steelblue", opacity=1)
    .encode(
        x=alt.X("Greek:N", sort=list(points["Greek"]), title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("z:Q", scale=z_scale, title=z_title),
        tooltip=[
            "Greek:N",
            alt.Tooltip("z:Q", title="Std errors from Black-Scholes", format="+.2f"),
            alt.Tooltip("MC:Q", format=".5f"),
            alt.Tooltip("Black-Scholes:Q", format=".5f"),
        ],
    )
)
show_chart(
    (band + zero + dots).properties(
        title=alt.Title(
            "Error of each Greek",
            subtitle="In standard errors. The grey band is ±1.96.",
        )
    )
)

st.dataframe(table.round(5), width="stretch")

st.subheader("Pathwise vs finite difference")
compare = pd.DataFrame(
    {
        GREEK_LABELS[g]: {
            "Pathwise": result[g]["pathwise"],
            "Pathwise stderr": result[g]["pathwise_stderr"],
            "Finite difference": result[g]["finite_difference"],
            "FD stderr": result[g]["finite_difference_stderr"],
        }
        for g in ("delta", "vega")
    }
).T
st.dataframe(compare.round(5), width="stretch")
