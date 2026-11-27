import time

import altair as alt
import pandas as pd
import streamlit as st

from app_common import fits_in_memory, show_chart, sidebar_hint, thousands
from mc_pricer.gbm import simulate_gbm_paths
from mc_pricer.mc import mc_price_asian_arithmetic
from mc_pricer.rng import RNG
from mc_pricer.variance_reduction import (
    mc_price_asian_arithmetic_antithetic,
    mc_price_asian_arithmetic_cv,
)

st.title("Variance reduction: arithmetic Asian call")
sidebar_hint()

# This page keeps its own market defaults, which match the README's benchmark.
with st.sidebar.form("inputs"):
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
    n_paths = st.select_slider(
        "Paths",
        options=[10_000, 50_000, 100_000, 200_000],
        value=100_000,
        format_func=thousands,
    )
    steps = st.select_slider("Monitoring dates", options=[12, 32, 64, 128], value=64)
    seed = st.number_input("Seed", value=42, step=1)
    st.form_submit_button("Run benchmark", type="primary")

st.caption(
    "Antithetic stderr is computed from pair means. The control variate is the "
    "geometric-average Asian payoff on the same paths, whose price is known in closed form."
)


@st.cache_data(max_entries=8, show_spinner="Running three estimators...")
def benchmark(S0, K, T, r, sigma, n_paths, steps, seed):
    """Run the three estimators. Returns {method: (result, seconds)}."""
    common = {"steps": steps, "n_paths": n_paths, "option_type": "call"}

    def plain():
        paths = simulate_gbm_paths(
            S0, T, r, sigma, steps=steps, n_paths=n_paths, rng=RNG(seed=seed)
        )
        return mc_price_asian_arithmetic(paths, K, T, r, option_type="call")

    runs = {
        "Plain MC": plain,
        "Antithetic": lambda: mc_price_asian_arithmetic_antithetic(
            S0, K, T, r, sigma, rng=RNG(seed=seed), **common
        ),
        "Control variate": lambda: mc_price_asian_arithmetic_cv(
            S0, K, T, r, sigma, rng=RNG(seed=seed), **common
        ),
    }
    timed = {}
    for name, run in runs.items():
        t0 = time.perf_counter()
        result = run()
        timed[name] = (result, time.perf_counter() - t0)
    return timed


if not fits_in_memory(n_paths, steps, bytes_per_step=56):
    st.stop()

try:
    timed = benchmark(S0, K, T, r, sigma, n_paths, steps, int(seed))
except ValueError as exc:
    st.error(str(exc))
    st.stop()

methods = list(timed)
results = [timed[m][0] for m in methods]
seconds = [timed[m][1] for m in methods]
stderrs = [res["stderr"] for res in results]
plain_se, plain_time = stderrs[0], seconds[0]
vrf = [(plain_se / s) ** 2 for s in stderrs]
efficiency = [f * plain_time / t for f, t in zip(vrf, seconds, strict=True)]

bars_data = pd.DataFrame(
    {
        "Method": methods,
        "Std error": stderrs,
        "Label": [f"VRF {f:,.0f}" if f >= 100 else f"VRF {f:.1f}" for f in vrf],
    }
)
base = alt.Chart(bars_data).encode(
    x=alt.X("Method:N", sort=methods, title=None, axis=alt.Axis(labelAngle=0)),
    # Headroom above the tallest bar, so its label stays inside the chart.
    y=alt.Y(
        "Std error:Q", scale=alt.Scale(domain=[0, max(stderrs) * 1.18]), title="Standard error"
    ),
)
bars = base.mark_bar(size=60).encode(
    color=alt.Color(
        "Method:N",
        scale=alt.Scale(domain=methods, range=["steelblue", "seagreen", "coral"]),
        legend=None,
    ),
    tooltip=["Method:N", alt.Tooltip("Std error:Q", format=".6f"), "Label:N"],
)
labels = base.mark_text(dy=-10, fontSize=13).encode(text="Label:N")
show_chart((bars + labels).properties(title=f"Arithmetic Asian call, {n_paths:,} paths"))

st.dataframe(
    pd.DataFrame(
        {
            "Method": methods,
            "Price": [res["price"] for res in results],
            "Std error": stderrs,
            "Variance reduction factor": vrf,
            "Run time (ms)": [t * 1000 for t in seconds],
            "Efficiency gain": efficiency,
        }
    ).set_index("Method"),
    column_config={
        "Price": st.column_config.NumberColumn(format="%.4f"),
        "Std error": st.column_config.NumberColumn(format="%.6f"),
        "Variance reduction factor": st.column_config.NumberColumn(format="%.1f"),
        "Run time (ms)": st.column_config.NumberColumn(format="%.0f"),
        "Efficiency gain": st.column_config.NumberColumn(format="%.1f"),
    },
    width="stretch",
)
st.caption(
    "Variance reduction factor = (plain stderr / method stderr)², the factor by which the "
    "method cuts the paths needed for a given accuracy. Efficiency gain = variance reduction "
    "factor × (plain run time / method run time), the speed-up for a given accuracy once run "
    "time is counted. Plain run time includes simulating its paths."
)
