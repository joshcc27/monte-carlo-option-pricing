import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_common import fits_in_memory, market_inputs, show_chart, sidebar_hint, store_market
from mc_pricer.bs_analytics import bs_price
from mc_pricer.gbm import simulate_gbm_paths
from mc_pricer.mc import mc_price_european
from mc_pricer.mc_results import discount_payoffs
from mc_pricer.payoffs import european_payoff
from mc_pricer.rng import RNG, QMCGenerator

st.title("Convergence: pseudo-random vs Sobol")
sidebar_hint()

with st.sidebar.form("inputs"):
    st.header("Parameters")
    S0, K, T, r, sigma = market_inputs()
    steps = st.select_slider(
        "Steps (Sobol dimension)",
        options=[1, 4, 16, 32, 64],
        value=32,
        help="The payoff only needs S_T, but more steps raise the Sobol dimension. "
        "That is where the Brownian bridge matters.",
    )
    k_min, k_max = st.slider("Paths: 2^k for k in", min_value=6, max_value=18, value=(8, 14))
    n_seeds = st.slider("Seeds / scrambles per point", min_value=4, max_value=32, value=8)
    show_qmc = st.checkbox("Sobol, time-ordered", value=True)
    show_bb = st.checkbox("Sobol with Brownian bridge", value=True)
    st.form_submit_button("Run convergence", type="primary", on_click=store_market)


@st.cache_data(max_entries=8, show_spinner="Simulating...")
def convergence_curves(S0, K, T, r, sigma, steps, k_min, k_max, n_seeds, show_qmc, show_bb):
    bs_ref = bs_price(S0, K, T, r, sigma, option_type="call")
    counts = 2 ** np.arange(k_min, k_max + 1)

    def price(rng, n):
        paths = simulate_gbm_paths(S0, T, r, sigma, steps=steps, n_paths=int(n), rng=rng)
        return mc_price_european(paths[:, -1], K, T, r, option_type="call")["price"]

    curves = {"Pseudo-random MC": []}
    if show_qmc:
        curves["Sobol, time-ordered"] = []
    if show_bb:
        curves["Sobol + Brownian bridge"] = []
    estimates = {name: [] for name in curves}

    for n in counts:
        runs = {name: [] for name in curves}
        for s in range(n_seeds):
            runs["Pseudo-random MC"].append(price(RNG(seed=1000 + s), n))
            if show_qmc:
                runs["Sobol, time-ordered"].append(price(QMCGenerator(steps, seed=s), n))
            if show_bb:
                qmc = QMCGenerator(steps, seed=s, brownian_bridge=True)
                runs["Sobol + Brownian bridge"].append(price(qmc, n))
        for name, values in runs.items():
            values = np.asarray(values)
            curves[name].append(np.sqrt(np.mean((values - bs_ref) ** 2)))
            estimates[name].append(values)

    # Standard deviation of the discounted payoff, for the sigma / sqrt(N) line.
    pilot = simulate_gbm_paths(S0, T, r, sigma, steps=1, n_paths=200_000, rng=RNG(seed=7))
    pilot_payoffs = european_payoff(pilot[:, -1], K, "call")
    payoff_sd = discount_payoffs(pilot_payoffs, T=T, r=r).std(ddof=1)
    return counts, curves, estimates, bs_ref, payoff_sd


st.markdown(
    "Each point is the root-mean-square error against Black-Scholes over several "
    "independent seeds (pseudo-random) or independent scrambles (Sobol). Path counts "
    "are powers of two, which Sobol sequences need. The dashed black line is "
    "σ_payoff / √N, the theoretical RMSE of plain Monte Carlo."
)

# Sobol paths peak at about 74 bytes per path-step, more than pseudo-random ones.
if fits_in_memory(2**k_max, steps, bytes_per_step=74):
    counts, curves, estimates, bs_ref, payoff_sd = convergence_curves(
        S0, K, T, r, sigma, steps, k_min, k_max, n_seeds, show_qmc, show_bb
    )

    theory_name = "σ / √N"
    colors = {
        "Pseudo-random MC": "steelblue",
        "Sobol, time-ordered": "coral",
        "Sobol + Brownian bridge": "seagreen",
        theory_name: "black",
    }
    runs = pd.DataFrame(
        [
            {"N": int(n), "RMSE": float(e), "Method": name}
            for name, rmse in curves.items()
            for n, e in zip(counts, rmse, strict=True)
        ]
    )
    theory = pd.DataFrame(
        {"N": counts.astype(int), "RMSE": payoff_sd / np.sqrt(counts), "Method": theory_name}
    )
    shown = [*curves, theory_name]
    x = alt.X(
        "N:Q",
        scale=alt.Scale(type="log", base=2),
        title="Number of paths N",
        axis=alt.Axis(format=",d"),
    )
    y = alt.Y("RMSE:Q", scale=alt.Scale(type="log"), title="RMSE vs Black-Scholes")
    color = alt.Color(
        "Method:N",
        scale=alt.Scale(domain=shown, range=[colors[m] for m in shown]),
        # One column, so the legend fits a phone screen.
        legend=alt.Legend(orient="bottom", columns=1, title=None),
    )
    tooltip = ["Method:N", alt.Tooltip("N:Q", format=","), alt.Tooltip("RMSE:Q", format=".2e")]
    lines = alt.Chart(runs).mark_line(point=True).encode(x=x, y=y, color=color, tooltip=tooltip)
    theory_line = (
        alt.Chart(theory)
        .mark_line(strokeDash=[4, 4])
        .encode(x=x, y=y, color=color, tooltip=tooltip)
    )
    show_chart(
        (lines + theory_line).properties(
            title=alt.Title(
                "European call",
                subtitle=f"Black-Scholes {bs_ref:.4f} · {steps} steps · {n_seeds} seeds per point",
            )
        ),
        # The height includes the title, axes and legend, and the plot spans four decades.
        height=500,
    )

    st.subheader(f"Estimates at N = {counts[-1]:,}")
    rows = []
    for name, per_n in estimates.items():
        values = per_n[-1]
        rows.append(
            {
                "Method": name,
                "Mean of runs": values.mean(),
                "Std error of mean": values.std(ddof=1) / np.sqrt(values.size),
                "Error vs BS": values.mean() - bs_ref,
            }
        )
    st.dataframe(pd.DataFrame(rows).set_index("Method").round(6), width="stretch")
    st.caption(
        "The i.i.d. standard error printed by the pricers does not apply to Sobol "
        "points, which are dependent by construction. The Sobol error bars here come "
        "from the spread across independent scrambles (randomised QMC)."
    )
