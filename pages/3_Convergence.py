import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mc_pricer.bs_analytics import bs_price
from mc_pricer.gbm import simulate_gbm_paths
from mc_pricer.mc import mc_price_european
from mc_pricer.mc_results import discount_payoffs
from mc_pricer.payoffs import european_payoff
from mc_pricer.rng import RNG, QMCGenerator

st.set_page_config(page_title="Convergence", layout="wide")
st.title("Convergence: pseudo-random vs Sobol")

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
    steps = st.select_slider(
        "Steps (Sobol dimension)",
        options=[1, 4, 16, 32, 64],
        value=32,
        help="The payoff only needs S_T, but more steps raise the Sobol dimension. "
        "That is where the Brownian bridge matters.",
    )
    k_min, k_max = st.slider("Paths: 2^k for k in", min_value=6, max_value=18, value=(8, 16))
    n_seeds = st.slider("Seeds / scrambles per point", min_value=4, max_value=32, value=10)
    show_qmc = st.checkbox("Sobol, time-ordered", value=True)
    show_bb = st.checkbox("Sobol with Brownian bridge", value=True)


@st.cache_data
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
    "are powers of two, which Sobol sequences need. The dotted line is "
    "σ_payoff / √N, the theoretical RMSE of plain Monte Carlo."
)

if st.button("Run convergence"):
    with st.spinner("Simulating..."):
        counts, curves, estimates, bs_ref, payoff_sd = convergence_curves(
            S0, K, T, r, sigma, steps, k_min, k_max, n_seeds, show_qmc, show_bb
        )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    styles = {
        "Pseudo-random MC": ("o-", "steelblue"),
        "Sobol, time-ordered": ("s--", "coral"),
        "Sobol + Brownian bridge": ("^-", "seagreen"),
    }
    for name, rmse in curves.items():
        marker, color = styles[name]
        ax.loglog(counts, rmse, marker, color=color, label=name)
    ax.loglog(counts, payoff_sd / np.sqrt(counts), "k:", linewidth=1, label="σ / √N")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of paths N")
    ax.set_ylabel("RMSE vs Black-Scholes")
    ax.set_title(f"European call, BS = {bs_ref:.4f}, {steps} steps, {n_seeds} seeds per point")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

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
