"""Shared pieces of the Streamlit pages: the page list, market inputs kept across pages,
a memory check and chart display."""

import streamlit as st

# (file, title, icon, URL path, one-line description). The navigation and the overview's
# page cards are both built from this list.
PAGES = [
    (
        "pages/1_Pricer.py",
        "Pricer",
        ":material/calculate:",
        "Pricer",
        "Price a European, Asian or barrier option, with optional variance reduction.",
    ),
    (
        "pages/2_Greeks.py",
        "Greeks",
        ":material/functions:",
        "Greeks",
        "Monte Carlo delta, vega, gamma, theta and rho against Black-Scholes.",
    ),
    (
        "pages/3_Convergence.py",
        "Convergence",
        ":material/trending_down:",
        "Convergence",
        "How fast the error falls with pseudo-random and Sobol sampling.",
    ),
    (
        "pages/4_Variance_Reduction.py",
        "Variance Reduction",
        ":material/compress:",
        "Variance_Reduction",
        "Plain, antithetic and control-variate estimators side by side.",
    ),
    (
        "pages/5_Binomial.py",
        "Binomial",
        ":material/account_tree:",
        "Binomial",
        "CRR tree prices with European and American exercise.",
    ),
]


def sidebar_hint():
    st.caption("Change the inputs in the sidebar. On a phone, open it with » at the top left.")


def show_chart(chart, height=320):
    """Draw an Altair chart at full width. Altair keeps text legible at any screen size."""
    st.altair_chart(chart.properties(height=height), width="stretch")


def thousands(n):
    return f"{n:,}"


# Peak memory per simulated path-step, measured: about 48 bytes for the pricers, 56 for the
# variance reduction page and 74 for Sobol paths. A free Community Cloud app is guaranteed
# only about 1 GB, so each simulation is kept under about 650 MB.
MAX_SIM_BYTES = 650e6

MARKET_DEFAULTS = {"S0": 100.0, "K": 100.0, "r": 0.05, "T": 1.0, "sigma": 0.20}


def fits_in_memory(n_paths, steps, bytes_per_step=48):
    """Return True if the simulation fits the memory budget, otherwise show an error."""
    need = n_paths * steps * bytes_per_step
    if need <= MAX_SIM_BYTES:
        return True
    st.error(
        f"{n_paths:,} paths × {steps} steps would need about {need / 1e9:.1f} GB, more than "
        "the hosted app has. Reduce the paths or the steps."
    )
    return False


def _load_market():
    # Widget state is dropped when you leave a page, so the values live under separate keys
    # and are copied into the widgets on every run.
    for name, value in MARKET_DEFAULTS.items():
        st.session_state.setdefault(name, value)
        st.session_state[f"_{name}"] = st.session_state[name]


def store_market():
    """Keep the submitted market inputs for the other pages. Use as the form's on_click."""
    for name in MARKET_DEFAULTS:
        st.session_state[name] = st.session_state[f"_{name}"]


def market_inputs():
    """Spot, strike, rate, maturity and volatility inputs, remembered across pages."""
    _load_market()
    S0 = st.number_input("Spot S0", min_value=0.01, step=1.0, key="_S0")
    K = st.number_input("Strike K", min_value=0.01, step=1.0, key="_K")
    r = st.number_input(
        "Rate r", min_value=-0.10, max_value=0.30, step=0.005, format="%.3f", key="_r"
    )
    T = st.number_input("Maturity T", min_value=0.01, step=0.25, key="_T")
    sigma = st.number_input(
        "Vol σ", min_value=0.01, max_value=2.0, step=0.01, format="%.2f", key="_sigma"
    )
    return S0, K, T, r, sigma
