from pathlib import Path

import streamlit as st

from app_common import PAGES

st.title("Monte Carlo Option Pricer")
st.markdown(
    "Monte Carlo pricing and Greeks for European, arithmetic Asian and barrier options under "
    "geometric Brownian motion. Each estimator is tested against a closed-form price where one "
    "exists, and against an independent estimator where it doesn't. Code, tests and write-up "
    "are on [GitHub](https://github.com/joshcc27/monte-carlo-option-pricing)."
)

st.subheader("Paths needed for a one-cent 95% half-width")
col1, col2, col3 = st.columns(3)
col1.metric("Plain Monte Carlo", "4.8 million")
col2.metric("Antithetic", "1.9 million")
col3.metric("Control variate", "4,000")
st.caption(
    "Arithmetic Asian call with 64 monitoring dates, S0 = 100, K = 95, r = 1.5%, σ = 25%, "
    "T = 1. The control variate is the geometric Asian, which has an exact price."
)
st.image(
    str(Path(__file__).resolve().parents[1] / "docs" / "asian_variance_reduction.png"),
    caption="Standard error and variance reduction factor against path count, from notebooks/02.",
)

st.subheader("Pages")
# One row of columns at a time, so the cards stack in page order on a phone.
for row_start in range(0, len(PAGES), 3):
    columns = st.columns(3)
    for column, (path, title, icon, _, description) in zip(
        columns, PAGES[row_start : row_start + 3], strict=False
    ):
        with column.container(border=True):
            st.page_link(path, label=title, icon=icon)
            st.caption(description)
