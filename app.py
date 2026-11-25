import streamlit as st

st.set_page_config(page_title="MC Pricer", layout="wide")
st.title("Monte Carlo Option Pricer")
st.markdown(
    """
Monte Carlo pricing and Greeks for European, arithmetic Asian and barrier
options under geometric Brownian motion, checked against closed-form prices.

The sidebar has five pages:

- **Pricer**: price a European, Asian or barrier option with optional
  antithetic or control-variate sampling, and see the payoff histogram.
- **Greeks**: delta, vega, gamma, theta and rho for a European option,
  estimated by Monte Carlo and compared with Black-Scholes.
- **Convergence**: RMSE against path count for pseudo-random and Sobol
  sampling, averaged over several seeds.
- **Variance Reduction**: standard error of plain, antithetic and
  control-variate estimators for an arithmetic Asian call.
- **Binomial**: CRR tree price of a European option and its convergence to
  Black-Scholes.
"""
)
