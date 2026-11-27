"""Smoke tests for the Streamlit app: every page runs at its defaults and shows a result."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = [
    "pages/1_Pricer.py",
    "pages/2_Greeks.py",
    "pages/3_Convergence.py",
    "pages/4_Variance_Reduction.py",
    "pages/5_Binomial.py",
]


def open_app():
    return AppTest.from_file(APP, default_timeout=120).run()


def test_overview_page_runs():
    at = open_app()
    assert not at.exception
    assert len(at.metric) == 3  # paths needed by each estimator
    assert len(at.page_link) == len(PAGES)


@pytest.mark.parametrize("page", PAGES)
def test_page_shows_a_result_without_pressing_anything(page):
    at = open_app()
    at.switch_page(page).run()
    assert not at.exception
    assert len(at.metric) + len(at.dataframe) > 0
    assert not at.error


def test_market_inputs_carry_across_pages():
    at = open_app()
    at.switch_page("pages/5_Binomial.py").run()
    at.number_input(key="_S0").set_value(120.0)
    at.form_submit_button[0].click().run()
    assert not at.exception
    at.switch_page("pages/2_Greeks.py").run()
    assert at.number_input(key="_S0").value == 120.0


def test_pricer_refuses_a_simulation_too_big_for_the_hosted_app():
    at = open_app()
    at.switch_page("pages/1_Pricer.py").run()
    at.select_slider[0].set_value(500_000)  # paths
    at.select_slider[1].set_value(252)  # steps
    at.form_submit_button[0].click().run()
    assert not at.exception
    assert "would need about" in at.error[0].value
    assert len(at.metric) == 0
