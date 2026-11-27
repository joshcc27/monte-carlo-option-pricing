import streamlit as st

from app_common import PAGES

st.set_page_config(page_title="MC Pricer", layout="wide")

navigation = st.navigation(
    [st.Page("pages/0_Overview.py", title="Overview", icon=":material/home:", default=True)]
    + [
        st.Page(path, title=title, icon=icon, url_path=url_path)
        for path, title, icon, url_path, _ in PAGES
    ]
)
navigation.run()
