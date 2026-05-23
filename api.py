import sys
from pathlib import Path

import streamlit as st

from importers.importer import *

st.set_page_config(
    page_title="Gamma",
    layout="wide",
)

uploaded_file = st.file_uploader("Upload a file", type="xlsx")
if uploaded_file is not None:
    n = import_nodes(uploaded_file)
    st.dataframe(n)

if uploaded_file is not None:
    r = import_results(uploaded_file)
    st.dataframe(r)


