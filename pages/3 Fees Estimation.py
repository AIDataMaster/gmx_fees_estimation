# %% import libraries
import streamlit as st

from functions.functions import (
    centered_header
)

# %% main code

# Set page configuration
st.set_page_config(page_title="Analytics: Fee Change Estimation", layout="wide", page_icon=":abacus:")

centered_header('Analytics: Fee Change Estimation')

st.write(
    """
    --- 

    This page contains the final calculated results.
    """
)