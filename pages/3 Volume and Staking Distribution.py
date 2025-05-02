# %% import libraries
import streamlit as st
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from functions.functions import (
    centered_header, 
    load_data,
    plot_trading_volume_distribution, 
    plot_staked_amount_distribution,
    plot_volume_by_staking_segment,
    init_key
)

# %% main code

# Set page configuration
st.set_page_config(page_title="Analytics: Volume and Staking Distribution", layout="wide", page_icon=":bar_chart:")

centered_header("Analytics: Volume and Staking Distribution")

st.markdown(
    """
    --- 

    This page provides distribution analysis of **trading volume** and **staking amounts** across user segments.
    
    The visualizations allow us to understand user behavior, define meaningful tiers, and support decision-making for fee configuration or rewards structures.

    ---
    
    ### Visualizations Included

    1. **Distribution of Trading Volume**: 
        - Users are grouped into trading volume segments.
        - Displays total trading volume and trader counts across segments.
    
    2. **Distribution of Staked Amounts**:
        - Users are grouped based on their GMX staking holdings.
        - Displays total staked amounts and staker counts across segments.

    ---
    
    ### Purpose

    - Identify concentration of volume and staking among user groups.
    - Design or adjust Pro Tiers and discount schemes based on real user distributions.
    - Support future fee model optimization based on actual data, not assumptions.

    *Use the sidebar to navigate between pages and explore different analysis sections.*

    """,
    unsafe_allow_html=True
)

# Add a section header before the Trading Volume Distribution plot
st.markdown(
    """
    ---
    ### Distribution of Trading Volume (Last 30 Days)

    Below we present the distribution of user trading volumes segmented into volume brackets. 
    This allows for better understanding of user concentration and supports defining volume-based tiers.
    """,
    unsafe_allow_html=True
)

# Upload data
daily_trading_volume = init_key("daily_trading_volume", lambda: load_data("daily_trading_volume"))
gmx_staked_last = init_key("gmx_staked_last", lambda: load_data("gmx_staked_last"))

# Plot the Trading Volume Distribution chart
plot_trading_volume_distribution(daily_trading_volume)

# --- New Section: Distribution by Staking Segment ---
st.markdown(
    """
    ---
    ### Distribution by Staking Segment

    Compare staking concentration and associated trading behavior across user tiers.
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    plot_staked_amount_distribution(gmx_staked_last)

with col2:
    plot_volume_by_staking_segment(daily_trading_volume, gmx_staked_last)
