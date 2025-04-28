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

    This page contains the solution approach, key steps to solve the task.

    ### Steps to Solve:

    1. **Data Collection**:
    - Extract 3 months of historical data: trading volumes, user staking balances, and fees.
    - Map each user to their applicable Pro Tier based on monthly volume and staking holdings.
    - Identify applicable Referral Tiers for traders and affiliates.

    2. **Discount Application**:
    - Apply Pro Tier discounts to user fees.
    - Apply Referral discounts separately based on referral activity.
    - Combine Pro and Referral discounts following the defined logic (ensuring minimum affiliate rewards).

    3. **Fee Reduction Calculation**:
    - Calculate open/close fees before and after applying discounts.
    - Estimate total fee revenue reduction across the platform.

    4. **APR Impact Estimation**:
    - Adjust pool APRs downward proportionally based on fee reductions.
    - Compare APRs before and after the new scheme.

    5. **User Segmentation and Distribution Analysis**:
    - Count and categorize users by:
        - Both volume and staking met
        - Only volume met
        - Only staking met
    - Analyze volume and staking distributions per Pro Tier segment.

    6. **Summary and Visualizations**:
    - Summarize key financial impacts (fee reductions, APR changes).
    - Create distribution charts for users and trading volumes across segments.

    ---

    > *Final calculated outputs, tables, and charts will be added to this page after solving the task.*
    """
)