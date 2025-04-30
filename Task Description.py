# %%
# libraries
import streamlit as st

from functions.functions import (
    centered_header, 
    load_data, 
    init_key
)


# %%
# Set up the page configuration
st.set_page_config(page_title="Analytics: Fee Change Estimation", layout="wide", page_icon=":receipt:")

# Page 1 content
centered_header("Analytics: Fee Change Estimation")

st.markdown(
    """
    ---

    ### Goals

    1. Calculate changes to pool APRs and reductions in open/close fees if a scheme with Volume Tiers, Staking Discounts, and Referral Rewards is implemented.
    2. Estimate how much of the negative impact from fee reductions could be offset by expected price impacts.

    ---

    ### Constraints

    - The period for all calculations: **January 1 to March 31** (3 months after enabling liquidation fees).
    - The Pro Tiers configuration should be kept abstract and flexible for future adjustments.

    ---

    ## Inputs

    ### Input 1: Pro Tiers Configuration (Default)

    | **Pro Tier** | **Fee** | **Monthly Volume** |  | **GMX Staking Holdings** |  | **Bonus: Both Criteria Met** |
    |--------------|---------|--------------------|--|---------------------------|--|-------------------------------|
    | Pro 1        | 4bps / 6bps | 0 - 5M         |  | <20k                      |  |                               |
    | Pro 2        | 3.5bps / 5.5bps (**10% discount**) | 5M - 40M | OR | 20k GMX | Both = | **15% discount** |
    | Pro 3        | 3bps / 5bps (**20% discount**) | 40M - 200M | OR | 50k GMX | Both = | **25% discount** |
    | Pro 4        | 2.5bps / 4.5bps (**30% discount**) | 200M+ | OR | 100k GMX | Both = | **35% discount** |

    ---

    ### Input 2: Referral Configuration

    | Tier | Requirement | Discount for Trader | Discount for Affiliate |
    |------|-------------|----------------------|-------------------------|
    | Tier 1 | No requirement | 5% | 5% |
    | Tier 2 | At least 15 active users per week and $5M combined weekly volume | 10% | 10% |
    | Tier 3 | At least 30 active users per week and $25M combined weekly volume | 10% | 15% |

    - **Minimum Affiliate Reward**: X%  
    - The final trader discount, affiliate reward, and protocol income are determined by the combination of Pro Tiers and Referral discounts.

    ---

    ### Referral Example Logic

    | Example | Affiliate Discount | Trader Discount | Pro Discount | Min Affiliate Reward | Total Referral Rebate | Affiliate Reward Cap | Affiliate Gets | Trader Gets | Protocol Gets |
    |---------|---------------------|-----------------|--------------|----------------------|-----------------------|----------------------|----------------|-------------|---------------|
    | 1 | 10% | 10% | 5%  | 5% | 20% | Max(20%-5%,5%) = 15% | 10% | 10% | 80% |
    | 2 | 10% | 10% | 13% | 5% | 20% | Max(20%-13%,5%) = 7% | 7% | 13% | 80% |
    | 3 | 10% | 10% | 18% | 5% | 20% | Max(20%-18%,5%) = 5% | 5% | 18% | 77% |
    | 4 | 10% | 10% | 25% | 5% | 20% | Max(20%-25%,5%) = 5% | 5% | 25% | 70% |

    ---

    ## Outputs

    ### High-level Fee Reduction Estimation

    | Category | As-is | New Scheme |
    |----------|-------|------------|
    | Open/Close Fees | X1 | X2 (-Y%) |
    | Total Fees      | X1 | X2 (-Y%) |

    ---

    ### APR Impact for Pools

    | Pool | APR Before | APR After |
    |------|------------|-----------|
    | btc-usdc | X1 | X1 (-Y%) |
    | eth-usdc | X2 | X2 (-Y%) |
    | ... | X3 | X3 (-Y%) |

    ---

    ### User Distribution Across Pro Categories

    | Pro Tier | Both (Volume + Staking) | Only Volume | Only Staking |
    |----------|-------------------------|-------------|--------------|
    | Pro 1 | X1 (Y% of total) | X1 (Y% of total) | X1 (Y% of total) |
    | Pro 2 | X2 (Y% of total) | X2 (Y% of total) | X2 (Y% of total) |
    | Pro 3 | X3 (Y% of total) | X3 (Y% of total) | X3 (Y% of total) |
    | Pro 4 | X4 (Y% of total) | X4 (Y% of total) | X4 (Y% of total) |

    ---

    ### Volume and Staking Distribution

    - Distribution of total volume across user segments.
    - Distribution of total staked amounts across user segments.

    *Visualizations such as bar charts or histograms are recommended to show distributions clearly.*

    """,
    unsafe_allow_html=True
)

# Upload data
daily_trading_volume = init_key("daily_trading_volume", lambda: load_data("daily_trading_volume"))
gmx_staking = init_key("gmx_staking", lambda: load_data("gmx_staking"))
gmx_staked_last = init_key("gmx_staked_last", lambda: load_data("gmx_staked_last"))

# %%
