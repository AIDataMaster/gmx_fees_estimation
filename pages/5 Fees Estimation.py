# %% Imports
import streamlit as st
import pandas as pd
import numpy as np

from datetime import date

from functions.functions import (
    centered_header,
    load_data,
    assign_discount,
    plot_user_distribution,
    init_key
)

# %% Page config
st.set_page_config(page_title="Analytics: Fee Estimation", layout="wide", page_icon=":money_with_wings:")

centered_header("Analytics: Fee Estimation")

st.markdown(
    """
    ---
    """
)

# %% Load data (uses session_state via load_data)
daily_trading_volume = init_key("daily_trading_volume", lambda: load_data("daily_trading_volume"))
gmx_staking = init_key("gmx_staking", lambda: load_data("gmx_staking"))
gmx_staked_last = init_key("gmx_staked_last", lambda: load_data("gmx_staked_last"))
fees_data = init_key("fees_data", lambda: load_data("fees_data"))

# %% Section 1 — Calculate Pro Tiers
with st.expander("1. Calculate Pro Tiers", expanded=False):
    st.markdown("This step defines Pro Tiers for users based on trading volume and staking amounts.")

    if 'pro_tiers_df_vs' not in st.session_state:
        with st.spinner("Calculating Pro Tiers..."):
            # Generate daily date reference
            dates_df = pd.DataFrame(pd.date_range(start='2024-12-01', end='2025-03-31', freq='D'), columns=['date'])
            dates_df['month'] = dates_df['date'].dt.to_period('M').dt.start_time
            dates_df['date'] = dates_df['date'].dt.date
            dates_df['month'] = dates_df['month'].dt.date

            # Filter trading volume
            daily_trading_volume['block_date'] = pd.to_datetime(daily_trading_volume['block_date']).dt.date
            trading_volume_f = daily_trading_volume[
                (daily_trading_volume['block_date'] >= date(2024, 12, 1)) &
                (daily_trading_volume['block_date'] < date(2025, 4, 1))
            ].rename(columns={'block_date': 'date'})

            # Cross join traders with all dates
            dates_df['_key'] = 1
            trading_volume_f['_key'] = 1
            pro_tiers_df_v = pd.merge(dates_df, trading_volume_f[['trader', '_key']].drop_duplicates(), on='_key').drop('_key', axis=1)
            trading_volume_f.drop('_key', axis=1, inplace=True)

            # Add trading volume
            pro_tiers_df_v = pro_tiers_df_v.merge(trading_volume_f, on=['date', 'trader'], how='left')
            pro_tiers_df_v['trading_volume'].fillna(0, inplace=True)

            # Calculate 30-day rolling volume
            pro_tiers_df_v = pro_tiers_df_v.sort_values(by=['trader', 'date']).reset_index(drop=True)
            pro_tiers_df_v['trading_volume_30d'] = (
                pro_tiers_df_v.groupby('trader')
                .rolling(30, on='date', min_periods=1)['trading_volume']
                .sum()
                .reset_index(level=0, drop=True)
                .values
            )

            # Assign Pro Tier based on volume
            pro_tiers_df_v['pro_tier_v'] = pro_tiers_df_v['trading_volume_30d'].apply(
                lambda x: 4 if x >= 200_000_000 else (3 if x >= 40_000_000 else (2 if x >= 5_000_000 else 1))
            )
            pro_tiers_df_v[['trading_volume_30d_sh', 'pro_tier_v_sh']] = (
                pro_tiers_df_v.groupby('trader')[['trading_volume_30d', 'pro_tier_v']].shift(1)
            )
            pro_tiers_df_v = pro_tiers_df_v[pro_tiers_df_v['date'] >= date(2025, 1, 1)].reset_index(drop=True)

            # Handle staking
            gmx_staking['block_date'] = pd.to_datetime(gmx_staking['block_date']).dt.date
            gmx_staking_st = (
                gmx_staking[gmx_staking['block_date'] <= date(2025, 1, 1)]
                .sort_values(by=['account', 'block_date'])
                .groupby('account')
                .tail(1)
            )
            gmx_staking_st['block_date'] = date(2025, 1, 1)

            gmx_staking_st2 = gmx_staking[gmx_staking['block_date'] > date(2025, 1, 1)]
            gmx_staking_t = (
                pd.concat([gmx_staking_st, gmx_staking_st2])
                .sort_values(by='block_date')
                .rename(columns={'block_date': 'date', 'account': 'trader'})
                .reset_index(drop=True)
            )

            # Merge with volume
            pro_tiers_df_vs = pro_tiers_df_v.merge(gmx_staking_t, on=['date', 'trader'], how='left')
            pro_tiers_df_vs['staked_amount'] = (
                pro_tiers_df_vs.groupby('trader')['staked_amount']
                .ffill()
                .fillna(0)
            )

            # Assign staking tiers
            pro_tiers_df_vs['pro_tier_s'] = pro_tiers_df_vs['staked_amount'].apply(
                lambda x: 4 if x >= 100_000 else (3 if x >= 50_000 else (2 if x >= 20_000 else 1))
            )

            # Compute discount
            pro_tiers_df_vs['pro_discount'] = pro_tiers_df_vs.apply(assign_discount, axis=1)
            pro_tiers_df_vs['pro_tier'] = pro_tiers_df_vs[['pro_tier_v_sh', 'pro_tier_s']].max(axis=1).astype(int)

            # Save to session state
            st.session_state['pro_tiers_df_vs'] = pro_tiers_df_vs

            st.success("✅ Pro Tier assignment complete.")
    else:
        st.success("✅ Pro Tier data already loaded from session state.")


# %% Section 2 — Fees Data with Pro Tiers
with st.expander("2. Fees Data with Pro Tiers", expanded=False):
    st.markdown("We attach Pro Tier data to raw fee transactions to prepare for fee impact analysis.")

    if 'fees_data_tiers' not in st.session_state:
        with st.spinner("Merging Pro Tier data with fees data..."):
            # Ensure required dependency from Section 1
            if 'pro_tiers_df_vs' not in st.session_state:
                st.error("❌ Please run Section 1 first to calculate Pro Tiers.")
            else:
                # Preprocess fees_data
                fees_data['block_date'] = pd.to_datetime(fees_data['block_date']).dt.date
                fees_data = fees_data.rename(columns={'block_date': 'date', 'account': 'trader'})

                # Filter date range
                fees_data = fees_data[
                    (fees_data['date'] >= date(2025, 1, 1)) &
                    (fees_data['date'] < date(2025, 4, 1))
                ].reset_index(drop=True)

                # Merge with Pro Tiers
                pro_tiers_df_vs = st.session_state['pro_tiers_df_vs']
                fees_data_tiers = fees_data.merge(
                    pro_tiers_df_vs[['date', 'trader', 'pro_tier', 'pro_tier_s', 'pro_tier_v_sh', 'pro_discount']],
                    on=['date', 'trader'],
                    how='left'
                )

                # Fill missing values
                fees_data_tiers['pro_tier'].fillna(1, inplace=True)
                fees_data_tiers['pro_tier_s'].fillna(1, inplace=True)
                fees_data_tiers['pro_tier_v_sh'].fillna(1, inplace=True)
                fees_data_tiers['pro_discount'].fillna(0, inplace=True)

                # Save to session state
                st.session_state['fees_data_tiers'] = fees_data_tiers

                st.success("✅ Fees data successfully enriched with Pro Tier info.")
    else:
        st.success("✅ Fees data with Pro Tiers already loaded from session state.")

# %% Section 3 — User Distribution Across Categories
with st.expander("3. User Distribution Across Categories", expanded=False):
    st.markdown("We analyze the overlap of volume- and staking-based qualifications across Pro Tiers.")

    if 'tiers_stats' not in st.session_state:
        with st.spinner("Computing user category splits by Pro Tier..."):
            pro_tiers_df_vs = st.session_state['pro_tiers_df_vs']
            tiers_stats_time = pro_tiers_df_vs.copy()

            n_tiers = 4
            tiers_stats_time['both'] = False
            tiers_stats_time['only_volume'] = False
            tiers_stats_time['only_staking'] = False

            for tier_n in np.arange(1, n_tiers + 1):
                tier_df = tiers_stats_time[tiers_stats_time['pro_tier'] == tier_n]
                idx = tier_df.index
                vol_cond = tier_df['pro_tier_v_sh'] == tier_n
                stk_cond = tier_df['pro_tier_s'] == tier_n

                tiers_stats_time.loc[idx, 'both'] = vol_cond & stk_cond
                tiers_stats_time.loc[idx, 'only_volume'] = vol_cond & ~stk_cond
                tiers_stats_time.loc[idx, 'only_staking'] = ~vol_cond & stk_cond

            tiers_stats_time = (
                tiers_stats_time[['date', 'pro_tier', 'both', 'only_volume', 'only_staking']]
                .groupby(['date', 'pro_tier'])
                .sum()
                .reset_index()
            )

            total_traders = pro_tiers_df_vs['trader'].nunique()
            tiers_stats_time_2 = (100 * tiers_stats_time.iloc[:, 2:] / total_traders).round(2)
            tiers_stats_time_2.columns = [f"{col}_perc" for col in tiers_stats_time_2.columns]

            tiers_stats = pd.concat([tiers_stats_time, tiers_stats_time_2], axis=1)
            tiers_stats_snapshot = tiers_stats.groupby('pro_tier').tail(1).reset_index(drop=True)

            st.session_state['tiers_stats'] = tiers_stats
            st.session_state['tiers_stats_snapshot'] = tiers_stats_snapshot

            st.success("✅ Distribution stats calculated.")
    else:
        tiers_stats = st.session_state['tiers_stats']
        tiers_stats_snapshot = st.session_state['tiers_stats_snapshot']
        st.success("✅ Distribution stats loaded from session state.")

    with st.container():
        st.markdown("### 📋 User Distribution Across Pro Categories (Snapshot - March 31, 2025)")

        snapshot = tiers_stats_snapshot.copy()

        # Format values as "count (percent%)" with commas and full perc
        snapshot['Both (Volume + Staking)'] = snapshot.apply(
            lambda row: f"{row['both']:,.0f} ({row['both_perc']}%)", axis=1)
        snapshot['Only Volume'] = snapshot.apply(
            lambda row: f"{row['only_volume']:,.0f} ({row['only_volume_perc']}%)", axis=1)
        snapshot['Only Staking'] = snapshot.apply(
            lambda row: f"{row['only_staking']:,.0f} ({row['only_staking_perc']}%)", axis=1)

        display_df = snapshot[['pro_tier', 'Both (Volume + Staking)', 'Only Volume', 'Only Staking']].copy()
        display_df['pro_tier'] = display_df['pro_tier'].apply(lambda x: f"Pro {x}")
        display_df = display_df.rename(columns={'pro_tier': 'Pro Tier'})
        display_df = display_df.set_index('Pro Tier')

        # Center and display the styled table
        styled_table = display_df.style.set_properties(**{
            'text-align': 'center'
        }).set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]}
        ])

        st.table(styled_table)

        # Optional charts (organized in columns)
        st.markdown("### 📈 Line Charts: Absolute & Relative Distributions")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Absolute - Both**")
            plot_user_distribution(tiers_stats, value_type='absolute', category='both')

        with col2:
            st.markdown("**Relative - Both (%)**")
            plot_user_distribution(tiers_stats, value_type='relative', category='both')

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Absolute - Only Volume**")
            plot_user_distribution(tiers_stats, value_type='absolute', category='only_volume')

        with col4:
            st.markdown("**Relative - Only Volume (%)**")
            plot_user_distribution(tiers_stats, value_type='relative', category='only_volume')

        col5, col6 = st.columns(2)
        with col5:
            st.markdown("**Absolute - Only Staking**")
            plot_user_distribution(tiers_stats, value_type='absolute', category='only_staking')

        with col6:
            st.markdown("**Relative - Only Staking (%)**")
            plot_user_distribution(tiers_stats, value_type='relative', category='only_staking')

# %% Section 4 — Calculate New Discounts
with st.expander("4. Calculate New Discounts", expanded=False):
    st.markdown("Referral and Pro Tier discounts are combined to compute final trader and affiliate rewards.")

    if 'fees_data_tiers' not in st.session_state:
        st.error("❌ Please complete Section 2 first (Fees Data with Pro Tiers).")
    else:
        if 'fees_with_discounts' not in st.session_state:
            with st.spinner("Calculating combined referral and pro tier rewards..."):
                fees_data_tiers = st.session_state['fees_data_tiers'].copy()

                # Rename referral columns for clarity
                rename_cols = {
                    'referral_total_rebate_factor': 'total_referral_rebate',
                    'referral_adjusted_affiliate_reward_factor': 'affiliate_rebate',
                    'referral_trader_discount_factor': 'trader_rebate'
                }
                fees_data_tiers = fees_data_tiers.rename(columns=rename_cols)

                # Step-by-step logic
                fees_data_tiers['min_affiliate_reward'] = 0.05

                # Cap: max(total rebate - pro discount, 5%)
                fees_data_tiers['affiliate_reward_cap'] = (
                    fees_data_tiers['total_referral_rebate'] - fees_data_tiers['pro_discount']
                ).clip(lower=0.05)

                # Final values
                fees_data_tiers['affiliate_gets'] = fees_data_tiers[[
                    'affiliate_rebate', 'affiliate_reward_cap'
                ]].min(axis=1)

                fees_data_tiers['trader_gets'] = fees_data_tiers[[
                    'trader_rebate', 'pro_discount'
                ]].max(axis=1)

                # Save result
                st.session_state['fees_with_discounts'] = fees_data_tiers

                st.success("✅ Final discount calculations complete.")
        else:
            st.success("✅ Discounts already calculated and loaded from session state.")


# %% Section 5 — Recalculate Fees
with st.expander("5. Recalculate Fees", expanded=False):
    st.markdown("We apply new discounts to compute updated fees for open/close trades.")

    if 'fees_with_discounts' not in st.session_state:
        st.error("❌ Please complete Section 4 first to calculate discounts.")
    else:
        if 'fees_with_new_fees' not in st.session_state:
            with st.spinner("Applying new discount structure to fee values..."):
                fees_data_tiers = st.session_state['fees_with_discounts'].copy()

                # Apply the new discount logic
                fees_data_tiers['affiliate_rebate_new'] = (
                    fees_data_tiers['position_fee_amount'] * fees_data_tiers['affiliate_gets']
                )
                fees_data_tiers['trader_rebate_new'] = (
                    fees_data_tiers['position_fee_amount'] * fees_data_tiers['trader_gets']
                )
                fees_data_tiers['total_referral_rebate_new'] = (
                    fees_data_tiers['affiliate_rebate_new'] + fees_data_tiers['trader_rebate_new']
                )
                fees_data_tiers['protocol_fee_amount_new'] = (
                    fees_data_tiers['position_fee_amount'] - fees_data_tiers['total_referral_rebate_new']
                )

                # Save to session state
                st.session_state['fees_with_new_fees'] = fees_data_tiers

                st.success("✅ New fee calculations completed.")
        else:
            st.success("✅ New fees already calculated and loaded from session state.")


# %% Section 6 — Calculate Total Fees and Protocol Fees
with st.expander("6. Calculate Total Fees and Protocol Fees", expanded=True):
    st.markdown("Final totals are computed for original vs. new fees and protocol revenue.")

    if 'fees_with_new_fees' not in st.session_state:
        st.error("❌ Please complete Section 5 first to recalculate fees.")
    else:
        with st.spinner("Calculating monthly and total fee stats..."):
            fees_data_tiers = st.session_state['fees_with_new_fees'].copy()

            # Step 1: USD conversion
            fees_data_tiers['protocol_fee_usd'] = fees_data_tiers['protocol_fee_amount'] * fees_data_tiers['collateral_token_price_avg']
            fees_data_tiers['liquidation_fee_usd'] = fees_data_tiers['liquidation_fee_amount'] * fees_data_tiers['collateral_token_price_avg']
            fees_data_tiers['borrowing_fee_usd'] = fees_data_tiers['borrowing_fee_amount'] * fees_data_tiers['collateral_token_price_avg']
            fees_data_tiers['protocol_fee_usd_new'] = fees_data_tiers['protocol_fee_amount_new'] * fees_data_tiers['collateral_token_price_avg']

            # Step 2: Total fees (old vs. new)
            fees_data_tiers['total_fee_usd'] = (
                fees_data_tiers['protocol_fee_usd'] +
                fees_data_tiers['liquidation_fee_usd'] +
                fees_data_tiers['borrowing_fee_usd']
            )
            fees_data_tiers['total_fee_usd_new'] = (
                fees_data_tiers['protocol_fee_usd_new'] +
                fees_data_tiers['liquidation_fee_usd'] +
                fees_data_tiers['borrowing_fee_usd']
            )

            # Step 3: Add month
            fees_data_tiers['month'] = pd.to_datetime(fees_data_tiers['date']).dt.to_period('M').dt.start_time

            # Step 4: Monthly group
            fees_stats = fees_data_tiers[[
                'month', 'total_fee_usd', 'total_fee_usd_new', 'protocol_fee_usd', 'protocol_fee_usd_new'
            ]]
            fees_stats_m = fees_stats.groupby('month').sum().reset_index()
            cols_to_convert = ['total_fee_usd', 'total_fee_usd_new', 'protocol_fee_usd', 'protocol_fee_usd_new']
            fees_stats_m[cols_to_convert] = fees_stats_m[cols_to_convert].astype('int64')

            # Step 5: Monthly % changes
            fees_stats_m['total_discount'] = (
                100 * (fees_stats_m['total_fee_usd_new'] / fees_stats_m['total_fee_usd'] - 1)
            ).round(2)
            fees_stats_m['protocol_discount'] = (
                100 * (fees_stats_m['protocol_fee_usd_new'] / fees_stats_m['protocol_fee_usd'] - 1)
            ).round(2)

            # Step 6: Total row
            fees_stats_all = fees_stats.iloc[:, 1:].sum().to_frame().T
            fees_stats_all[cols_to_convert] = fees_stats_all[cols_to_convert].astype('int64')
            fees_stats_all['total_discount'] = (
                100 * (fees_stats_all['total_fee_usd_new'] / fees_stats_all['total_fee_usd'] - 1)
            ).round(2)
            fees_stats_all['protocol_discount'] = (
                100 * (fees_stats_all['protocol_fee_usd_new'] / fees_stats_all['protocol_fee_usd'] - 1)
            ).round(2)
            fees_stats_all.insert(0, 'month', 'Total')

            # Step 7: Concatenate monthly and total
            fees_stats_final = pd.concat([fees_stats_m, fees_stats_all], ignore_index=True)

            # Final cleanup
            fees_stats_final.rename(columns={
                'protocol_fee_usd': 'Open/Close Fees',
                'protocol_fee_usd_new': 'Open/Close Fees (New)',
                'total_fee_usd': 'Total Fees',
                'total_fee_usd_new': 'Total Fees (New)',
                'protocol_discount': 'Open/Close Fees Change (%)',
                'total_discount': 'Total Fees Change (%)'
            }, inplace=True)

            display_cols = [
                'month',
                'Open/Close Fees', 'Open/Close Fees (New)', 'Open/Close Fees Change (%)',
                'Total Fees', 'Total Fees (New)', 'Total Fees Change (%)'
            ]

            styled_fees_table = fees_stats_final[display_cols].style.format(
                {
                    'Open/Close Fees': '${:,.0f}',
                    'Open/Close Fees (New)': '${:,.0f}',
                    'Open/Close Fees Change (%)': '{:.2f}%',
                    'Total Fees': '${:,.0f}',
                    'Total Fees (New)': '${:,.0f}',
                    'Total Fees Change (%)': '{:.2f}%'
                }
            ).set_properties(**{'text-align': 'center'}).set_table_styles([
                {"selector": "th", "props": [("text-align", "center")]}
            ])

            # ✅ Display final table
            st.markdown("### 📊 High-Level Fee Impact Summary")
            st.table(styled_fees_table)
