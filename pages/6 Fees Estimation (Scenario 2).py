# %% Imports
import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import plotly.express as px

from datetime import date, timedelta

from functions.functions import (
    centered_header,
    load_data,
    assign_discount,
    plot_user_distribution,
    init_key,
    render_final_tiers_table,
    assign_tiers_vectorized,
    assign_discount_advanced_percent,
    assign_discount_advanced_bps,
    get_default_pro_tiers_sum,
    get_default_pro_tiers_or,
)

# %% Page config
st.set_page_config(page_title="Analytics: Fee Estimation", layout="wide", page_icon=":money_with_wings:")

centered_header("Analytics: Fee Estimation")
st.markdown("---")

# Load Scenario 1 config from session
if "pro_tiers_config_scenario_2" not in st.session_state:
    logic = st.session_state.get("logic_applied_scenario_2", "OR")
    default_config = get_default_pro_tiers_sum() if logic == "SUM" else get_default_pro_tiers_or()
    st.session_state["pro_tiers_config_scenario_2"] = default_config.copy()
    st.session_state["pro_tiers_applied_version_scenario_2"] = "default"

config = st.session_state["pro_tiers_config_scenario_2"]
logic_label = st.session_state.get("logic_applied_scenario_2", "OR")

# %% Load data (uses session_state via load_data)
daily_trading_volume = init_key("daily_trading_volume", lambda: load_data("daily_trading_volume"))
gmx_staking = init_key("gmx_staking", lambda: load_data("gmx_staking"))
gmx_staked_last = init_key("gmx_staked_last", lambda: load_data("gmx_staked_last"))
fees_data = init_key("fees_data", lambda: load_data("fees_data"))

# %% Section 1 — Calculate Pro Tiers
with st.expander("1. Calculate Pro Tiers", expanded=False):
    st.markdown("This step defines Pro Tiers for users based on trading volume and staking amounts.")
    st.markdown(f"✅ **Applied Config:** `{st.session_state.get('pro_tiers_applied_version_scenario_2','default')}`")
    st.markdown(f"\n📌 **Condition Logic:** `{logic_label}`")

    if logic_label == "OR":
        # Step — Compute % discount vs baseline BPS
        config["discount_pos"] = 1 - (config["discount_bps_pos"] / 0.0004)
        config["discount_neg"] = 1 - (config["discount_bps_neg"] / 0.0006)
        config["discount_both_pos"] = 1 - (config["discount_both_bps_pos"] / 0.0004)
        config["discount_both_neg"] = 1 - (config["discount_both_bps_neg"] / 0.0006)
    
    render_final_tiers_table(config, logic_label=logic_label)
    recalculate = st.button("🔁 Recalculate Pro Tiers")

    if 'pro_tiers_df_vs_scenario_2' not in st.session_state or recalculate:
        refresh_warning = st.empty()
        if recalculate:
            refresh_warning.warning("🔄 All sections will be refreshed with the new config.")

        with st.spinner("Calculating Pro Tiers..."):
            # Step 1: Generate date reference
            dates_df = pd.DataFrame(pd.date_range(start='2024-12-01', end='2025-03-31', freq='D'), columns=['date'])
            dates_df['month'] = dates_df['date'].dt.to_period('M').dt.start_time
            dates_df['date'] = dates_df['date'].dt.date
            dates_df['month'] = dates_df['month'].dt.date

            # Step 2: Filter and join trading volume
            daily_trading_volume['block_date'] = pd.to_datetime(daily_trading_volume['block_date']).dt.date
            trading_volume_f = daily_trading_volume[
                (daily_trading_volume['block_date'] >= date(2024, 12, 1)) &
                (daily_trading_volume['block_date'] < date(2025, 4, 1))
            ].rename(columns={'block_date': 'date'})

            dates_df['_key'] = 1
            trading_volume_f['_key'] = 1
            pro_tiers_df_v = pd.merge(dates_df, trading_volume_f[['trader', '_key']].drop_duplicates(), on='_key').drop('_key', axis=1)
            trading_volume_f.drop('_key', axis=1, inplace=True)
            pro_tiers_df_v = pro_tiers_df_v.merge(trading_volume_f, on=['date', 'trader'], how='left')
            pro_tiers_df_v['trading_volume'].fillna(0, inplace=True)

            # Step 3: Calculate 30-day rolling volume
            pro_tiers_df_v = pro_tiers_df_v.sort_values(by=['trader', 'date']).reset_index(drop=True)
            pro_tiers_df_v['trading_volume_30d'] = (
                pro_tiers_df_v.groupby('trader')
                .rolling(30, on='date', min_periods=1)['trading_volume']
                .sum()
                .reset_index(level=0, drop=True)
                .values
            )

            # Step 4: Assign volume tier (vectorized)
            with st.spinner("Assigning Volume Tiers..."):
                volume_thresholds = config["volume_min"].tolist()
                pro_tiers_df_v["pro_tier_v"] = assign_tiers_vectorized(
                    pro_tiers_df_v["trading_volume_30d"], volume_thresholds
                )

            pro_tiers_df_v[['trading_volume_30d_sh', 'pro_tier_v_sh']] = (
                pro_tiers_df_v.groupby('trader')[['trading_volume_30d', 'pro_tier_v']].shift(1)
            )
            pro_tiers_df_v = pro_tiers_df_v[pro_tiers_df_v['date'] >= date(2025, 1, 1)].reset_index(drop=True)

            # Step 5: Handle staking
            cutoff_date = date(2025, 1, 1)
            cutoff_minus_1d = cutoff_date - timedelta(days=1)
            gmx_staking['block_date'] = pd.to_datetime(gmx_staking['block_date']).dt.date
            gmx_staking_st = (
                gmx_staking[gmx_staking['block_date'] <= cutoff_minus_1d]
                .sort_values(by=['account', 'block_date'])
                .groupby('account')
                .tail(1)
            )
            gmx_staking_st['block_date'] = cutoff_minus_1d
            gmx_staking_st2 = gmx_staking[gmx_staking['block_date'] > cutoff_minus_1d]

            gmx_staking_t = (
                pd.concat([gmx_staking_st, gmx_staking_st2])
                .sort_values(by='block_date')
                .rename(columns={'block_date': 'date', 'account': 'trader'})
                .reset_index(drop=True)
            )

            # Step 6: Merge staking and forward-fill
            pro_tiers_df_vs = pro_tiers_df_v.merge(gmx_staking_t, on=['date', 'trader'], how='left')
            pro_tiers_df_vs['staked_amount'] = (
                pro_tiers_df_vs.groupby('trader')['staked_amount']
                .ffill()
                .fillna(0)
            )

            # Step 7: Assign staking tier (vectorized)
            with st.spinner("Assigning Staking Tiers..."):
                staking_thresholds = config["staking_min"].tolist()
                pro_tiers_df_vs["pro_tier_s"] = assign_tiers_vectorized(
                    pro_tiers_df_vs["staked_amount"], staking_thresholds
                )

            # Shift staking data
            pro_tiers_df_vs[['staked_amount_sh', 'pro_tier_s_sh']] = (
                pro_tiers_df_vs.groupby('trader')[['staked_amount', 'pro_tier_s']].shift(1)
            )
            pro_tiers_df_vs['staked_amount_sh'].fillna(0,inplace=True)
            pro_tiers_df_vs['pro_tier_s_sh'].fillna(1,inplace=True)

            # Truncate to start from Jan 1
            pro_tiers_df_vs = pro_tiers_df_vs[pro_tiers_df_vs['date'] >= cutoff_date].reset_index(drop=True)

            # Step 8: Compute final discount and pro tier
            with st.spinner("Computing Discounts..."):
                if logic_label == "OR":
                    df_perc = assign_discount_advanced_percent(pro_tiers_df_vs, config)
                    df_bps = assign_discount_advanced_bps(pro_tiers_df_vs, config)
                    pro_tiers_df_vs = pd.concat([pro_tiers_df_vs, df_perc, df_bps], axis=1)
                elif logic_label == "SUM":
                    # Create numeric mapping for each tier (1-indexed)
                    discount_map = config.copy()
                    discount_map["pro_tier_v_sh"] = np.arange(1, len(config) + 1)
                    discount_map["pro_tier_s_sh"] = np.arange(1, len(config) + 1)

                    # Merge volume discounts
                    pro_tiers_df_vs = pro_tiers_df_vs.merge(
                        discount_map[["pro_tier_v_sh", "discount_volume"]],
                        on="pro_tier_v_sh", how="left"
                    )

                    # Merge staking discounts
                    pro_tiers_df_vs = pro_tiers_df_vs.merge(
                        discount_map[["pro_tier_s_sh", "discount_staking"]],
                        on="pro_tier_s_sh", how="left"
                    )

                    # Final discount = volume + staking
                    pro_tiers_df_vs["pro_discount"] = (
                        pro_tiers_df_vs["discount_volume"].fillna(0) +
                        pro_tiers_df_vs["discount_staking"].fillna(0)
                    )

                pro_tiers_df_vs['pro_tier'] = pro_tiers_df_vs[['pro_tier_v_sh', 'pro_tier_s_sh']].max(axis=1).astype(int)

            # Ensure integer types for tier columns
            for col in ['pro_tier', 'pro_tier_s', 'pro_tier_s_sh', 'pro_tier_v', 'pro_tier_v_sh']:
                pro_tiers_df_vs[col] = pro_tiers_df_vs[col].fillna(1).astype(int)

            # Step 9: Save to session state
            st.session_state['pro_tiers_df_vs_scenario_2'] = pro_tiers_df_vs
            st.success("✅ Pro Tier assignment complete.")
            refresh_warning.empty()
    else:
        st.success("✅ Pro Tier data already loaded from session state.")

    # Optional: Display computed final Pro Tiers
    with st.container():
        show_tiers_table = st.toggle("🔍 Show Final Pro Tier Assignments (sample of 500)", value=False)
        if show_tiers_table and 'pro_tiers_df_vs_scenario_2' in st.session_state:
            st.markdown("### 📋 Sample of Final Pro Tier Assignment Table")
            if logic_label == "OR":
                display_cols = [
                    'date', 'trader',
                    'trading_volume_30d','trading_volume_30d_sh',
                    'pro_tier_v', 'pro_tier_v_sh',
                    'staked_amount','staked_amount_sh',
                    'pro_tier_s','pro_tier_s_sh',
                    'pro_tier',
                    'pro_discount_pos','pro_discount_neg',
                    'pro_discount_bps_pos','pro_discount_bps_neg'
                ]
            elif logic_label == "SUM":
                display_cols = [
                    'date', 'trader',
                    'trading_volume_30d','trading_volume_30d_sh',
                    'pro_tier_v', 'pro_tier_v_sh','discount_volume',
                    'staked_amount','staked_amount_sh',
                    'pro_tier_s','pro_tier_s_sh','discount_staking',
                    'pro_tier','pro_discount'
                ]

            df = st.session_state['pro_tiers_df_vs_scenario_2'].copy()

            # Group by all tier combos and take one example from each
            stratified_sample = (
                df.groupby(['pro_tier', 'pro_tier_s_sh', 'pro_tier_v_sh'], group_keys=False)
                .apply(lambda x: x.sample(1, random_state=42))
            )

            # Fill to 500 with random sampling if needed
            if len(stratified_sample) < 500:
                additional = df.drop(stratified_sample.index).sample(
                    n=500 - len(stratified_sample), random_state=42
                )
                df_sample = pd.concat([stratified_sample, additional])
            else:
                df_sample = stratified_sample

            # Final sorting and styling
            df_sample = df_sample.sort_values(by=['date', 'trader'])
            
            styled_df = (
                df_sample[display_cols]
                .style.format({
                    'trading_volume_30d': '${:,.0f}',
                    'trading_volume_30d_sh': '${:,.0f}',
                    'discount_volume': '{:.2%}',
                    'staked_amount': '{:,.0f}',
                    'staked_amount_sh': '{:,.0f}',
                    'discount_staking': '{:.2%}',
                    'pro_discount': '{:.2%}',
                    "pro_discount_bps_pos": "{:.6f}",
                    "pro_discount_bps_neg": "{:.6f}",
                    "pro_discount_pos": "{:.2%}",
                    "pro_discount_neg": "{:.2%}",
                })
                .set_properties(**{'text-align': 'center'})
                .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
            )

            st.dataframe(styled_df, use_container_width=True)

# %% Section 2 — Fees Data with Pro Tiers
with st.expander("2. Fees Data with Pro Tiers", expanded=False):
    st.markdown("We attach Pro Tier data to raw fee transactions to prepare for fee impact analysis.")

    if 'fees_data_tiers_scenario_2' not in st.session_state or recalculate:
        with st.spinner("Merging Pro Tier data with fees data..."):
            if 'pro_tiers_df_vs_scenario_2' not in st.session_state:
                st.error("❌ Please run Section 1 first to calculate Pro Tiers.")
            else:
                # Step 1: Preprocess fees_data
                fees_data['block_date'] = pd.to_datetime(fees_data['block_date']).dt.date
                fees_data = fees_data.rename(columns={'block_date': 'date', 'account': 'trader'})

                # Step 2: Filter date range and position_fee_factor
                fees_data = fees_data[
                    (fees_data['date'] >= date(2025, 1, 1)) &
                    (fees_data['date'] < date(2025, 4, 1))
                ].reset_index(drop=True)
                fees_data = fees_data[
                    (fees_data['position_fee_factor'].isin([0.0004,0.0006]))
                ].reset_index(drop=True)

                # Step 3: Merge with Pro Tiers
                pro_tiers_df_vs = st.session_state['pro_tiers_df_vs_scenario_2']

                if logic_label == "OR":
                    fees_data_tiers = fees_data.merge(
                        pro_tiers_df_vs[['date', 'trader', 'trading_volume_30d_sh', 'staked_amount_sh',
                                        'pro_tier', 'pro_tier_s_sh', 'pro_tier_v_sh', 
                                        'pro_discount_pos','pro_discount_neg','pro_discount_bps_pos','pro_discount_bps_neg'
                                        ]],
                        on=['date', 'trader'],
                        how='left'
                    )
                elif logic_label == "SUM":
                    fees_data_tiers = fees_data.merge(
                        pro_tiers_df_vs[['date', 'trader', 'trading_volume_30d_sh', 'staked_amount_sh',
                                        'pro_tier', 'pro_tier_s_sh', 'pro_tier_v_sh', 
                                        'discount_volume','discount_staking','pro_discount']],
                        on=['date', 'trader'],
                        how='left'
                    )

                # Step 4: Fill missing Pro Tier values
                fees_data_tiers['trading_volume_30d_sh'] = fees_data_tiers['trading_volume_30d_sh'].fillna(0)
                fees_data_tiers['staked_amount_sh'] = fees_data_tiers['staked_amount_sh'].fillna(0)
                fees_data_tiers['pro_tier'] = fees_data_tiers['pro_tier'].fillna(1).astype(int)
                fees_data_tiers['pro_tier_s_sh'] = fees_data_tiers['pro_tier_s_sh'].fillna(1).astype(int)
                fees_data_tiers['pro_tier_v_sh'] = fees_data_tiers['pro_tier_v_sh'].fillna(1).astype(int)
                

                if logic_label == "SUM":
                    fees_data_tiers['discount_volume'] = fees_data_tiers['discount_volume'].fillna(0)
                    fees_data_tiers['discount_staking'] = fees_data_tiers['discount_staking'].fillna(0)
                    fees_data_tiers['pro_discount'] = fees_data_tiers['pro_discount'].fillna(0)
                elif logic_label == "OR":
                    fees_data_tiers['pro_discount_pos'] = fees_data_tiers['pro_discount_pos'].fillna(0)
                    fees_data_tiers['pro_discount_neg'] = fees_data_tiers['pro_discount_neg'].fillna(0)
                    fees_data_tiers['pro_discount_bps_pos'] = fees_data_tiers['pro_discount_bps_pos'].fillna(0.0004)
                    fees_data_tiers['pro_discount_bps_neg'] = fees_data_tiers['pro_discount_bps_neg'].fillna(0.0006)

                    fees_data_tiers['pro_discount'] = np.where(
                        fees_data_tiers['pi_sign'], 
                        fees_data_tiers['pro_discount_pos'], 
                        fees_data_tiers['pro_discount_neg'])
                    fees_data_tiers['pro_discount_bps'] = np.where(
                        fees_data_tiers['pi_sign'], 
                        fees_data_tiers['pro_discount_bps_pos'], 
                        fees_data_tiers['pro_discount_bps_neg'])

                # Step 5: Save result
                st.session_state['fees_data_tiers_scenario_2'] = fees_data_tiers
                st.success("✅ Fees data successfully enriched with Pro Tier info.")
    else:
        st.success("✅ Fees data with Pro Tiers already loaded from session state.")

    # Optional: Display a sample of the merged data
    with st.container():
        show_fees_table = st.toggle("🔍 Show Fees Data Sample (stratified)", value=False)
        if show_fees_table and 'fees_data_tiers_scenario_2' in st.session_state:
            st.markdown("### 📋 Sample of Fees Data with Pro Tier Info")

            df = st.session_state['fees_data_tiers_scenario_2'].copy()

            # Stratified sample: all (tier combo) groups, at least one
            stratified_sample = (
                df.groupby(['pro_tier', 'pro_tier_s_sh', 'pro_tier_v_sh'], group_keys=False)
                .apply(lambda x: x.sample(1, random_state=42))
            )

            if len(stratified_sample) < 500:
                extra = df.drop(stratified_sample.index).sample(n=500 - len(stratified_sample), random_state=42)
                df_sample = pd.concat([stratified_sample, extra])
            else:
                df_sample = stratified_sample

            df_sample = df_sample.sort_values(by=['date', 'trader'])

            styled_fees_df = (
                df_sample
                .style.format({
                    'collateral_token_price_avg': '${:,.4f}',
                    'referral_total_rebate_factor': '{:.2%}',
                    'referral_trader_discount_factor': '{:.2%}',
                    'referral_adjusted_affiliate_reward_factor': '{:.2%}',
                    'trading_volume_30d_sh': '${:,.0f}',
                    'staked_amount_sh': '{:,.0f}',
                    'pro_tier': '{:,.0f}',
                    'pro_tier_s_sh': '{:,.0f}',
                    'pro_tier_v_sh': '{:,.0f}',
                    'discount_volume': '{:.2%}',
                    'discount_staking': '{:.2%}',
                    'pro_discount': '{:.2%}',
                    'pro_discount_bps': "{:.6f}",
                    "pro_discount_pos": "{:.2%}",
                    "pro_discount_neg": "{:.2%}",
                    "pro_discount_bps_pos": "{:.6f}",
                    "pro_discount_bps_neg": "{:.6f}",
                })
                .set_properties(**{'text-align': 'center'})
                .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
            )

            st.dataframe(styled_fees_df, use_container_width=True)

# %% Section 3 — User Distribution Across Categories
with st.expander("3. User Distribution Across Categories", expanded=False):
    st.markdown("We analyze the overlap of volume- and staking-based qualifications across Pro Tiers.")

    if 'tiers_stats_scenario_2' not in st.session_state or recalculate:
        with st.spinner("Computing user category splits by Pro Tier..."):
            pro_tiers_df_vs = st.session_state['pro_tiers_df_vs_scenario_2']
            tiers_stats_time = pro_tiers_df_vs.copy()

            n_tiers = len(config)
            if logic_label == 'OR':
                tiers_stats_time['both'] = False
                tiers_stats_time['only_volume'] = False
                tiers_stats_time['only_staking'] = False

                for tier_n in np.arange(1, n_tiers + 1):
                    tier_df = tiers_stats_time[tiers_stats_time['pro_tier'] == tier_n]
                    idx = tier_df.index
                    vol_cond = tier_df['pro_tier_v_sh'] == tier_n
                    stk_cond = tier_df['pro_tier_s_sh'] == tier_n

                    tiers_stats_time.loc[idx, 'both'] = vol_cond & stk_cond
                    tiers_stats_time.loc[idx, 'only_volume'] = vol_cond & ~stk_cond
                    tiers_stats_time.loc[idx, 'only_staking'] = ~vol_cond & stk_cond

                cols_to_select = ['pro_tier', 'pro_tier_s_sh', 'pro_tier_v_sh','both', 'only_volume', 'only_staking']
            elif logic_label == 'SUM':
                tiers_stats_time['only_volume'] = True
                tiers_stats_time['only_staking'] = True
                cols_to_select = ['pro_tier', 'pro_tier_s_sh', 'pro_tier_v_sh','only_volume', 'only_staking']

            # Save sampled pre-group table
            pre_group_sample_df = tiers_stats_time.copy()
            stratified_sample = (
                pre_group_sample_df.groupby(cols_to_select, group_keys=False)
                .apply(lambda x: x.sample(1, random_state=42))
            )
            if len(stratified_sample) < 500:
                extra = pre_group_sample_df.drop(stratified_sample.index).sample(
                    n=500 - len(stratified_sample), random_state=42
                )
                df_sample = pd.concat([stratified_sample, extra])
            else:
                df_sample = stratified_sample
            df_sample = df_sample.sort_values(by=['date', 'trader'])
            st.session_state['user_split_sample_scenario_2'] = df_sample

            # Grouping for charts
            if logic_label == 'OR':
                tiers_stats_time = (
                    tiers_stats_time[['date', 'pro_tier', 'both', 'only_volume', 'only_staking']]
                    .groupby(['date', 'pro_tier'])
                    .sum()
                    .reset_index()
                )

            elif logic_label == 'SUM':

                # Group by date and tier for staking and volume independently
                staking_counts = (
                    tiers_stats_time[['date', 'pro_tier_s_sh', 'only_staking']]
                    .groupby(['date', 'pro_tier_s_sh'])
                    .sum()
                    .reset_index()
                    .rename(columns={'pro_tier_s_sh': 'pro_tier'})
                )
                volume_counts = (
                    tiers_stats_time[['date', 'pro_tier_v_sh', 'only_volume']]
                    .groupby(['date', 'pro_tier_v_sh'])
                    .sum()
                    .reset_index()
                    .rename(columns={'pro_tier_v_sh': 'pro_tier'})     
                )
                # Merge both
                tiers_stats_time = pd.merge(staking_counts, volume_counts, on=['date', 'pro_tier'], how='outer').fillna(0)

            # Add percentage columns
            total_traders = pro_tiers_df_vs['trader'].nunique()
            tiers_stats_time_2 = (100 * tiers_stats_time.iloc[:, 2:] / total_traders).round(2)
            tiers_stats_time_2.columns = [f"{col}_perc" for col in tiers_stats_time_2.columns]

            tiers_stats = pd.concat([tiers_stats_time, tiers_stats_time_2], axis=1)
            tiers_stats = tiers_stats.sort_values(by=['date'], ascending=True).reset_index(drop=True)
            tiers_stats_snapshot = tiers_stats.groupby('pro_tier').tail(1)
            tiers_stats_snapshot = tiers_stats_snapshot.sort_values(by=['pro_tier'], ascending=True).reset_index(drop=True)

            st.session_state['tiers_stats_scenario_2'] = tiers_stats
            st.session_state['tiers_stats_snapshot_scenario_2'] = tiers_stats_snapshot

            st.success("✅ Distribution stats calculated.")
    else:
        tiers_stats = st.session_state['tiers_stats_scenario_2']
        tiers_stats_snapshot = st.session_state['tiers_stats_snapshot_scenario_2']
        st.success("✅ Distribution stats loaded from session state.")

    # 🔍 Optional Preview of Pre-Grouped Sample
    with st.container():
        show_user_split_sample = st.toggle("🔍 Show User-Level Split Sample (before grouping data)", value=False)
        if show_user_split_sample and 'user_split_sample_scenario_2' in st.session_state:
            st.markdown("### 📋 Sample of User Classification by Volume vs Staking")

            styled_sample = (
                st.session_state['user_split_sample_scenario_2']
                .style.format({
                    'collateral_token_price_avg': '${:,.4f}',
                    'referral_total_rebate_factor': '{:.2%}',
                    'referral_trader_discount_factor': '{:.2%}',
                    'referral_adjusted_affiliate_reward_factor': '{:.2%}',
                    'trading_volume_30d_sh': '${:,.0f}',
                    'staked_amount': '{:,.0f}',
                    'pro_tier': '{:,.0f}',
                    'pro_tier_s_sh': '{:,.0f}',
                    'pro_tier_v_sh': '{:,.0f}',
                    'discount_volume': '{:.2%}',
                    'discount_staking': '{:.2%}',
                    'pro_discount': '{:.2%}',
                    'pro_discount_bps': "{:.6f}",
                    "pro_discount_pos": "{:.2%}",
                    "pro_discount_neg": "{:.2%}",
                    "pro_discount_bps_pos": "{:.6f}",
                    "pro_discount_bps_neg": "{:.6f}",
                })
                .set_properties(**{'text-align': 'center'})
                .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
            )
            st.dataframe(styled_sample, use_container_width=True)

    with st.container():
        st.markdown("### 📋 User Distribution Across Pro Categories (Snapshot - March 31, 2025)")

        snapshot = tiers_stats_snapshot.copy()

        if logic_label == "OR":
            snapshot['Both (Volume + Staking)'] = snapshot.apply(
                lambda row: f"{row['both']:,.0f} ({row['both_perc']}%)", axis=1)
            snapshot['Only Volume'] = snapshot.apply(
                lambda row: f"{row['only_volume']:,.0f} ({row['only_volume_perc']}%)", axis=1)
            snapshot['Only Staking'] = snapshot.apply(
                lambda row: f"{row['only_staking']:,.0f} ({row['only_staking_perc']}%)", axis=1)
            snapshot = snapshot.rename(columns={'date':'Date'})

            display_df = snapshot[['pro_tier', 'Date', 'Both (Volume + Staking)', 'Only Volume', 'Only Staking']].copy()
            display_df['pro_tier'] = display_df['pro_tier'].apply(lambda x: f"Pro {x}")
            display_df = display_df.rename(columns={'pro_tier': 'Pro Tier'})
            display_df = display_df.set_index('Pro Tier')
        elif logic_label == "SUM":
            snapshot['Only Volume'] = snapshot.apply(
                lambda row: f"{row['only_volume']:,.0f} ({row['only_volume_perc']}%)", axis=1)
            snapshot['Only Staking'] = snapshot.apply(
                lambda row: f"{row['only_staking']:,.0f} ({row['only_staking_perc']}%)", axis=1)
            snapshot = snapshot.rename(columns={'date':'Date'})

            display_df = snapshot[['pro_tier', 'Date', 'Only Volume', 'Only Staking']].copy()
            display_df['pro_tier'] = display_df['pro_tier'].apply(lambda x: f"Pro {x}")
            display_df = display_df.rename(columns={'pro_tier': 'Pro Tier'})
            display_df = display_df.set_index('Pro Tier')

        styled_table = display_df.style.set_properties(**{
            'text-align': 'center'
        }).set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]}
        ])

        st.table(styled_table)

        # Charts
        st.markdown("### 📈 Line Charts: Absolute & Relative Distributions")

        if logic_label == "OR":
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

    if 'fees_data_tiers_scenario_2' not in st.session_state:
        st.error("❌ Please complete Section 2 first (Fees Data with Pro Tiers).")
    else:
        if 'fees_with_discounts_scenario_2' not in st.session_state or recalculate:
            with st.spinner("Calculating combined referral and pro tier rewards..."):
                fees_data_tiers = st.session_state['fees_data_tiers_scenario_2'].copy()

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

                if logic_label == "OR":
                    fees_data_tiers['pro_discount_higher'] = fees_data_tiers['pro_discount'] > fees_data_tiers['trader_rebate']

                st.session_state['fees_with_discounts_scenario_2'] = fees_data_tiers
                st.success("✅ Final discount calculations complete.")
        else:
            st.success("✅ Discounts already calculated and loaded from session state.")

    # 🔍 Optional Preview of Sample
    with st.container():
        show_discount_sample = st.toggle("🔍 Show Sample of Final Referral + Pro Tier Discounts", value=False)
        if show_discount_sample and 'fees_with_discounts_scenario_2' in st.session_state:
            st.markdown("### 📋 Sample of Combined Discount Calculation")

            df = st.session_state['fees_with_discounts_scenario_2'].copy()

            stratified_sample = (
                df.groupby(['pro_tier', 'pro_tier_s_sh', 'pro_tier_v_sh',
                            'affiliate_gets','trader_gets'], group_keys=False)
                .apply(lambda x: x.sample(1, random_state=42))
            )

            if len(stratified_sample) < 500:
                extra = df.drop(stratified_sample.index).sample(
                    n=500 - len(stratified_sample), random_state=42
                )
                df_sample = pd.concat([stratified_sample, extra])
            else:
                df_sample = stratified_sample

            df_sample = df_sample.sort_values(by=['date', 'trader'])

            styled_sample = (
                df_sample
                .style.format({
                    'collateral_token_price_avg': '${:,.4f}',
                    'total_referral_rebate': '{:.2%}',
                    'trader_rebate': '{:.2%}',
                    'affiliate_rebate': '{:.2%}',
                    'trading_volume_30d_sh': '${:,.0f}',
                    'staked_amount': '{:,.0f}',
                    'pro_tier': '{:,.0f}',
                    'pro_tier_s_sh': '{:,.0f}',
                    'pro_tier_v_sh': '{:,.0f}',
                    'discount_volume': '{:.2%}',
                    'discount_staking': '{:.2%}',
                    'pro_discount': '{:.2%}',
                    'pro_discount_bps': "{:.6f}",
                    "pro_discount_pos": "{:.2%}",
                    "pro_discount_neg": "{:.2%}",
                    "pro_discount_bps_pos": "{:.6f}",
                    "pro_discount_bps_neg": "{:.6f}",
                    'min_affiliate_reward': '{:.2%}',
                    'affiliate_reward_cap': '{:.2%}',
                    'affiliate_gets': '{:.2%}',
                    'trader_gets': '{:.2%}'
                })
                .set_properties(**{'text-align': 'center'})
                .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
            )
            st.dataframe(styled_sample, use_container_width=True)

# %% Section 5 — Recalculate Fees
with st.expander("5. Recalculate Fees", expanded=False):
    st.markdown("We apply new discounts to compute updated fees for open/close trades.")

    if 'fees_with_discounts_scenario_2' not in st.session_state:
        st.error("❌ Please complete Section 4 first to calculate discounts.")
    else:
        if 'fees_with_new_fees_scenario_2' not in st.session_state or recalculate:

            with st.spinner("Applying new discount structure to fee values..."):
                fees_data_tiers = st.session_state['fees_with_discounts_scenario_2'].copy()

                if logic_label == "OR":

                    # Calculate new position_fee_amount using proportional rule where pro_discount is applied
                    fees_data_tiers['position_fee_amount_new'] = np.where(
                        fees_data_tiers['pro_discount_higher'],
                        (fees_data_tiers['pro_discount_bps'] * fees_data_tiers['position_fee_amount']) / fees_data_tiers['position_fee_factor'],
                        fees_data_tiers['position_fee_amount']
                    )
                    
                    # Apply the new discount logic
                    fees_data_tiers['affiliate_rebate_new'] = (
                        fees_data_tiers['position_fee_amount_new'] * fees_data_tiers['affiliate_gets']
                    )
                    fees_data_tiers['trader_rebate_new'] = np.where(
                        fees_data_tiers['pro_discount_higher'],
                        0,
                        fees_data_tiers['position_fee_amount'] * fees_data_tiers['trader_gets']
                    )
                    fees_data_tiers['total_referral_rebate_new'] = (
                        fees_data_tiers['affiliate_rebate_new'] + fees_data_tiers['trader_rebate_new']
                    )
                    fees_data_tiers['protocol_fee_amount_new'] = (
                        fees_data_tiers['position_fee_amount_new'] - fees_data_tiers['total_referral_rebate_new']
                    )
                elif logic_label == "SUM":
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
                st.session_state['fees_with_new_fees_scenario_2'] = fees_data_tiers

                st.success("✅ New fee calculations completed.")
        else:
            st.success("✅ New fees already calculated and loaded from session state.")

    # 🔍 Optional: Sample of Fee Recalculation Results
    with st.container():
        show_new_fees_sample = st.toggle("🔍 Show Sample of New Fee Calculations", value=False)
        if show_new_fees_sample and 'fees_with_new_fees_scenario_2' in st.session_state:
            st.markdown("### 📋 Sample of New Fee Recalculation Results")

            df = st.session_state['fees_with_new_fees_scenario_2'].copy()

            stratified_sample = (
                df.groupby(['pro_tier', 'pro_tier_s_sh', 'pro_tier_v_sh'], group_keys=False)
                .apply(lambda x: x.sample(1, random_state=42))
            )

            if len(stratified_sample) < 500:
                extra = df.drop(stratified_sample.index).sample(
                    n=500 - len(stratified_sample), random_state=42
                )
                df_sample = pd.concat([stratified_sample, extra])
            else:
                df_sample = stratified_sample

            df_sample = df_sample.sort_values(by=['date', 'trader'])

            styled_sample = (
                df_sample
                .style.format({
                    'collateral_token_price_avg': '${:,.4f}',
                    'total_referral_rebate': '{:.2%}',
                    'trader_rebate': '{:.2%}',
                    'affiliate_rebate': '{:.2%}',
                    'trading_volume_30d_sh': '${:,.0f}',
                    'staked_amount': '{:,.0f}',
                    'pro_tier': '{:,.0f}',
                    'pro_tier_s_sh': '{:,.0f}',
                    'pro_tier_v_sh': '{:,.0f}',
                    'discount_volume': '{:.2%}',
                    'discount_staking': '{:.2%}',
                    'pro_discount': '{:.2%}',
                    'pro_discount_bps': "{:.6f}",
                    "pro_discount_pos": "{:.2%}",
                    "pro_discount_neg": "{:.2%}",
                    "pro_discount_bps_pos": "{:.6f}",
                    "pro_discount_bps_neg": "{:.6f}",
                    'min_affiliate_reward': '{:.2%}',
                    'affiliate_reward_cap': '{:.2%}',
                    'affiliate_gets': '{:.2%}',
                    'trader_gets': '{:.2%}'
                })
                .set_properties(**{'text-align': 'center'})
                .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
            )
            st.dataframe(styled_sample, use_container_width=True)

# %% Section 6 — Calculate Total Fees and Protocol Fees
with st.expander("6. Calculate Total Fees and Protocol Fees", expanded=True):
    st.markdown("Final totals are computed for original vs. new fees and protocol revenue.")

    if 'fees_with_new_fees_scenario_2' not in st.session_state:
        st.error("❌ Please complete Section 5 first to recalculate fees.")
    else:
        with st.spinner("Calculating monthly and total fee stats..."):
            fees_data_tiers = st.session_state['fees_with_new_fees_scenario_2'].copy()

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
                'month':'Month',
                'protocol_fee_usd': 'Open/Close Fees',
                'protocol_fee_usd_new': 'Open/Close Fees (New)',
                'total_fee_usd': 'Total Fees',
                'total_fee_usd_new': 'Total Fees (New)',
                'protocol_discount': 'Open/Close Fees Change (%)',
                'total_discount': 'Total Fees Change (%)'
            }, inplace=True)

            display_cols = [
                'Month',
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


        # 📊 Optional chart for monthly fee impact visualization
        show_table = st.toggle("📈 Show Monthly Fees By Tiers", value=False)
        if show_table:
            # Monthly Fees Stats By Tier
            # Monthly group
            fees_stats_tiers = fees_data_tiers[[
                'month', 'pro_tier', 'total_fee_usd', 'total_fee_usd_new', 'protocol_fee_usd', 'protocol_fee_usd_new'
            ]]
            fees_stats_tiers_m = fees_stats_tiers.groupby(['month','pro_tier']).sum().reset_index()
            fees_stats_tiers_m[cols_to_convert] = fees_stats_tiers_m[cols_to_convert].astype('int64')
            
            # Monthly % changes
            fees_stats_tiers_m['total_discount'] = (
                100 * (fees_stats_tiers_m['total_fee_usd_new'] / fees_stats_tiers_m['total_fee_usd'] - 1)
            ).round(2)
            fees_stats_tiers_m['protocol_discount'] = (
                100 * (fees_stats_tiers_m['protocol_fee_usd_new'] / fees_stats_tiers_m['protocol_fee_usd'] - 1)
            ).round(2)

            # Total row
            fees_stats_tiers_all = fees_stats_tiers.iloc[:, 2:].sum().to_frame().T
            fees_stats_tiers_all[cols_to_convert] = fees_stats_tiers_all[cols_to_convert].astype('int64')
            fees_stats_tiers_all['total_discount'] = (
                100 * (fees_stats_tiers_all['total_fee_usd_new'] / fees_stats_tiers_all['total_fee_usd'] - 1)
            ).round(2)
            fees_stats_tiers_all['protocol_discount'] = (
                100 * (fees_stats_tiers_all['protocol_fee_usd_new'] / fees_stats_tiers_all['protocol_fee_usd'] - 1)
            ).round(2)
            fees_stats_tiers_all.insert(0, 'pro_tier', 'All')
            fees_stats_tiers_all.insert(0, 'month', 'Total')

            # Concatenate monthly and total
            fees_stats_tiers_final = pd.concat([fees_stats_tiers_m, fees_stats_tiers_all], ignore_index=True)

            # Final cleanup
            fees_stats_tiers_final.rename(columns={
                'month':'Month',
                'pro_tier':'Pro Tier',
                'protocol_fee_usd': 'Open/Close Fees',
                'protocol_fee_usd_new': 'Open/Close Fees (New)',
                'total_fee_usd': 'Total Fees',
                'total_fee_usd_new': 'Total Fees (New)',
                'protocol_discount': 'Open/Close Fees Change (%)',
                'total_discount': 'Total Fees Change (%)'
            }, inplace=True)

            display_cols_tiers = [
                'Month',
                'Pro Tier',
                'Open/Close Fees', 'Open/Close Fees (New)', 'Open/Close Fees Change (%)',
                'Total Fees', 'Total Fees (New)', 'Total Fees Change (%)'
            ]

            styled_fees_table_tiers = fees_stats_tiers_final[display_cols_tiers].style.format(
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

            # # ✅ Display final table
            # st.markdown("### 📊 High-Level Fee Impact Summary")
            st.table(styled_fees_table_tiers)


        # 📊 Optional chart for monthly fee impact visualization
        show_chart = st.toggle("📈 Show Monthly Fee Impact Charts", value=False)
        if show_chart:
            fees_stats_final = fees_stats_final.copy()
            chart_data = fees_stats_final[fees_stats_final['Month'] != 'Total'].copy()

            # ✅ Create proper string month for X-axis
            chart_data['month_str'] = pd.to_datetime(chart_data['Month']).dt.strftime('%b %Y')

            # ✅ Total Fees Chart
            fig_total = px.bar(
                chart_data,
                x='month_str',
                y=['Total Fees', 'Total Fees (New)'],
                barmode='group',
                labels={'value': 'USD', 'month_str': 'Month', 'variable': 'Fee Type'},
                title="📊 Monthly Total Fees: Original vs. New",
                text_auto=True
            )
            fig_total.update_traces(texttemplate='$%{y:,.0f}', textposition='inside')
            fig_total.update_layout(yaxis_title='USD', xaxis_title='Month')
            st.plotly_chart(fig_total, use_container_width=True)

            # ✅ Open/Close Fees Chart
            fig_open_close = px.bar(
                chart_data,
                x='month_str',
                y=['Open/Close Fees', 'Open/Close Fees (New)'],
                barmode='group',
                labels={'value': 'USD', 'month_str': 'Month', 'variable': 'Fee Type'},
                title="📊 Monthly Open/Close Fees: Original vs. New",
                text_auto=True
            )
            fig_open_close.update_traces(texttemplate='$%{y:,.0f}', textposition='inside')
            fig_open_close.update_layout(yaxis_title='USD', xaxis_title='Month')
            st.plotly_chart(fig_open_close, use_container_width=True)


# %%
