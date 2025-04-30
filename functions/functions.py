# %%
# import libraries
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# %% functions

# Define a function for centered headers
def centered_header(text):
    st.markdown(f"<h1 style='text-align: center;'>{text}</h1>", unsafe_allow_html=True)

# Load data (supports split chunks for large files)
def load_data(file_name, data_dir='./data'):
    """
    Load CSV data or split-chunked CSVs if file_name is 'fees_data' or 'gmx_staking'.

    Args:
        file_name (str): Base file name or special name for chunked folders.
        data_dir (str): Root directory of the data files or chunk folders.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if file_name in st.session_state:
        return st.session_state[file_name]

    # Map specific chunked files to their folder names
    chunk_folder_map = {
        'fees_data': 'fees',
        'gmx_staking': 'staking'
    }

    if file_name in chunk_folder_map:
        folder_path = os.path.join(data_dir, chunk_folder_map[file_name])
        if not os.path.isdir(folder_path):
            st.error(f"Chunk folder not found: {folder_path}")
            return None

        with st.spinner(f"Loading chunked {file_name.replace('_', ' ')}..."):
            chunks = []
            for file in sorted(os.listdir(folder_path)):
                if file.endswith('.csv'):
                    chunk_path = os.path.join(folder_path, file)
                    chunk = pd.read_csv(chunk_path)
                    chunks.append(chunk)

            if not chunks:
                st.error(f"No CSV chunks found in: {folder_path}")
                return None

            dataset = pd.concat(chunks, ignore_index=True)
            st.session_state[file_name] = dataset
            return dataset

    # Fallback: single file loading
    file_path = os.path.join(data_dir, f"{file_name}.csv")
    if os.path.isfile(file_path):
        with st.spinner(f"Loading {file_name}.csv..."):
            dataset = pd.read_csv(file_path)
            st.session_state[file_name] = dataset
            return dataset
    else:
        st.error(f"File not found: {file_path}")
        return None

def plot_trading_volume_distribution(daily_trading_volume: pd.DataFrame):
    """
    Plots the distribution of total trading volume by segment and trader count over the last 30 days.
    """

    # 1. Calculate 30-day window
    today = datetime.today()
    thirty_days_ago = today - timedelta(days=30)

    # 2. Ensure 'block_date' is datetime
    daily_trading_volume['block_date'] = pd.to_datetime(daily_trading_volume['block_date'])

    # 3. Filter for last 30 days
    trading_volume_30d = daily_trading_volume.loc[daily_trading_volume['block_date'].dt.date >= thirty_days_ago.date(), :]
    trading_volume_30d = trading_volume_30d[['trader', 'trading_volume']].groupby('trader').sum().reset_index()

    # 4. Define volume buckets
    bins = [0, 1_000_000, 2_000_000, 3_000_000, 5_000_000, 6_000_000,
            10_000_000, 20_000_000, 50_000_000, 100_000_000,
            200_000_000, 1_000_000_000, 4_000_000_000]
    labels = [
        '0–1M', '1M–2M', '2M–3M', '3M–5M', '5M–6M', '6M–10M',
        '10M–20M', '20M–50M', '50M–100M', '100M–200M',
        '200M–1B', '1B–4B'
    ]

    # 5. Categorize by segment
    trading_volume_30d['volume_segment'] = pd.cut(
        trading_volume_30d['trading_volume'],
        bins=bins,
        labels=labels,
        right=False
    )

    # 6. Aggregate
    trader_counts = trading_volume_30d.groupby('volume_segment')['trader'].count().reset_index(name='trader_count')
    total_volume = trading_volume_30d.groupby('volume_segment')['trading_volume'].sum().reset_index(name='total_volume')

    summary = trader_counts.merge(total_volume, on='volume_segment')

    # 7. Plot
    fig = px.bar(
        summary,
        x='volume_segment',
        y='total_volume',
        text='trader_count',
        labels={'volume_segment': 'Volume Segment', 'total_volume': 'Total Trading Volume'},
        title='Distribution of Total Trading Volume by Segment and Count of Traders (30D)',
        height=700
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        title_x=0,
        xaxis_title='Segment',
        yaxis_title='Last 30-day Trading Volume / Total Count',
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    # 8. Show plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def plot_staked_amount_distribution(gmx_staked_last: pd.DataFrame):
    """
    Plots the distribution of total staked GMX amounts by segment and staker count.
    """

    # 1. Define staking buckets
    bins = [0, 1, 100, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, float('inf')]
    labels = [
        '<1 GMX', '1-100 GMX', '100-500 GMX', '500-1k GMX', '1k-2k GMX',
        '2k-5k GMX', '5k-10k GMX', '10k–20k GMX', '20k–50k GMX', 
        '50k–100k GMX', '>100k GMX'
    ]

    # 2. Categorize each account
    gmx_staked_last['staked_segment'] = pd.cut(
        gmx_staked_last['staked_amount'],
        bins=bins,
        labels=labels,
        right=False
    )

    # 3. Group by staking segment
    staker_counts = gmx_staked_last.groupby('staked_segment')['account'].count().reset_index(name='staker_count')
    total_staked = gmx_staked_last.groupby('staked_segment')['staked_amount'].sum().reset_index(name='total_staked')

    # Merge counts and total staked amounts
    staking_summary = staker_counts.merge(total_staked, on='staked_segment')

    # 4. Plot
    fig = px.bar(
        staking_summary,
        x='staked_segment',
        y='total_staked',
        text='staker_count',
        labels={'staked_segment': 'Staked Amount Segment', 'total_staked': 'Total Staked Amount'},
        title='Distribution of Total Staked Amount by Segment and Count of Stakers',
        height=700
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        title_x=0,
        xaxis_title='Staked Amount Segment',
        yaxis_title='Total Staked Amount',
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    # 5. Show plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Define combined discount
def assign_discount(row):
    v, s = row['pro_tier_v_sh'], row['pro_tier_s']
    if v == 4 and s == 4: return 0.35
    if v == 3 and s == 3: return 0.25
    if v == 2 and s == 2: return 0.15
    if v == 1 and s == 1: return 0
    if v == 4 or s == 4: return 0.30
    if v == 3 or s == 3: return 0.20
    if v == 2 or s == 2: return 0.10
    return 0

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

def plot_user_distribution(df: pd.DataFrame, value_type: str = 'absolute', category: str = 'both'):
    """
    Plots user distribution (absolute or relative) for a selected category (both, only_volume, or only_staking),
    split by pro_tier. Y-axis is log-scaled.

    Parameters:
        df (pd.DataFrame): Input dataframe with user counts and percentages.
        value_type (str): 'absolute' or 'relative'
        category (str): one of 'both', 'only_volume', 'only_staking'
    """
    assert value_type in ['absolute', 'relative'], "value_type must be 'absolute' or 'relative'"
    assert category in ['both', 'only_volume', 'only_staking'], "category must be one of 'both', 'only_volume', 'only_staking'"

    y_col = category if value_type == 'absolute' else f"{category}_perc"
    y_title = "Number of Users (log scale)" if value_type == 'absolute' else "Percentage of Users (%)"

    fig = go.Figure()

    color_map = {
        1: 'blue',
        2: 'green',
        3: 'orange',
        4: 'red'
    }

    for tier in sorted(df['pro_tier'].unique()):
        sub = df[df['pro_tier'] == tier]
        fig.add_trace(go.Scatter(
            x=sub['date'],
            y=sub[y_col],
            mode='lines+markers',
            name=f'Pro {tier}',
            line=dict(color=color_map.get(tier, None))
        ))

    fig.update_layout(
        title=f"User Distribution - {category.replace('_', ' ').title()} ({value_type.title()})",
        xaxis_title='Date',
        yaxis_title=y_title,
        height=700,
        yaxis_type='log' if value_type == 'absolute' else 'linear',
        legend_title="Pro Tier"
    )

    st.plotly_chart(fig, use_container_width=True)
