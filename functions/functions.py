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

# Load data
def load_data(file_name, data_dir='./data'):
    """
    Load transaction data from a parquet file.
    
    Args:
        data_dir (str): Directory where the data file is located.
        file_name (str): Name of the parquet file to load.

    Returns:
        pd.DataFrame: Loaded DataFrame from the parquet file.
    """
    file_path = os.path.join(data_dir, f"{file_name}.csv")

    # Check if the data is already in session state
    if file_name in st.session_state:
        return st.session_state[file_name]

    if os.path.isfile(file_path):
        # Show a progress spinner while loading
        with st.spinner('Loading data...'):
            start_time = time.time()  # Start time for performance measurement
            
            # Load data from file
            dataset = pd.read_csv(file_path, engine='pyarrow')
            elapsed_time = time.time() - start_time  # Calculate time taken to load data

            # Save to session state
            st.session_state[file_name] = dataset

            # # Display time spent and data size
            # st.write(f"Time spent loading data '{file_name}': {elapsed_time:.2f} seconds")
            # st.write(f"Size of the table '{file_name}': {dataset.shape[0]:,} rows, {dataset.shape[1]:,} columns")

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
        title_x=0.4,
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
        title_x=0.4,
        xaxis_title='Staked Amount Segment',
        yaxis_title='Total Staked Amount',
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    # 5. Show plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
