# %%
# import libraries
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from plotly.subplots import make_subplots

# %% functions

# Define a function for centered headers
def centered_header(text):
    st.markdown(f"<h1 style='text-align: center;'>{text}</h1>", unsafe_allow_html=True)

# # Load data
# def load_data(data_dir='./data', file_name='all_block_txs.parquet'):
#     """
#     Load transaction data from a parquet file.
    
#     Args:
#         data_dir (str): Directory where the data file is located.
#         file_name (str): Name of the parquet file to load.

#     Returns:
#         pd.DataFrame: Loaded DataFrame from the parquet file.
#     """
#     file_path = os.path.join(data_dir, file_name)

#     # Check if the data is already in session state
#     if 'all_block_txs' in st.session_state:
#         return st.session_state.all_block_txs

#     if os.path.isfile(file_path):
#         # Show a progress spinner while loading
#         with st.spinner('Loading data...'):
#             start_time = time.time()  # Start time for performance measurement
            
#             # Load data from file
#             all_block_txs = pd.read_parquet(file_path, engine='pyarrow')
#             elapsed_time = time.time() - start_time  # Calculate time taken to load data

#             # Save to session state
#             st.session_state.all_block_txs = all_block_txs

#             # Display time spent and data size
#             st.write(f"Time spent loading data: {elapsed_time:.2f} seconds")
#             st.write(f"Size of the table: {all_block_txs.shape[0]} rows, {all_block_txs.shape[1]} columns")

#             return all_block_txs
#     else:
#         st.error(f"File not found: {file_path}")
#         return None

# # get list of operators
# def create_operator_analysis_table(transaction_data):
    
#     # Set first block number and last block number
#     first_block_number = int(transaction_data['block'].min())
#     last_block_number = int(transaction_data['block'].max())
#     n_blocks = last_block_number - first_block_number + 1

#     # Perform calculations
#     select_columns = ['block', 'from_hash']
#     txs_f = transaction_data.loc[:, select_columns].drop_duplicates()
#     txs_f['value'] = 1

#     unique_blocks = np.arange(first_block_number, last_block_number + 1)
#     unique_senders = txs_f['from_hash'].unique()

#     blocks_df = pd.DataFrame(unique_blocks, columns=['block'])
#     senders_df = pd.DataFrame(unique_senders, columns=['from_hash'])

#     operator_analysis_table = pd.merge(blocks_df, senders_df, how='cross')
#     operator_analysis_table = operator_analysis_table.merge(txs_f, on=['block', 'from_hash'], how='left').fillna(0)
#     operator_analysis_table['block'] = operator_analysis_table['block'].astype(str)

#     list_of_operators = operator_analysis_table.loc[:, ['from_hash', 'value']].groupby(['from_hash']).sum().reset_index().sort_values(by=['value'], ascending=False).reset_index(drop=True)
#     list_of_operators['sent_rate'] = np.round(100 * list_of_operators['value'] / n_blocks, decimals=2)

#     return operator_analysis_table, list_of_operators

# # Function to find consecutive sequences
# def find_consecutive_sequences(blocks):
#     if not blocks:
#         return []

#     blocks.sort()
#     sequences = []
#     current_sequence = [blocks[0]]

#     for i in range(1, len(blocks)):
#         if blocks[i] == blocks[i - 1] + 1:
#             current_sequence.append(blocks[i])
#         else:
#             if len(current_sequence) > 1:
#                 sequences.append(current_sequence)
#             current_sequence = [blocks[i]]

#     if len(current_sequence) > 1:
#         sequences.append(current_sequence)

#     return sequences

# # Function to find consecutive missed blocks per operator
# def find_operator_consecutive_sequences(operators):
#     sequences_dict = {}
#     single_missed_blocks_dict = {}
    
#     for operator in operators['from_hash'].unique():
#         operator_blocks = operators.loc[operators['from_hash'] == operator, 'block'].astype(int).tolist()
#         sequences = find_consecutive_sequences(operator_blocks)
#         sequences_dict[operator] = sequences
        
#         # Add single missed blocks as sequences
#         single_missed_blocks = [block for block in operator_blocks if block not in [seq for sublist in sequences for seq in sublist]]
#         single_missed_blocks_dict[operator] = single_missed_blocks
        
#     return sequences_dict, single_missed_blocks_dict

# # Function to shorten operator names
# def shorten_operator_name(operator):
#     return operator[:4] + '...' + operator[-4:]

# # Function to create scatter plot matrix
# def plot_specific_scatter_matrix(df, numerical_features, target_features, hue_feature):
#     """
#     Create scatter plots for specific pairs of features using Plotly.

#     Parameters:
#     df (pd.DataFrame): The DataFrame containing the data.
#     numerical_features (list): List of numerical features to include in the pairplot.
#     target_features (list): List of target features for which to plot against all numerical features.
#     hue_feature (str): The column name for the hue feature (e.g., 'from_hash').

#     Returns:
#     None: Displays the plot.
#     """
    
#     # Create a mapping of hue_feature values to a color
#     unique_hues = df[hue_feature].unique()
#     hue_colors = {hue: f'rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.6)' for hue in unique_hues}
    
#     num_features = len(numerical_features)
    
#     # Set subplot size (e.g., 1000x1000)
#     fig = make_subplots(
#         rows=len(target_features), 
#         cols=num_features,
#         subplot_titles=[f'{target} vs {feature}' for target in target_features for feature in numerical_features]
#     )

#     # Add scatter plots for each target feature vs all numerical features
#     for i, target in enumerate(target_features):
#         for j, feature in enumerate(numerical_features):
#             x_data = df[feature].dropna()
#             y_data = df[target].dropna()

#             # Align the data to avoid NaN issues
#             mask = x_data.index.intersection(y_data.index)
#             x_data = x_data.loc[mask]
#             y_data = y_data.loc[mask]
#             hue_data = df[hue_feature].loc[mask]  # Get hue data based on the mask

#             # Add scatter plot for target vs feature
#             for hue in unique_hues:
#                 hue_mask = hue_data == hue
#                 fig.add_trace(go.Scatter(
#                     x=x_data[hue_mask], 
#                     y=y_data[hue_mask], 
#                     mode='markers',
#                     marker=dict(
#                         color=hue_colors[hue],  # Color based on hue
#                         opacity=0.6,
#                     ),
#                     name=str(hue),  # Add hue to legend
#                     text=df.index  # Optional hover text
#                 ), row=i+1, col=j+1)

#             # Calculate and add trend line
#             if not x_data.empty and not y_data.empty:
#                 # Reshape data for regression model
#                 X = x_data.values.reshape(-1, 1)
#                 y = y_data.values
#                 model = LinearRegression()
#                 model.fit(X, y)

#                 # Generate values for trend line
#                 x_range = np.linspace(x_data.min(), x_data.max(), 100)
#                 y_trend = model.predict(x_range.reshape(-1, 1))

#                 # Add trend line to the plot
#                 fig.add_trace(go.Scatter(
#                     x=x_range, 
#                     y=y_trend, 
#                     mode='lines', 
#                     line=dict(color='black', width=2),  # Customize trend line appearance
#                     name='Trend Line',
#                 ), row=i+1, col=j+1)

#     # Update layout to increase the size of the overall figure
#     fig.update_layout(
#         title='Scatter Plot Matrix for Selected Features',
#         height=400 * len(target_features),  # Adjust height based on number of target features
#         width=400 * num_features,  # Adjust width based on number of features
#         showlegend=False,  # Show legend
#     )

#     # Show the plot
#     return fig

