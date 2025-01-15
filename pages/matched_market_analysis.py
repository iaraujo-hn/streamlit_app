import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import pearsonr
from statsmodels.stats.correlation_tools import cov_nearest

# Function to flatten the correlation matrix
def flatten_corr_matrix(corr_matrix, p_mat):
    ut = np.triu_indices(len(corr_matrix), k=1)
    
    if not (ut[0].max() < p_mat.shape[0] and ut[1].max() < p_mat.shape[1]):
        raise IndexError(f'ut indices out of bounds: {ut}')
    
    flat_corr = pd.DataFrame({
        'Potential Control Markets': corr_matrix.index[ut[0]],
        'Potential Test Markets': corr_matrix.columns[ut[1]],
        'correlation': corr_matrix.values[ut],
        'p_value': p_mat.values[ut]
    })
    return flat_corr

# Function to perform the analysis
def perform_analysis(data):
    data_filtered = data.iloc[:, 1:]
    
    corr_matrix = data_filtered.corr()
    
    p_mat = np.zeros(corr_matrix.shape)
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            _, p = pearsonr(data_filtered.iloc[:, i], data_filtered.iloc[:, j])
            p_mat[i, j] = p
            p_mat[j, i] = p
    
    corr_results = flatten_corr_matrix(corr_matrix, pd.DataFrame(p_mat, index=corr_matrix.index, columns=corr_matrix.columns))
    corr_results_filtered = corr_results[(corr_results['correlation'] > 0.8) & (corr_results['p_value'] < 0.05)]

    # Calculate summary statistics (sum, mean, median) for each market
    summary_stats = data_filtered.describe().T[['mean', '50%']]  # '50%' is the median
    summary_stats['sum'] = data_filtered.sum()

    # Rename columns to avoid conflicts later
    summary_stats.rename(columns={'mean': 'mean_kpi', '50%': 'median_kpi', 'sum': 'sum_kpi'}, inplace=True)
    
    # Merge correlation results with summary statistics for both control and test markets
    corr_results_merged = corr_results_filtered.merge(summary_stats, left_on='Potential Control Markets', right_index=True)
    corr_results_merged = corr_results_merged.merge(summary_stats, left_on='Potential Test Markets', right_index=True, suffixes=('_control', '_test'))
    
    # Filter to include only market pairs where medians are within 30% of each other
    corr_results_merged = corr_results_merged[
        (abs(corr_results_merged['median_kpi_control'] - corr_results_merged['median_kpi_test']) / corr_results_merged['median_kpi_control']) <= 0.3
    ]
    
    return corr_results_merged


# Streamlit app layout
# App Description
st.markdown('# Matched Market Analysis')
st.markdown('---')
st.write("""
         The Matched Market Analysis Tool is designed to assist the team in selecting the best test and control market pairs for an A/B test.
         This app can be helpful especially in situations where the client is planning to test a new strategy. 
         The client would need to identify a test market where the new strategy will be implemented and control markets where the new strategy will not be implemented.
         The KPI used in this analysis depends on what we are trying to measure, and could be # of orders, sales, website traffic, free trials, etc.
         """)

# Data Requirements
if "show_requirements" not in st.session_state:
    st.session_state.show_requirements = False

def toggle_requirements():
    st.session_state.show_requirements = not st.session_state.show_requirements

if st.button("Hide Data Requirements" if st.session_state.show_requirements else "Show Data Requirements", on_click=toggle_requirements):
    pass

if st.session_state.show_requirements:
    st.write("""
    1. The data file must be in CSV format.
    2. The first column should contain the date in `YYYY-MM-DD` format.
    3. Each subsequent column should represent a market and its corresponding KPI for each day.
    4. Replace any special character in the Market names with "_" as they may interfere with the functions of the app.
    4. Replace NA values for the KPI with 0.
    5. The file should contain at least 30 days of data for proper analysis.
    """)

st.sidebar.title("Upload file:")

# File upload
uploaded_file = st.sidebar.file_uploader("##### Choose a CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    if st.sidebar.button("Run Analysis"):
        corr_results = perform_analysis(data)
        st.session_state['corr_results'] = corr_results  # Save results in session_state
    
if 'corr_results' in st.session_state:
    corr_results = st.session_state['corr_results']
    st.markdown('---')
    st.write('##### Select Control and Test Markets:')
    
    controls = sorted(corr_results['Potential Control Markets'].unique())
    tests = sorted(corr_results['Potential Test Markets'].unique())
    
    with st.expander("Filter Controls", expanded=False):
        selected_controls = st.multiselect("Select Control Markets", options=controls, default=controls)
        
    with st.expander("Filter Tests", expanded=False):
        selected_tests = st.multiselect("Select Test Markets", options=tests, default=tests)
    
    filtered_results = corr_results[
        corr_results['Potential Control Markets'].isin(selected_controls) & 
        corr_results['Potential Test Markets'].isin(selected_tests)
    ]
    
    st.dataframe(filtered_results)  # Display filtered results
    
    csv = filtered_results.to_csv(index=False)
    st.download_button(
        label="Download Results",
        data=csv,
        file_name="filtered_correlation_analysis_results.csv",
        mime="text/csv"
    )
    
    # Plot top 5 markets over time
    st.write("""
         ### Top 5 Markets Over Time
         This chart shows the top 5 markets with the largest volumes over time.
         """)
    data['date'] = pd.to_datetime(data.iloc[:, 0])
    data_long = data.melt(id_vars=['date'], var_name='Markets', value_name='kpi')
    
    top_5_markets = data_long.groupby('Markets')['kpi'].sum().nlargest(5).index
    filtered_data = data_long[data_long['Markets'].isin(top_5_markets)]
    
    pivot_data = filtered_data.pivot_table(index='date', columns='Markets', values='kpi', aggfunc='sum')
    pivot_data.fillna(0, inplace=True)

    fig, ax = plt.subplots(dpi=350)
    ax.stackplot(pivot_data.index, pivot_data.T, labels=pivot_data.columns, alpha=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))  # Add thousand separators

    plt.xticks(rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.set_title("Top 5 Markets Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("KPI")
    st.pyplot(fig, use_container_width=True)  # Use container width to make it responsive

    # Plot total sum over time (Trend Over Time)
    st.write("""
        ### Trend Over Time
        This chart shows the trend over time for the sum of all markets included in this analysis.
        """)

    # Create columns to limit the chart width
    col1, col2, col3 = st.columns([1, 2, 1]) 

    with col2:  # Place the chart in the middle column to limit its width
        total_data = data_long.groupby('date')['kpi'].sum().reset_index()
        
        fig, ax = plt.subplots()
        sns.lineplot(data=total_data, x='date', y='kpi', ax=ax)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))  # Add thousand separators

        plt.xticks(rotation=45)
        st.pyplot(fig, use_container_width=True)
