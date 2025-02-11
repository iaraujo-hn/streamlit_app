import pandas as pd
import streamlit as st

def get_panel_imp(rr, tstat):
    """
    A function that estimates how many panel impressions are required given a destired tstat and average response rate (rr) 
    """
    panel_imp = (1.44 * tstat * (rr * (1 - rr))**0.5 / rr) ** 2
    return panel_imp
    
def get_total_imp(panel_imps, factor):
    """
    A function that returns total impressions from a known ratio of panel imps to total imps
    """
    total_imp = panel_imps * factor
    return f"{round(total_imp):,}"

def get_analysis_dataframe(rr, tstat, total_imps):
    """
    A function that returns a basic dataframe of key data points from a T-stat Impressions Analysis
    """
    results = {
        "Average Estimated RR%": [rr * 100],
        "Statisically Significant T-Stat": [tstat],
        "Total Impressions Per Week": [total_imps]
    }

    final_df = pd.DataFrame(results)
    return final_df

st.set_page_config(page_title="Intersect Impressions Estimator", layout="wide")

st.title('Intersect Impressions Estimator')
st.markdown("---")
st.write('This tool helps to estimate the impressions needed for a campaign to generate'
         ' a statistically signifcant result in Intersect.')

response_rate = st.number_input(
    "**Estimated Average Response Rate for Campaign (use decimal form):**",
    value=0.000015,
    format="%.6f"
)
tstat = st.number_input(
    "**T-Stat (2.0 is standard as statistically significant):**",
    value=2.0
)
factor = st.number_input(
    "**Ratio of Panel Impressions to National Impressions (use total impressions/tracked media events for most recent 4-8 weeks in Intersect Report):**",
    value=100
)

if st.button("Calculate"):
    panel_imps = get_panel_imp(response_rate, tstat)
    total_imp = get_total_imp(panel_imps, factor)
    analysis_dataframe = get_analysis_dataframe(response_rate, tstat, total_imp)

    st.markdown('---')
    st.markdown('**Results:**')
    st.dataframe(analysis_dataframe, hide_index=True)
    st.write(f'The analysis indicates that with a response rate of **{response_rate*100}%**, the amount of impressions needed for a statistically significant result is **{total_imp}** per week.')
