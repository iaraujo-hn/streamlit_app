import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import t, norm

# Function
def sample_size_estimator(
    control_group_size, control_group_conversions, 
    test_group_size, test_group_conversions, 
    confidence_level, power, estimated_cpm, estimated_traffic):
    
    confidence_level = float(confidence_level.strip('%')) / 100

    # Calculate conversion rates, aka p1&p2. Projected true probabilities of success in the two groups
    control_group_cvr = control_group_conversions / control_group_size
    test_group_cvr = test_group_conversions / test_group_size
    lift = abs(control_group_cvr - test_group_cvr)
    
    # Constants for now
    k = 2  # Ratio of test to control group sample size
    q1 = 1 - test_group_cvr
    q2 = 1 - control_group_cvr
    p_hat = (control_group_cvr + (k * test_group_cvr)) / (1 + k)
    q_hat = 1 - p_hat
    
    # Function to calculate sample size
    def calculate_sample_size(p1, p2, lift, confidence_level, power, k, p_hat, q_hat, q1, q2):
        part1 = np.sqrt(p_hat * q_hat * (1 + 1 / k)) * norm.ppf(confidence_level + (1 - confidence_level) / 2)
        part2 = np.sqrt(p1 * q1 + p2 * q2 / k) * norm.ppf(power)
        result = (part1 + part2) ** 2 / lift ** 2
        return result
    
    # Calculate control group size and test group size
    control_group_size = calculate_sample_size(control_group_cvr, test_group_cvr, lift, confidence_level, power, k, p_hat, q_hat, q1, q2)
    test_group_size = control_group_size * k
    total_sample_size = control_group_size + test_group_size
    
    # Function to calculate budget
    def calculate_budget(control_group_size, test_group_size, estimated_cpm):
        control_budget = control_group_size * estimated_cpm / 1000
        test_budget = test_group_size * estimated_cpm / 1000
        total_budget = control_budget + test_budget
        return control_budget, test_budget, total_budget
    
    # Calculate budget
    control_budget, test_budget, total_budget = calculate_budget(control_group_size, test_group_size, estimated_cpm)
    
    # Function to calculate duration, estimated run time(days)
    def calculate_duration(estimated_traffic, total_sample_size):
        if estimated_traffic:
            return round(total_sample_size / estimated_traffic)
        else:
            return ""
    
    # Calculate duration
    duration = calculate_duration(estimated_traffic, total_sample_size)
    
    # Storing everything in a dictionary
    results_dict = {
        "Control Group Size": f"{round(control_group_size):,}",
        "Test Group Size": f"{round(test_group_size):,}",
        "Total Sample Size": f"{round(total_sample_size):,}",
        "Control Budget": f"${round(control_budget):,}",
        "Test Budget": f"${round(test_budget):,}",
        "Total Budget": f"${round(total_budget):,}",
        "Estimated Duration (days)": f"{round(duration):,}"
    }
    return results_dict


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Streamlit page configuration #

# Page config
st.set_page_config(page_title="Sample Size Estimator", layout="wide")

st.title('Sample Size Estimator')
st.write('This tool calculates the required sample size for control and test groups in an A/B test, along with the associated budget and estimated duration. Enter the parameters in the sidebar to see the results.')

st.sidebar.title('A/B Test Parameters')
# Control group input
st.sidebar.write('### Control Group')
control_group_size = st.sidebar.number_input("Control Group Size", value=5686901)
control_group_conversions = st.sidebar.number_input("Control Group Conversions", value=2102)

# Test group input
st.sidebar.write('### Test Group')
test_group_size = st.sidebar.number_input("Test Group Size", value=839045)
test_group_conversions = st.sidebar.number_input("Test Group Conversions", value=325)

# Parameters input
st.sidebar.write('### Parameters')
confidence_level = st.selectbox("##### Select Statistical Confidence", ['90%', '95%', '99%'], index=1)
power = st.sidebar.number_input("Power", value=0.8)
estimated_cpm = st.sidebar.number_input("Estimated CPM", value=7.2)
estimated_traffic = st.sidebar.number_input("Estimated Traffic", value=1342086)
st.sidebar.markdown("---")
st.sidebar.image("images/hn-logo.png", output_format="PNG", use_column_width="always")


if st.button("Calculate"):
    results = sample_size_estimator(
        control_group_size, control_group_conversions, 
        test_group_size, test_group_conversions, 
        confidence_level, power, estimated_cpm, estimated_traffic
    )
    
    # Convert results to DataFrame and transpose it
    results_df = pd.DataFrame([results]).T
    results_df.columns = ['Value']

    # Show results
    st.write("#### Results")
    st.dataframe(results_df)

# streamlit run 1_Home.py --server.enableXsrfProtection false