import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm

# Function to apply styling
def highlight_below_threshold(val, threshold):
    color = '#e6e6ff' if val < threshold else ''
    return f'background-color: {color}'

def highlight_lift_row(row, lift_value):
    return ['background-color: #ccffe6' if row['Assumed Lift'] == lift_value else '' for _ in row]

# Function to calculate sample size
def calculate_sample_size(p1, p2, lift, confidence_level, power, k, p_hat, q_hat, q1, q2):
    part1 = np.sqrt(p_hat * q_hat * (1 + 1 / k)) * norm.ppf(confidence_level + (1 - confidence_level) / 2)
    part2 = np.sqrt(p1 * q1 + p2 * q2 / k) * norm.ppf(power)
    result = (part1 + part2) ** 2 / lift ** 2
    return result

# Function to calculate duration
def calculate_duration(estimated_traffic, total_sample_size):
    if estimated_traffic:
        return round(total_sample_size / estimated_traffic)
    else:
        return ""

# Function
def flight_simulator(test_cvr, estimated_traffic, confidence_level, lift_value_):
    power = 0.8
    confidence_level = float(confidence_level.strip('%')) / 100
    ratios = [(30, 70), (35, 65), (40, 60), (45, 55), (50, 50)]
    
    lift_value = float(lift_value_.strip('%')) / 100
    assumed_lift = np.arange(lift_value - 0.04, lift_value + 0.05, 0.01)  # Use user-defined lift range
    results = []  # Empty list results
    
    test_group_cvr = test_cvr / 100 # Calculate conversion rate
    
    for lift_loop in assumed_lift:
        control_group_cvr = test_group_cvr * (1 - lift_loop)  # Use the lift to calculate control group CVR
        lift = abs(control_group_cvr - test_group_cvr)
        
        duration_results = {
            "Assumed Lift": f"{lift_loop * 100:.0f}%",
            "Control Group CVR": f"{control_group_cvr * 100:.6f}%",
            "Test Group CVR": f"{test_group_cvr * 100:.6f}%",
            "Absolute Lift": f"{lift * 100:.6f}%",
        }
        
        # Control-test group ratios and relevant inputs
        for control_ratio, test_ratio in ratios:
            # Formulas
            k = test_ratio / control_ratio  # Ratio of test to control group sample size
            q1 = 1 - test_group_cvr
            q2 = 1 - control_group_cvr
            p_hat = (control_group_cvr + (k * test_group_cvr)) / (1 + k)
            q_hat = 1 - p_hat
            
            control_group_size = calculate_sample_size(
                control_group_cvr, test_group_cvr, lift, confidence_level, power, k, p_hat, q_hat, q1, q2
            )
            test_group_size = control_group_size * k
            total_sample_size = control_group_size + test_group_size   
            
            # Apply duration function
            duration = calculate_duration(estimated_traffic, total_sample_size)
            duration_results[f"Split {control_ratio}/{test_ratio}"] = duration
        
        results.append(duration_results)
    
    # Dictionary to dataframe
    result_df = pd.DataFrame(results)

    return result_df

# Streamlit page configuration #

# Page config
st.set_page_config(page_title="Campaign Duration Simulator", layout="wide")

st.title('Campaign Duration Simulator')
st.write('This tool helps you estimate how long it will take for your campaign to achieve a specific conversion rate lift.')

st.sidebar.title('Parameters')

# Test group input
test_cvr = st.sidebar.number_input("#### Test Group CVR", value=0.004010, format="%.6f")

# Other parameters
estimated_traffic = st.sidebar.number_input("#### Estimated Traffic", value=1342086)

# Threshold input
threshold_days = st.sidebar.number_input("#### Threshold Days", value=100)

# Confidence level input
confidence_level = st.sidebar.selectbox("#### Select Statistical Confidence", ['90%', '95%', '99%'], index=1)

# Lift value input with selectbox
lift_value = st.sidebar.selectbox(
    "#### Select Assumed Lift (%)",
    [f'{i}%' for i in range(1, 51)],
    index=10  # Default value of 11%
)

st.sidebar.markdown("---")
st.sidebar.image("images/hn-logo.png", output_format="PNG", use_column_width="always")

if st.button("Calculate"):
    results = flight_simulator(test_cvr, estimated_traffic, confidence_level, lift_value)
    
    # Apply styling to DataFrame
    styled_results = results.style.applymap(
        lambda x: highlight_below_threshold(x, threshold_days), subset=pd.IndexSlice[:, results.columns.str.startswith('Split')]
    ).apply(
        lambda x: highlight_lift_row(x, lift_value), axis=1
    )
    # Show results
    st.sidebar.markdown("---")
    st.write(f"### Results for {confidence_level} Confidence Interval")
    st.sidebar.markdown("---")
    st.dataframe(styled_results, use_container_width=True, hide_index=True)

    # Add legend for colors
    st.markdown("""
        <ul style="list-style-type: none;">
            <li style="background-color: #e6e6ff; font-size: 14px;">Durations below the threshold days.</li>
            <li style="background-color: #ccffe6; font-size: 14px;">Assumed lift value.</li>
        </ul>
    """, unsafe_allow_html=True)

st.markdown("---")
st.write('### Parameters Breakdown:')
st.markdown(f"""
            <b>Test Group CVR:</b> Conversion rate of previous tests or conversion rate (CVR) of current campaigns.\n
            <b>Estimated Traffic:</b> Estimated daily impressions.\n
            <b>Thereshold Days:</b> Maximum expected campaign length.\n
            <b>Select Statistical Confidence:</b> Choose the confidence interval for the results.\n
            <b>Select Assumed Lift %:</b> Use the lift from previous tests or the expected lift.
""", unsafe_allow_html=True)
st.markdown("---")

