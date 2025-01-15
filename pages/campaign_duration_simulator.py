import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm

## Styling

# styling threshold cells
def highlight_below_threshold(val, threshold):
    color = '#e6e6ff' if val < threshold else ''
    return f'background-color: {color}'

# styling row for selected lift
def highlight_lift_row(row, lift_value):
    return ['background-color: #ccffe6' if row['Assumed Lift'] == lift_value else '' for _ in row]

# calculate sample size
def calculate_sample_size(p1, p2, lift, confidence_level, power, k, p_hat, q_hat, q1, q2):
    part1 = np.sqrt(p_hat * q_hat * (1 + 1 / k)) * norm.ppf(confidence_level + (1 - confidence_level) / 2)
    part2 = np.sqrt(p1 * q1 + p2 * q2 / k) * norm.ppf(power)
    result = (part1 + part2) ** 2 / lift ** 2
    return result

# calculate duration
def calculate_duration(estimated_traffic, total_sample_size):
    if estimated_traffic:
        return round(total_sample_size / estimated_traffic)
    else:
        return ""

# main function
def flight_simulator(test_cvr, estimated_traffic, confidence_level, lift_value_):
    power = 0.8
    confidence_level = float(confidence_level.strip('%')) / 100
    ratios = [(30, 70), (35, 65), (40, 60), (45, 55), (50, 50)]
    
    lift_value = float(lift_value_.strip('%')) / 100
    assumed_lift = np.arange(lift_value - 0.04, lift_value + 0.05, 0.01)  # user-defined lift range
    results = []  # empty list for results
    
    test_group_cvr = test_cvr / 100 # conversion rate
    
    for lift_loop in assumed_lift:
        control_group_cvr = test_group_cvr * (1 - lift_loop)  # use the lift to calculate control group CVR
        lift = abs(control_group_cvr - test_group_cvr)
        
        duration_results = {
            "Assumed Lift": f"{lift_loop * 100:.0f}%",
            "Control Group CVR": f"{control_group_cvr * 100:.6f}%",
            "Test Group CVR": f"{test_group_cvr * 100:.6f}%",
            "Absolute Lift": f"{lift * 100:.6f}%",
        }
        
        # control/test group ratio
        for control_ratio, test_ratio in ratios:
            k = test_ratio / control_ratio  # ratio of test to control group sample size
            q1 = 1 - test_group_cvr
            q2 = 1 - control_group_cvr
            p_hat = (control_group_cvr + (k * test_group_cvr)) / (1 + k)
            q_hat = 1 - p_hat
            
            control_group_size = calculate_sample_size(
                control_group_cvr, test_group_cvr, lift, confidence_level, power, k, p_hat, q_hat, q1, q2
            )
            test_group_size = control_group_size * k
            total_sample_size = control_group_size + test_group_size   
            
            # apply duration function
            duration = calculate_duration(estimated_traffic, total_sample_size)
            duration_results[f"Split {control_ratio}/{test_ratio}"] = duration
        
        results.append(duration_results)
    
    # results
    result_df = pd.DataFrame(results)

    return result_df

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Streamlit page configuration #

st.set_page_config(page_title="Campaign Duration Simulator", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stButton button:hover {
        border: 2px solid #38b348;
        color: #01355c;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('Campaign Duration Simulator')
st.markdown("---")
st.write("The Campaign Duration Simulator is designed to help you estimate the time required for your campaign to achieve a specified conversion rate improvement.\n\nBy inputting key parameters such as your test group's conversion rate, estimated daily traffic, statistical confidence level, and the assumed lift, this tool provides an easy to understand estimation of the campaign's duration.")
st.write("#### Features")
st.markdown(f"""
            <b>Test Group CVR:</b> Conversion rate of previous tests or conversion rate of current campaigns.\n
            <b>Estimated Traffic:</b> Estimated daily impressions.\n
            <b>Threshold Days:</b> Maximum expected campaign length.\n
            <b>Statistical Confidence:</b> Choose the confidence interval for the results.\n
            <b>Assumed Lift %:</b> Use the lift from previous tests or the expected lift.\n
            <b>Absolute Lift %:</b> Absolute difference between Control and Test groups conversion rate.
""", unsafe_allow_html=True)
st.sidebar.title('Parameters')

# Parameters input
test_cvr = st.sidebar.number_input("#### Test Group CVR", value=0.004010, format="%.6f")
estimated_traffic = st.sidebar.number_input("#### Estimated Traffic", value=1000000)
threshold_days = st.sidebar.number_input("#### Threshold Days", value=100)
confidence_level = st.sidebar.selectbox("#### Statistical Confidence", ['85%', '90%', '95%', '99%'], index=1) # it should be fixed at 90%

# lift value input with selectbox
lift_value = st.sidebar.selectbox(
    "#### Assumed Lift (%)",
    [f'{i}%' for i in range(5, 51)],
    index=7  # Default value of 11%
)

# add parameters breakdown
st.markdown("---")

if st.sidebar.button("Calculate"):
    results = flight_simulator(test_cvr, estimated_traffic, confidence_level, lift_value)
    
    # apply styling
    styled_results = results.style.map(
        lambda x: highlight_below_threshold(x, threshold_days), subset=pd.IndexSlice[:, results.columns.str.startswith('Split')]
    ).apply(
        lambda x: highlight_lift_row(x, lift_value), axis=1
    )

    # show results
    st.write(f"### Results for {confidence_level} Confidence Interval")
    st.markdown(    """
    <p style="font-size: 13px; text-align: right;">*Split: The percentage distribution between Control and Test Groups. For example, a 30/70 split indicates 30% in the Control Group and 70% in the Test Group.</p>
    """,
    unsafe_allow_html=True)
    st.dataframe(styled_results, use_container_width=True, hide_index=True)

    # add legend for colors
    st.markdown("""
        <ul style="list-style-type: none;">
            <li style="font-size: 14px; "><span style="background-color: #e6e6ff; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></span>Durations below the threshold days.</li>
            <li style="font-size: 14px;"><span style="background-color: #ccffe6; width: 12px; height: 12px; display: inline-block; margin-right: 5px;"></span>Assumed lift value.</li>
        </ul>
    """, unsafe_allow_html=True)
