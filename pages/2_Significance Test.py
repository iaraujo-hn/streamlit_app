import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import t
import math
from math import sqrt
from scipy.stats import norm

# Function
def significance_test(control_size, control_conversion, test_size, test_conversion, confidence_interval):
    # Calculate conversion rates
    control_cvr = control_conversion / control_size
    test_cvr = test_conversion / test_size

    # Fixes the error when control and test cvr are the same 
    control_cvr = control_cvr*1.000000001 if control_cvr == test_cvr else control_cvr

    # Calculate standard deviations
    control_std = np.sqrt(control_cvr * (1 - control_cvr))
    test_std = np.sqrt(test_cvr * (1 - test_cvr))
    
    cvr_lift = (test_cvr / control_cvr)-1
    
    # Calculate pooled standard deviation
    sp = np.sqrt(((control_size - 1) * control_std ** 2 + (test_size - 1) * test_std ** 2) / (test_size + control_size - 2))

    # T-Score
    t_score = np.abs((test_cvr - control_cvr) / (sp * np.sqrt(1/test_size + 1/control_size))) 
    
    # Calculate degrees of freedom
    df = test_size + control_size - 2

    # Calculate p-value
    p_value = 2 * (1 - t.cdf(np.abs(t_score), df))
    
    # Print results
    confidence_interval = float(confidence_interval.strip('%')) / 100
    threshold = 1 - confidence_interval
    
    if p_value < threshold:
        significant_result = '<span style="color:green;"><b>Significant!</b></span>'
    else:
        significant_result = '<span style="color:red;"><b>Not significant!</b></span>'

    if p_value < threshold and test_cvr > control_cvr:
        result_message = f'<b>{significant_result}</b>\n\nTest Group conversion rate ({test_cvr*100:.4f}%) was {cvr_lift * 100:.1f}% higher than Control Group conversion rate ({control_cvr*100:.4f}%).\n\n<b>You can be {int(confidence_interval*100)}% confident that Test Group will perform better than Control Group.</b>'
    elif p_value < threshold and test_cvr < control_cvr:
        result_message = f'<b>{significant_result}</b>\n\nTest Group conversion rate ({test_cvr*100:.4f}%) was {cvr_lift * 100:.1f}% lower than Control Group conversion rate ({control_cvr*100:.4f}%).\n\n<b>You can be {int(confidence_interval*100)}% confident that Control Group will perform better than Test Group.</b>'
    elif p_value > threshold and test_cvr > control_cvr:
        result_message = f'<b>{significant_result}</b>\n\nTest Group conversion rate ({test_cvr*100:.4f}%) was {cvr_lift * 100:.1f}% higher than Control Group conversion rate ({control_cvr*100:.4f}%).\n\n<b>However, you cannot be {int(confidence_interval*100)}% confident that Test Group will perform better than Control Group.</b>'
    elif p_value > threshold and test_cvr < control_cvr:
        result_message = f'<b>{significant_result}</b>\n\nTest Group conversion rate ({test_cvr*100:.4f}%) was {cvr_lift * 100:.1f}% lower than Control Group conversion rate ({control_cvr*100:.4f}%).\n\n<b>However, you cannot be {int(confidence_interval*100)}% confident that Control Group will perform better than Test Group.</b>'
        
    result_dict = {
        'Metric': ['Control Rate', 'Test Rate', 't-score', 'p-value', 'Result', 'Confidence level'],
        'Result': [f'{control_cvr*100:.3}%', f'{test_cvr*100:.3}%', t_score, p_value, significant_result, f'{confidence_interval*100}%']
    }
    
    result_df = pd.DataFrame(result_dict)

    
    return result_message, result_df

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Streamlit page configuration #

st.set_page_config(page_title="Statistical Significance Test", layout="wide")

st.title('Statistical Significance Test')

st.markdown("---")

st.write('This tool helps you determine if the difference in conversion rates between your control and test groups is statistically significant using a two-tailed test. Enter the sizes and conversions for both groups, along with your desired confidence level.\n\nThe tool will then tell you if the difference is significant and provide detailed results, helping you make informed decisions about your test outcomes.')

st.sidebar.title('A/B Test Parameters')

# Control group input
st.sidebar.write('### Control Group')
control_size = st.sidebar.number_input("Control Group Size", value=1000)
control_conversion = st.sidebar.number_input("Control Group Conversions", value=50)

st.sidebar.write('### Test Group')
test_size = st.sidebar.number_input("Test Group Size", value=1000)
test_conversion = st.sidebar.number_input("Test Group Conversions", value=60)

# confidence interval
confidence_interval = st.sidebar.selectbox("#### Statistical Confidence", ['90%', '95%', '99%'], index=1)

# Run results
if st.sidebar.button("Calculate"):
    st.write("#### Result:")
    result_message, result_df = significance_test(control_size, control_conversion, test_size, test_conversion, confidence_interval)
    
    st.markdown(f'<p style="font-size:28px; color:green;">{result_message}</p>', unsafe_allow_html=True)

    st.write('#### Test Results:')
    st.write(result_df.to_html(escape=False), unsafe_allow_html=True)
    

st.sidebar.markdown("---")
st.sidebar.image("./images/hn-logo.png", output_format="PNG", use_column_width="always")