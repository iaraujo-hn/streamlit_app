import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import t

# Function
def significance_test(control_size, control_conversion, test_size, test_conversion, confidence_interval):
    # Calculate conversion rates
    control_cvr = control_conversion / control_size
    test_cvr = test_conversion / test_size

    # Fix error when control and test cvr are the same
    control_cvr = control_cvr * 1.000000001 if control_cvr == test_cvr else control_cvr

    # Calculate standard deviations
    control_std = np.sqrt(control_cvr * (1 - control_cvr))
    test_std = np.sqrt(test_cvr * (1 - test_cvr))
    
    cvr_lift = (test_cvr / control_cvr) - 1
    
    # Pooled standard deviation
    sp = np.sqrt(((control_size - 1) * control_std ** 2 + (test_size - 1) * test_std ** 2) / (test_size + control_size - 2))

    # T-Score
    t_score = np.abs((test_cvr - control_cvr) / (sp * np.sqrt(1/test_size + 1/control_size)))
    
    # Degrees of freedom
    df = test_size + control_size - 2

    # P-value
    p_value = 2 * (1 - t.cdf(np.abs(t_score), df))
    
    # Confidence threshold
    confidence_interval = float(confidence_interval.strip('%')) / 100
    threshold = 1 - confidence_interval

    significant_result = '<span style="color:green;"><b>Significant!</b></span>' if p_value < threshold else '<span style="color:red;"><b>Not significant!</b></span>'
    result_dict = {
        'Metric': ['Control Rate', 'Test Rate', 't-score', 'p-value', 'Confidence Level'],
        'Result': [f'{control_cvr*100:.3}%', f'{test_cvr*100:.3}%', t_score, p_value, f'{confidence_interval*100}%']
    }
    
    result_df = pd.DataFrame(result_dict)
    return significant_result, result_df

# Streamlit Page Configuration
st.set_page_config(page_title="Statistical Significance Test", layout="wide")

st.title('Statistical Significance Test')
st.markdown("---")
st.write('This tool determines if the difference in conversion rates between your control and test groups is statistically significant.')

# Sidebar Input Options
st.sidebar.title('A/B Test Parameters')
input_mode = st.sidebar.radio("Input Mode", ["Manual Input", "Upload File"])

if input_mode == "Manual Input":
    # Manual Input
    st.sidebar.write('### Control Group')
    control_size = st.sidebar.number_input("Control Group Size", value=1000)
    control_conversion = st.sidebar.number_input("Control Group Conversions", value=50)

    st.sidebar.write('### Test Group')
    test_size = st.sidebar.number_input("Test Group Size", value=1000)
    test_conversion = st.sidebar.number_input("Test Group Conversions", value=60)

    confidence_interval = st.sidebar.selectbox("#### Statistical Confidence", ['90%', '95%', '99%'], index=1)

    if st.sidebar.button("Calculate"):
        result_message, result_df = significance_test(control_size, control_conversion, test_size, test_conversion, confidence_interval)
        st.markdown(f'<p style="font-size:28px; color:green;">{result_message}</p>', unsafe_allow_html=True)
        st.write('#### Test Results:')
        st.write(result_df.to_html(escape=False), unsafe_allow_html=True)

elif input_mode == "Upload File":
    # File Upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.sidebar.write("### Select Columns for A/B Test")
        control_size_col = st.sidebar.selectbox("Control Size Column", data.columns)
        control_conversion_col = st.sidebar.selectbox("Control Conversion Column", data.columns)
        test_size_col = st.sidebar.selectbox("Test Size Column", data.columns)
        test_conversion_col = st.sidebar.selectbox("Test Conversion Column", data.columns)

        confidence_interval = st.sidebar.selectbox("#### Statistical Confidence", ['90%', '95%', '99%'], index=1)

    if st.sidebar.button("Run A/B Test"):
        results = []

        for idx, row in data.iterrows():
            significant_result, result_df = significance_test(
                row[control_size_col],
                row[control_conversion_col],
                row[test_size_col],
                row[test_conversion_col],
                confidence_interval
            )
            # Append original row and calculated results
            row_result = row.to_dict()  # Convert the current row to a dictionary
            row_result.update({  # Add calculated metrics
                'Control Rate': result_df.iloc[0, 1],
                'Test Rate': result_df.iloc[1, 1],
                'T-Score': result_df.iloc[2, 1],
                'P-Value': result_df.iloc[3, 1],
                'Confidence Level': result_df.iloc[4, 1],
                'Significant': significant_result
            })
            results.append(row_result)

        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Convert DataFrame to HTML for better rendering
        html_table = results_df.to_html(escape=False, index=False)
        
        # Display the results as a markdown-rendered HTML table
        st.write("#### Results from Uploaded File")
        st.markdown(html_table, unsafe_allow_html=True)