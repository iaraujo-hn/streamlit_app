import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import t

# Function to calculate conversion rates
def calculate_conversion_rates(control_size, control_conversion, test_size, test_conversion):
    control_cvr = control_conversion / control_size
    test_cvr = test_conversion / test_size
    control_cvr = control_cvr * 1.000000001 if control_cvr == test_cvr else control_cvr  # Fix potential issues
    return control_cvr, test_cvr

# Function to calculate pooled standard deviation and t-score
def calculate_t_statistics(control_size, control_cvr, test_size, test_cvr):
    control_std = np.sqrt(control_cvr * (1 - control_cvr))
    test_std = np.sqrt(test_cvr * (1 - test_cvr))
    sp = np.sqrt(((control_size - 1) * control_std ** 2 + (test_size - 1) * test_std ** 2) / (control_size + test_size - 2))
    t_score = np.abs((test_cvr - control_cvr) / (sp * np.sqrt(1 / control_size + 1 / test_size)))
    return sp, t_score

# Function to calculate p-value and significance
def calculate_p_value(t_score, control_size, test_size, confidence_interval):
    df = control_size + test_size - 2
    p_value = 2 * (1 - t.cdf(np.abs(t_score), df))
    threshold = 1 - confidence_interval
    is_significant = p_value < threshold
    return p_value, is_significant

# Function to generate the detailed result message
def generate_result_message(is_significant, control_cvr, test_cvr, cvr_lift, confidence_level):
    if is_significant:
        if test_cvr > control_cvr:
            message = (f'<span style="color:green;"><b>Significant!</b></span>\n\n'
                       f'Test Group conversion rate ({test_cvr*100:.4f}%) was {cvr_lift*100:.1f}% higher than '
                       f'Control Group conversion rate ({control_cvr*100:.4f}%).\n\n'
                       f'<b>You can be {confidence_level}% confident that Test Group will perform better than Control Group.</b>')
        else:
            message = (f'<span style="color:green;"><b>Significant!</b></span>\n\n'
                       f'Test Group conversion rate ({test_cvr*100:.4f}%) was {cvr_lift*100:.1f}% lower than '
                       f'Control Group conversion rate ({control_cvr*100:.4f}%).\n\n'
                       f'<b>You can be {confidence_level}% confident that Control Group will perform better than Test Group.</b>')
    else:
        if test_cvr > control_cvr:
            message = (f'<span style="color:red;"><b>Not significant!</b></span>\n\n'
                       f'Test Group conversion rate ({test_cvr*100:.4f}%) was {cvr_lift*100:.1f}% higher than '
                       f'Control Group conversion rate ({control_cvr*100:.4f}%).\n\n'
                       f'<b>However, you cannot be {confidence_level}% confident that Test Group will perform better than Control Group.</b>')
        else:
            message = (f'<span style="color:red;"><b>Not significant!</b></span>\n\n'
                       f'Test Group conversion rate ({test_cvr*100:.4f}%) was {cvr_lift*100:.1f}% lower than '
                       f'Control Group conversion rate ({control_cvr*100:.4f}%).\n\n'
                       f'<b>However, you cannot be {confidence_level}% confident that Control Group will perform better than Test Group.</b>')
    return message

# Function to perform significance test
def significance_test(control_size, control_conversion, test_size, test_conversion, confidence_interval):
    confidence_interval = float(confidence_interval.strip('%')) / 100
    control_cvr, test_cvr = calculate_conversion_rates(control_size, control_conversion, test_size, test_conversion)
    _, t_score = calculate_t_statistics(control_size, control_cvr, test_size, test_cvr)
    p_value, is_significant = calculate_p_value(t_score, control_size, test_size, confidence_interval)
    
    cvr_lift = (test_cvr / control_cvr) - 1
    result_message = generate_result_message(is_significant, control_cvr, test_cvr, cvr_lift, int(confidence_interval * 100))
    significance_result = "Significant" if is_significant else "Not Significant"
    result_df = pd.DataFrame({
        'Metric': ['Control Rate', 'Test Rate', 't-score', 'p-value', 'Confidence Level'],
        'Result': [f'{control_cvr*100:.3}%', f'{test_cvr*100:.3}%', t_score, p_value, f'{confidence_interval*100}%']
    })
    return result_message, significance_result, result_df

# Function to handle manual input
def handle_manual_input():
    control_size = st.sidebar.number_input("Control Group Size", value=1000)
    control_conversion = st.sidebar.number_input("Control Group Conversions", value=50)
    test_size = st.sidebar.number_input("Test Group Size", value=1000)
    test_conversion = st.sidebar.number_input("Test Group Conversions", value=60)
    confidence_interval = st.sidebar.selectbox("#### Statistical Confidence", ['90%', '95%', '99%'], index=1)

    if st.sidebar.button("Calculate"):
        result_message, _, result_df = significance_test(control_size, control_conversion, test_size, test_conversion, confidence_interval)
        st.write('#### Result:')
        st.markdown(f'<p style="font-size:20px;">{result_message}</p>', unsafe_allow_html=True)
        
        # Remove index and align column names to the left
        styled_df = result_df.style.hide_index().set_table_styles([{
            'selector': 'th',
            'props': [('text-align', 'left')]
        }])
        
        st.write('#### Test Results:')
        st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)

# Function to handle file upload
def handle_file_upload():
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
                _, significance_result, result_df = significance_test(
                    row[control_size_col],
                    row[control_conversion_col],
                    row[test_size_col],
                    row[test_conversion_col],
                    confidence_interval
                )
                row_result = row.to_dict()
                row_result.update({
                    'Control Rate': result_df.iloc[0, 1],
                    'Test Rate': result_df.iloc[1, 1],
                    'T-Score': result_df.iloc[2, 1],
                    'P-Value': result_df.iloc[3, 1],
                    'Confidence Level': result_df.iloc[4, 1],
                    'Significant': significance_result
                })
                results.append(row_result)

            results_df = pd.DataFrame(results)
            st.write("#### Results from Uploaded File")
            st.write(results_df.to_html(escape=False, index=False), unsafe_allow_html=True)

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Statistical Significance Test", layout="wide")
    st.title('Statistical Significance Test')
    st.markdown("---")
    st.write('This tool helps you determine if the difference in conversion rates between your control and test groups is statistically significant using a two-tailed test.')

    st.sidebar.title('A/B Test Parameters')
    input_mode = st.sidebar.radio("Input Mode", ["Manual Input", "Upload File"])

    if input_mode == "Manual Input":
        handle_manual_input()
    elif input_mode == "Upload File":
        handle_file_upload()

if __name__ == "__main__":
    main()
