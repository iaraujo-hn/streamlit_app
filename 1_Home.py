import streamlit as st


# Page config
st.set_page_config(page_title="Testing Tool")

st.markdown("""
    <style>
    .link {
        font-size: 24px;
        text-decoration: none;
        color: black !important;
        font-weight: bold;
        background-color: rgba(43, 74, 153, 0.05);
        padding: 5px; 
        border-radius: 10px;
    }
    .link:hover {
        background-color: rgba(43, 74, 153, 0.2);
    }
    }
    </style>
    """, unsafe_allow_html=True)


st.title("Testing Tools")
st.sidebar.success('Select page above')
st.sidebar.markdown("---")
st.sidebar.image("images/hn-logo.png", output_format="PNG", use_column_width="always")

# styled links
st.markdown('<a href="pages/2_Significance Test.py" class="link"> Significance Test </a>', unsafe_allow_html=True)
st.write('This tool helps you determine if the differences in conversion rates between your control group and test group are statistically significant. Enter your data on the left side to perform the test.')
st.image("images/significance_test.PNG", caption=" Significance Test ", use_column_width=True)

st.markdown('<a href="pages/3_Sample Size Estimator.py" class="link"> Sample Size Estimator </a>', unsafe_allow_html=True)
st.write('This tool calculates the required sample size for control and test groups in an A/B test, along with the associated budget and estimated duration. Enter the parameters in the sidebar to see the results.')
st.image("images/sample_size_estimator.PNG", caption="Sample Size Estimator", use_column_width=True)

# # streamlit run 1_Home.py --server.enableXsrfProtection false