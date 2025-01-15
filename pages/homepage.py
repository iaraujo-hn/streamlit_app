import streamlit as st 

# Page config
st.set_page_config(page_title="Horizon Next Apps", layout="wide")

# css code
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

st.title("Welcome to the Horizon Next Tools Platform!")
st.markdown("##### Unlock the power of data-driven decisions with our collection of testing tools, designed to help you achieve accurate and meaningful insights.")

st.markdown("--------------------")
st.markdown("### Our Tools: ")

st.markdown("##### 1. Significance Testing: ")
st.write('Perform a two-sample t-test to determine if your results are statistically significant. This tool helps assess whether the differences observed are meaningful.')

st.markdown("##### 2. Sample Size Estimator: ")
st.write('Maximize your chances of achieving statistical significance in your campaigns. This tool calculates the necessary sample size for control and test groups in an A/B test, aiding in budget planning and campaign duration.')

st.markdown("##### 3. Campaign Duration Simulator: ")
st.write('Estimate the optimal campaign duration when factors like lift and control group size are uncertain. This simulation tool helps you plan effectively by considering various scenarios.')

st.markdown("##### 4. Matched Market Analysis Tool")
st.write('Select the best test and control market pairs for an A/B test')
