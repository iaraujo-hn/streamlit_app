import streamlit as st
import numpy as np
from scipy.stats import norm

def calculate_mde(p, n1, n2, alpha, power):
    try:
        # Calculate Z-scores
        z_alpha = norm.ppf(1 - alpha/2)
        z_power = norm.ppf(power)
        
        # Calculate standard error
        se = np.sqrt(p * (1 - p) * (1/n1 + 1/n2))
        
        # Calculate absolute MDE and relative lift
        mde_absolute = (z_alpha + z_power) * se
        lift_pct = (mde_absolute / p) * 100
        
        return round(lift_pct, 2)
    except Exception as e:
        return f"Error: {str(e)}"

# Page configuration
st.set_page_config(page_title="MDE Calculator", layout="wide")
st.title("📈 Minimum Detectable Effect (MDE) Calculator")
st.markdown("---")

# Introduction Section
st.markdown("""
**Minimum Detectable Effect (MDE)** informs the minimum lift you can reliably detect in an A/B test.
It helps you understand:
- If the lift you're looking for is realistic based on business goals
- If you need to adjust test and control group sizes
""")

st.write("**Ideal Use Case:** Pre-campaign planning. Estimate the minimum detectable effect from given current campaign setup and determine if the test is feasible.")            

st.markdown("---")

# Sidebar inputs
st.sidebar.title("Parameters")

# Baseline conversion rate
st.sidebar.markdown("### Baseline Conversion Rate")
p = st.sidebar.number_input("Baseline conversion rate (p)", 
                          min_value=0.0001, max_value=1.0, 
                          value=0.1, step=0.01,
                          format="%.4f")

# Statistical parameters
st.sidebar.markdown("### Statistical Parameters")
confidence_level = st.sidebar.selectbox("Confidence level", 
                                      ["85%", "90%", "95%", "99%"], index=1)
power = st.sidebar.number_input("Statistical power (%)", 
                              min_value=50, max_value=99, 
                              value=80, step=1) / 100

# Convert confidence level to alpha
alpha_map = {"85%": 0.15, "90%": 0.10, "95%": 0.05, "99%": 0.01}
alpha = alpha_map[confidence_level]

# Sample size input
st.sidebar.markdown("### Sample Size")
n1 = st.sidebar.number_input("Control group size (n1)", 
                           min_value=1, value=1000)
n2 = st.sidebar.number_input("Test group size (n2)", 
                           min_value=1, value=1000)

# Main calculation
if st.sidebar.button("Calculate MDE"):
    with st.spinner("Calculating..."):
        try:
            mde = calculate_mde(p, n1, n2, alpha, power)
            
            # Display results
            st.markdown("## 📊 Results")
            st.markdown(f"### 🔍 Minimum Detectable Effect: **{mde:.1f}%**")
            st.markdown(f"""
            - **Test Group Adjustment**  
            To reduce your MDE by 20% (to **{mde*0.80:.1f}%**):  
            → Keep control group fixed at **{n1:,}**  
            → Increase test group to **{int(n2/(0.80**2)):,}**  
            *(Current test group: {n2:,} → New target: {int(n2/(0.80**2)):,})*

            """)

            # Calculation details
            with st.expander("Detailed Calculation Breakdown"):
                st.markdown(f"""
                **Input Parameters:**
                - Baseline conversion rate: {p:.4f}
                - Control group size: {n1:,}
                - Test group size: {n2:,}
                - Confidence level: {confidence_level} (α = {alpha})
                - Statistical power: {power:.0%}
                """)
                
        except Exception as e:
            st.error(f"Error in calculation: {str(e)}")