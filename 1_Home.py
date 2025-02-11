import streamlit as st

pg = st.navigation({
    "Main": [
        st.Page("app_pages/homepage.py", title="Homepage", icon=":material/home:")
        ],
    "Advanced Analytics Apps": [
        st.Page("app_pages/qa_tool.py", title="QA Tool", icon=':material/task:'),
        st.Page("app_pages/significance_test.py", title="Significance Test", icon=':material/check:'),
        st.Page("app_pages/sample_size_estimator.py", title="Sample Size Estimator", icon=':material/groups:'),
        st.Page("app_pages/campaign_duration_simulator.py", title="Campaign Duration Simulator", icon=':material/schedule:'),
        st.Page("app_pages/intersect_impressions_estimator.py", title="Intersect Impressions Estimator", icon=':material/query_stats:'),
        st.Page("app_pages/matched_market_analysis.py", title="Matched Market Analysis Tool", icon=':material/my_location:'),
        st.Page("app_pages/minimum_detectable_effect.py", title="Minimum Detectable Effect", icon=':material/trending_up:'),
        ]
    })

pg.run()

st.sidebar.image("app/images/hn-logo.png", output_format="PNG")