import streamlit as st
import pandas as pd

from utils.usgs_lookup import build_hydro_context
from utils.hydro_logic import (
    compute_bankfull_metrics,
    recommend_action,
    format_summary_table,
)
from utils.mapping import build_decision_map


st.set_page_config(
    page_title="ARC Hydrokinetic Deployment Advisor",
    page_icon="🌊",
    layout="wide",
)

st.title("ARC Hydrokinetic Deployment Advisor")
st.caption("Starter Streamlit app for Western North Carolina regional-curve deployment screening.")

with st.sidebar:
    st.header("Inputs")
    lat = st.number_input("Latitude", value=35.430100, format="%.6f")
    lon = st.number_input("Longitude", value=-83.447200, format="%.6f")
    depth_ft = st.number_input("Measured water depth (ft)", min_value=0.0, value=1.50, step=0.1)
    use_manual_da = st.checkbox("Manually enter drainage area", value=False)

    drainage_area_manual = None
    if use_manual_da:
        drainage_area_manual = st.number_input(
            "Drainage area (mi²)",
            min_value=0.0001,
            value=5.0,
            step=0.1,
            format="%.4f",
        )

    run_button = st.button("Run Deployment Recommendation", use_container_width=True)

st.markdown(
    """
This tool uses:
- manual **latitude**
- manual **longitude**
- manual **depth**
- USGS lookup when available
- Western North Carolina **regional curves** as a starting decision framework
"""
)

if run_button:
    with st.spinner("Building hydro context and recommendation..."):
        hydro = build_hydro_context(lat, lon)

        drainage_area_sqmi = drainage_area_manual if use_manual_da else hydro.drainage_area_sqmi

        if drainage_area_sqmi is None:
            st.error(
                "USGS drainage-area lookup was unavailable for this point. "
                "Check 'Manually enter drainage area' in the sidebar and try again."
            )
            st.stop()

        bankfull = compute_bankfull_metrics(drainage_area_sqmi)
        recommendation = recommend_action(depth_ft, bankfull)
        result_map = build_decision_map(lat, lon, depth_ft, hydro, recommendation)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Recommendation")
        deploy_value = recommendation["deploy"]
        if deploy_value == "YES":
            st.success(f"Deploy now? {deploy_value}")
        elif deploy_value == "MAYBE":
            st.warning(f"Deploy now? {deploy_value}")
        else:
            st.error(f"Deploy now? {deploy_value}")

        st.write(f"**Action:** {recommendation['action']}")
        st.write(f"**Score:** {recommendation['score']}")
        st.write(f"**Reason:** {recommendation['reason']}")
        st.write(f"**ARC navigation note:** {recommendation['nav_note']}")

        st.subheader("Hydro Context")
        st.write(f"**Stream name:** {hydro.stream_name or 'Unknown'}")
        st.write(f"**COMID:** {hydro.comid or 'N/A'}")
        st.write(f"**ReachCode:** {hydro.reachcode or 'N/A'}")
        st.write(f"**Drainage area:** {drainage_area_sqmi:.3f} mi²")
        st.write(f"**Lookup source:** {hydro.source}")
        st.write(f"**Notes:** {hydro.notes}")

    with col2:
        st.subheader("Map")
        try:
            from streamlit_folium import st_folium
            st_folium(result_map, width=None, height=500)
        except Exception:
            st.info("Install streamlit-folium to render the map inside Streamlit.")
            st.write("The rest of the app still works.")

    st.subheader("Regional-Curve Summary")
    summary_df = format_summary_table(drainage_area_sqmi, bankfull, depth_ft)
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Interpretation")
    st.markdown(
        """
This version is a first-pass screening tool. It is useful for:
- narrowing down likely deployment candidates
- comparing field depth to estimated bankfull geometry
- giving ARC an initial deploy / do-not-deploy recommendation

The next upgrades should add:
- measured **velocity**
- local slope
- nearby gage flow
- stream alignment / thalweg inference
- scoring of nearby candidate points rather than only the current point
"""
    )
else:
    st.info("Enter values in the sidebar, then click **Run Deployment Recommendation**.")