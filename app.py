import streamlit as st
import pandas as pd

from utils.usgs_lookup import build_hydro_context
from utils.hydro_logic import (
    compute_bankfull_metrics,
    recommend_action,
    format_summary_table,
)

st.set_page_config(
    page_title="ARC Hydrokinetic Deployment Advisor",
    page_icon="🌊",
    layout="wide",
)

st.title("ARC Hydrokinetic Deployment Advisor")
st.caption("Debug version with manual drainage area and map disabled.")

with st.sidebar:
    st.header("Inputs")

    lat = st.number_input("Latitude", value=35.430100, format="%.6f")
    lon = st.number_input("Longitude", value=-83.447200, format="%.6f")
    depth_ft = st.number_input(
        "Measured water depth (ft)",
        min_value=0.0,
        value=1.50,
        step=0.1
    )

    st.subheader("Drainage Area")
    st.write("For debugging, enter drainage area manually.")
    drainage_area_sqmi = st.number_input(
        "Drainage area (mi²)",
        min_value=0.0001,
        value=5.0,
        step=0.1,
        format="%.4f",
    )

    run_button = st.button("Run Deployment Recommendation", use_container_width=True)

st.markdown(
    """
This debug version uses:
- manual **latitude**
- manual **longitude**
- manual **depth**
- manual **drainage area**
- Western North Carolina **regional curves**
- **map disabled temporarily**
"""
)

if run_button:
    try:
        with st.spinner("Computing recommendation..."):
            # Optional hydro context lookup for stream name / IDs only
            hydro = build_hydro_context(lat, lon)

            bankfull = compute_bankfull_metrics(drainage_area_sqmi)
            recommendation = recommend_action(depth_ft, bankfull)
            summary_df = format_summary_table(drainage_area_sqmi, bankfull, depth_ft)

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

        with col2:
            st.subheader("Hydro Context")
            st.write(f"**Stream name:** {hydro.stream_name or 'Unknown'}")
            st.write(f"**COMID:** {hydro.comid or 'N/A'}")
            st.write(f"**ReachCode:** {hydro.reachcode or 'N/A'}")
            st.write(f"**Lookup source:** {hydro.source}")
            st.write(f"**Notes:** {hydro.notes}")

            st.subheader("Map")
            st.info("Map temporarily disabled for debugging.")

        st.subheader("Regional-Curve Summary")
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("Interpretation")
        st.markdown(
            """
This version is for debugging stability first.

Once this runs correctly, the next steps are:
- turn the map back on
- optionally re-enable automatic USGS drainage-area lookup
- add manual velocity input
"""
        )

    except Exception as e:
        st.error("The app hit an error.")
        st.code(str(e))

else:
    st.info("Enter values in the sidebar, then click **Run Deployment Recommendation**.")
