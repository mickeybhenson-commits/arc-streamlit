import streamlit as st

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
st.caption("Version with automatic drainage area from coordinates and manual fallback.")

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

    st.subheader("Drainage Area Options")
    use_manual_da = st.checkbox("Override with manual drainage area", value=False)

    drainage_area_manual = st.number_input(
        "Manual drainage area (mi²)",
        min_value=0.0001,
        value=5.0,
        step=0.1,
        format="%.4f",
        disabled=not use_manual_da,
    )

    run_button = st.button("Run Deployment Recommendation", use_container_width=True)

st.markdown(
    """
This version uses:
- manual **latitude**
- manual **longitude**
- manual **depth**
- **automatic drainage area from USGS** when available
- manual drainage-area fallback
- Western North Carolina **regional curves**
- **map disabled temporarily**
"""
)

if run_button:
    try:
        with st.spinner("Computing recommendation..."):
            hydro = build_hydro_context(lat, lon)

            if use_manual_da:
                drainage_area_sqmi = drainage_area_manual
                drainage_area_source = "Manual override"
            else:
                drainage_area_sqmi = hydro.drainage_area_sqmi
                drainage_area_source = hydro.source

            if drainage_area_sqmi is None:
                st.error(
                    "Automatic drainage-area lookup failed for this point. "
                    "Check 'Override with manual drainage area' and enter a value."
                )
                st.stop()

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
            st.write(f"**Drainage area used:** {drainage_area_sqmi:.3f} mi²")
            st.write(f"**Drainage area source:** {drainage_area_source}")
            st.write(f"**Hydro lookup notes:** {hydro.notes}")

            st.subheader("Map")
            st.info("Map temporarily disabled while drainage-area lookup is stabilized.")

        st.subheader("Regional-Curve Summary")
        st.dataframe(summary_df, use_container_width=True)

    except Exception as e:
        st.error("The app hit an error.")
        st.code(str(e))

else:
    st.info("Enter values in the sidebar, then click **Run Deployment Recommendation**.")
