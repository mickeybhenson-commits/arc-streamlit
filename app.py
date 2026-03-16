import streamlit as st

from utils.usgs_lookup import build_hydro_context
from utils.hydro_logic import (
    compute_bankfull_metrics,
    recommend_action,
    format_summary_table,
    get_demo_depths,
    estimate_demo_max_velocity,
    estimate_demo_locations,
    estimate_power_output,
    build_demo_scenario_table,
)
from utils.mapping import build_demo_map


DEFAULT_LAT = 35.311900
DEFAULT_LON = -83.181000


st.set_page_config(
    page_title="ARC Hydrokinetic Deployment Advisor",
    page_icon="🌊",
    layout="wide",
)

st.title("ARC Hydrokinetic Deployment Advisor")
st.caption("Demo mode with 25 preset water depths around 2 feet near Cullowhee Creek / Ramsey Center.")

with st.sidebar:
    st.header("Inputs")

    st.write("Default coordinates are preset near Cullowhee Creek by the Ramsey Center area.")
    lat = st.number_input("Latitude", value=DEFAULT_LAT, format="%.6f")
    lon = st.number_input("Longitude", value=DEFAULT_LON, format="%.6f")

    demo_depths = get_demo_depths()
    selected_depth = st.selectbox(
        "Select demo water depth (ft)",
        options=demo_depths,
        index=demo_depths.index(2.0),
        format_func=lambda x: f"{x:.2f} ft",
    )

    st.subheader("Drainage Area Options")
    use_manual_da = st.checkbox("Override with manual drainage area", value=False)
    drainage_area_manual = st.number_input(
        "Manual drainage area (mi²)",
        min_value=0.0001,
        value=1.9,
        step=0.1,
        format="%.4f",
        disabled=not use_manual_da,
    )

    st.subheader("Demo Turbine Settings")
    turbine_diameter_ft = st.number_input(
        "Turbine diameter (ft)",
        min_value=0.1,
        value=1.5,
        step=0.1,
    )
    cp = st.number_input(
        "Power coefficient, Cp",
        min_value=0.01,
        max_value=0.90,
        value=0.35,
        step=0.01,
        format="%.2f",
    )

    show_debug = st.checkbox("Show raw lookup debug output", value=False)

    run_button = st.button("Run Demo Recommendation", use_container_width=True)

st.markdown(
    """
This demo version uses:
- default coordinates near **Cullowhee Creek / Ramsey Center**
- **25 preset water depths around 2 feet**
- **automatic drainage area from coordinates** when available
- manual drainage-area fallback
- Western North Carolina **regional curves**
- **estimated** max velocity point, velocity, and power
"""
)

if run_button:
    try:
        with st.spinner("Computing demo recommendation..."):
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
                st.write("Hydro lookup notes:")
                st.code(hydro.notes)

                if show_debug:
                    st.subheader("Raw NLDI tot payload")
                    st.code(hydro.debug_nldi_tot_excerpt, language="json")

                    st.subheader("Raw StreamStats payload")
                    st.code(hydro.debug_streamstats_excerpt, language="json")

                st.stop()

            bankfull = compute_bankfull_metrics(drainage_area_sqmi)
            recommendation = recommend_action(selected_depth, bankfull)

            est_velocity = estimate_demo_max_velocity(selected_depth, bankfull)
            est_locations = estimate_demo_locations(
                arc_lat=lat,
                arc_lon=lon,
                depth_ft=selected_depth,
                bankfull=bankfull,
            )
            power = estimate_power_output(
                velocity_ft_s=est_velocity["estimated_max_velocity_ft_s"],
                turbine_diameter_ft=turbine_diameter_ft,
                cp=cp,
            )

            summary_df = format_summary_table(drainage_area_sqmi, bankfull, selected_depth)
            scenarios_df = build_demo_scenario_table(
                lat=lat,
                lon=lon,
                depths_ft=demo_depths,
                bankfull=bankfull,
                turbine_diameter_ft=turbine_diameter_ft,
                cp=cp,
            )

            demo_map = build_demo_map(
                arc_lat=lat,
                arc_lon=lon,
                max_lat=est_locations["max_velocity_lat"],
                max_lon=est_locations["max_velocity_lon"],
                deploy_lat=est_locations["deployment_lat"],
                deploy_lon=est_locations["deployment_lon"],
                selected_depth_ft=selected_depth,
                estimated_max_velocity_ft_s=est_velocity["estimated_max_velocity_ft_s"],
            )

        top1, top2 = st.columns([1, 1])

        with top1:
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

        with top2:
            st.subheader("Hydro Context")
            st.write(f"**Latitude used:** {lat:.6f}")
            st.write(f"**Longitude used:** {lon:.6f}")
            st.write(f"**Stream name:** {hydro.stream_name or 'Unknown'}")
            st.write(f"**COMID:** {hydro.comid or 'N/A'}")
            st.write(f"**ReachCode:** {hydro.reachcode or 'N/A'}")
            st.write(f"**Drainage area used:** {drainage_area_sqmi:.3f} mi²")
            st.write(f"**Drainage area source:** {drainage_area_source}")
            st.write(f"**Hydro lookup notes:** {hydro.notes}")

        st.subheader("Estimated Max Velocity and Deployment Outputs")
        out1, out2, out3 = st.columns(3)

        with out1:
            st.metric("Selected Demo Depth", f"{selected_depth:.2f} ft")
            st.metric("Estimated Max Velocity", f"{est_velocity['estimated_max_velocity_ft_s']:.2f} ft/s")
            st.write(f"**Confidence:** {est_velocity['confidence_label']}")
            st.write(est_velocity["note"])

        with out2:
            st.metric("Estimated Power", f"{power['power_watts']:.1f} W")
            st.metric("Estimated Power", f"{power['power_kw']:.4f} kW")
            st.write(f"**Turbine diameter:** {turbine_diameter_ft:.2f} ft")
            st.write(f"**Cp:** {cp:.2f}")

        with out3:
            st.write("**Estimated max velocity point (x, y, z)**")
            st.code(
                f"x = {est_locations['max_velocity_lon']:.6f}\n"
                f"y = {est_locations['max_velocity_lat']:.6f}\n"
                f"z = {selected_depth:.2f} ft",
                language="text",
            )
            st.write("**Estimated ARC deployment point**")
            st.code(
                f"x = {est_locations['deployment_lon']:.6f}\n"
                f"y = {est_locations['deployment_lat']:.6f}\n"
                f"z = {selected_depth:.2f} ft",
                language="text",
            )

        st.subheader("Map: ARC vs Estimated Max Velocity Point vs Deployment Point")
        try:
            from streamlit_folium import st_folium
            st_folium(demo_map, width=None, height=520)
        except Exception as map_error:
            st.error(f"Map error: {map_error}")

        st.subheader("Regional-Curve Summary")
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("All 25 Demo Depth Scenarios")
        st.dataframe(scenarios_df, use_container_width=True)

        if show_debug:
            st.subheader("Raw NLDI tot payload")
            st.code(hydro.debug_nldi_tot_excerpt, language="json")

            st.subheader("Raw StreamStats payload")
            st.code(hydro.debug_streamstats_excerpt, language="json")

    except Exception as e:
        st.error("The app hit an error.")
        st.code(str(e))

else:
    st.info("Default coordinates are already loaded. Choose one of the 25 demo depths and click Run Demo Recommendation.")
