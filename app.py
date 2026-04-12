import math as _math
import streamlit as st
import numpy as np

from utils.usgs_lookup import build_hydro_context
from utils.hydro_logic import (
    compute_bankfull_metrics,
    estimate_demo_locations,
    estimate_power_output,
    run_neural_velocity_inference
)

# --- CONFIG ---
DEFAULT_LAT = 35.306497
DEFAULT_LON = -83.184811
DEFAULT_SEARCH_DISTANCE_FT = 300

st.set_page_config(page_title="ARC Deployment Advisor", page_icon="🌊", layout="wide")

st.title("ARC Hydrokinetic Deployment Advisor")
st.caption("Neural Logic Enabled | 4-Turbine Array (2x2) | NC Regional Curve Constraints")

with st.sidebar:
    st.header("Vessel Inputs")
    lat = st.number_input("Ref Latitude", value=DEFAULT_LAT, format="%.6f")
    lon = st.number_input("Ref Longitude", value=DEFAULT_LON, format="%.6f")
    search_dist = st.number_input("Search Range (ft)", value=DEFAULT_SEARCH_DISTANCE_FT, step=25)
    selected_depth = st.slider("Water Depth (ft)", 0.5, 12.0, 2.0, 0.25)
    
    st.subheader("Physical Constraints")
    use_manual_da = st.checkbox("Manual DA Override")
    da_val = st.number_input("Drainage Area (mi²)", value=1.9, disabled=not use_manual_da)
    
    st.subheader("4-Turbine Array Settings")
    t_dia = st.number_input("Turbine Diameter (ft)", value=1.5)
    cp = st.slider("Power Coefficient (Cp)", 0.1, 0.5, 0.35)
    run_button = st.button("Run Neural Analysis", use_container_width=True)

if run_button:
    try:
        with st.spinner("Analyzing Reach and Flow Patterns..."):
            hydro = build_hydro_context(lat, lon)
            da_used = da_val if use_manual_da else (hydro.drainage_area_sqmi or 1.9)
            bankfull = compute_bankfull_metrics(da_used)
            
            # Neural Search Logic
            est_locations = estimate_demo_locations(
                arc_lat=lat, arc_lon=lon, depth_ft=selected_depth,
                bankfull=bankfull, turbine_diameter_ft=t_dia,
                reach_elevations=hydro.reach_elevations,
                reach_distances=hydro.reach_distances,
                downstream_bearing=hydro.downstream_bearing,
                search_distance_ft=search_dist,
                flowline_coords=hydro.flowline_coords
            )
            
            # Power Analysis for 2x2 Array
            power = estimate_power_output(
                velocity_ft_s=est_locations["best_candidate_score"],
                turbine_diameter_ft=t_dia, cp=cp
            )

            st.session_state.results = {
                "v": est_locations["best_candidate_score"],
                "locs": est_locations, "power": power, "hydro": hydro,
                "z": round(selected_depth * 0.20, 2), "da": da_used
            }
    except Exception as e:
        st.error(f"Error: {e}")

if "results" in st.session_state:
    r = st.session_state.results
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Neural Velocity", f"{r['v']:.2f} ft/s")
        st.write(f"**Drainage Area:** {r['da']:.2f} mi²")
    with col2:
        st.metric("Total Array Power (4 Turbines)", f"{r['power']['power_watts']:.1f} W")
        st.write(f"Station 1: {r['power']['station_1_watts']:.1f}W")
        st.write(f"Station 2: {r['power']['station_2_watts']:.1f}W")
    with col3:
        st.subheader("Target (x, y, z)")
        st.code(f"x: {r['locs']['max_velocity_lon']:.6f}\ny: {r['locs']['max_velocity_lat']:.6f}\nz: {r['z']} ft", language="text")
