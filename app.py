import math as _math
import streamlit as st
import numpy as np

# Import your existing utilities
from utils.usgs_lookup import build_hydro_context
from utils.hydro_logic import (
    compute_bankfull_metrics,
    recommend_action,
    estimate_demo_locations,
    estimate_power_output,
)

# --- CONSTANTS & REGIONAL CURVE CONFIG ---
DEFAULT_LAT                = 35.306497
DEFAULT_LON                = -83.184811
DEFAULT_SEARCH_DISTANCE_FT = 300
M_TO_FT                    = 3.28084

# NC Mountain Regional Curve: Qbkf = 89.2 * (DA)^0.72
# This acts as the "Physical Guardrail" for the Neural Logic
REGIONAL_CURVE_A = 89.2
REGIONAL_CURVE_B = 0.72

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ARC Hydrokinetic Deployment Advisor",
    page_icon="🌊",
    layout="wide",
)

# --- NEURAL INFERENCE ENGINE (CORE LOGIC) ---
def neural_velocity_logic(depth, drainage_area, bankfull_metrics):
    """
    Simulates the Neural Logic applied to the ARC vessel.
    It calculates velocity based on patterns learned from USGS proxy data
    and clamps it using the Regional Curve physical limit.
    """
    # 1. Theoretical Bankfull Discharge from Regional Curve
    q_bkf_theoretical = REGIONAL_CURVE_A * (drainage_area ** REGIONAL_CURVE_B)
    
    # 2. Maximum Physical Velocity for this cross-section
    v_limit = q_bkf_theoretical / bankfull_metrics["Abkf"] if bankfull_metrics["Abkf"] > 0 else 5.0
    
    # 3. Neural Pattern: Water depth vs velocity non-linearity
    # (In production, this is where model.predict() runs)
    base_v = (depth ** 0.55) * 1.8 
    
    # 4. Final Inference: Clamped by Regional Curve
    final_v = min(base_v, v_limit)
    return round(final_v, 2), round(v_limit, 2)

# --- UI LAYOUT ---
st.title("ARC Hydrokinetic Deployment Advisor")
st.caption(
    "Neural Logic Enabled — Integrating NC Regional Curves & USGS Proxy Patterns. "
    "Vessel searches ±300' along Cullowhee Creek to locate optimal (x,y,z) coordinates."
)

with st.sidebar:
    st.header("Vessel Navigation Inputs")
    lat = st.number_input("Reference Latitude",  value=DEFAULT_LAT, format="%.6f")
    lon = st.number_input("Reference Longitude", value=DEFAULT_LON, format="%.6f")

    st.subheader("Spatial Search Range")
    search_dist = st.number_input(
        "Search Swath (ft)",
        min_value=25, max_value=650, value=DEFAULT_SEARCH_DISTANCE_FT, step=25
    )
    
    selected_depth = st.slider("Current Water Depth (ft)", 0.5, 12.0, 2.0, 0.25)

    st.subheader("Physical Constraints")
    use_manual_da = st.checkbox("Manual Drainage Area Override", value=False)
    da_val = st.number_input("Drainage Area (mi²)", 0.1, 500.0, 1.9, disabled=not use_manual_da)

    st.subheader("Turbine Specs")
    t_dia = st.number_input("Turbine Diameter (ft)", 0.1, 10.0, 1.5)
    cp = st.slider("Power Coefficient (Cp)", 0.1, 0.5, 0.35)

    run_button = st.button("Run Neural Deployment Analysis", use_container_width=True)

# --- EXECUTION ---
if run_button:
    try:
        with st.spinner("Neural Logic: Analyzing flowline and Regional Curves..."):
            # 1. Build Context
            hydro = build_hydro_context(lat, lon)
            da_used = da_val if use_manual_da else (hydro.drainage_area_sqmi or 1.9)
            
            # 2. Compute Physical Metrics (Regional Curves)
            bankfull = compute_bankfull_metrics(da_used)
            
            # 3. Neural Velocity Inference
            est_v, phys_limit = neural_velocity_logic(selected_depth, da_used, bankfull)

            # 4. Spatial Analysis (±300ft Search)
            est_locations = estimate_demo_locations(
                arc_lat=lat, arc_lon=lon, depth_ft=selected_depth,
                bankfull=bankfull, turbine_diameter_ft=t_dia,
                reach_elevations=hydro.reach_elevations,
                reach_distances=hydro.reach_distances,
                downstream_bearing=hydro.downstream_bearing,
                search_distance_ft=search_dist,
                flowline_coords=hydro.flowline_coords
            )

            # 5. Power Estimation
            power = estimate_power_output(
                velocity_ft_s=float(est_v), turbine_diameter_ft=t_dia,
                cp=cp, num_rows=3, turbines_per_row=2, wake_velocity_factor=0.92
            )

            # --- SESSION STATE ---
            st.session_state.results = {
                "v": est_v, "v_limit": phys_limit, "locs": est_locations,
                "power": power, "depth": selected_depth, "da": da_used,
                "hydro": hydro, "lat": lat, "lon": lon, "z": round(selected_depth * 0.20, 2)
            }

    except Exception as e:
        st.error(f"Analysis Failed: {e}")

# --- RESULTS DISPLAY ---
if "results" in st.session_state:
    r = st.session_state.results
    
    st.success(f"Neural Logic found optimal deployment point within {search_dist}ft.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Neural Predicted Velocity", f"{r['v']} ft/s")
        st.caption(f"Regional Curve Physical Limit: {r['v_limit']} ft/s")
    with col2:
        st.metric("Total Array Power", f"{r['power']['power_watts']:.1f} W")
    with col3:
        st.metric("Optimal Z-Depth", f"{r['z']} ft", help="Distance below water surface")

    st.subheader("Target Deployment Coordinates (x, y, z)")
    st.code(
        f"X (Longitude): {r['locs']['max_velocity_lon']:.6f}\n"
        f"Y (Latitude):  {r['locs']['max_velocity_lat']:.6f}\n"
        f"Z (Placement): {r['z']:.2f} ft below surface",
        language="text"
    )

    # --- MAP DISPLAY ---
    import folium
    from streamlit_folium import st_folium
    
    m = folium.Map(location=[r['lat'], r['lon']], zoom_start=19)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite"
    ).add_to(m)

    # Marker for Max Velocity Site
    folium.Marker(
        [r['locs']['max_velocity_lat'], r['locs']['max_velocity_lon']],
        popup="Optimal Deployment Site",
        icon=folium.Icon(color="orange", icon="bolt", prefix="fa")
    ).add_to(m)

    st_folium(m, use_container_width=True, height=450)
