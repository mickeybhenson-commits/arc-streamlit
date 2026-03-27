import math as _math

import streamlit as st

from utils.usgs_lookup import build_hydro_context
from utils.hydro_logic import (
    compute_bankfull_metrics,
    recommend_action,
    format_summary_table,
    estimate_demo_max_velocity,
    estimate_demo_locations,
    estimate_power_output,
    build_demo_scenario_table,
)

DEFAULT_LAT                = 35.306497
DEFAULT_LON                = -83.184811
DEFAULT_SEARCH_DISTANCE_FT = 300
M_TO_FT                    = 3.28084


st.set_page_config(
    page_title="ARC Hydrokinetic Deployment Advisor",
    page_icon="🌊",
    layout="wide",
)

st.title("ARC Hydrokinetic Deployment Advisor")
st.caption(
    "Stable demo mode — water depths 0.50–12.00 ft. "
    "Reference point on Cullowhee Creek. Vessel searches both upstream AND downstream "
    "along the NHDPlus flowline centerline to locate the maximum velocity site."
)

with st.sidebar:
    st.header("Inputs")

    st.write(
        "**Reference point** on Cullowhee Creek. "
        "The vessel searches both upstream and downstream from this point "
        "to find the maximum velocity deployment site."
    )
    lat = st.number_input("Reference Point Latitude",  value=DEFAULT_LAT, format="%.6f")
    lon = st.number_input("Reference Point Longitude", value=DEFAULT_LON, format="%.6f")

    st.subheader("Search Range")
    search_distance_ft = st.number_input(
        "Search distance upstream + downstream (ft)",
        min_value=25,
        max_value=650,
        value=DEFAULT_SEARCH_DISTANCE_FT,
        step=25,
    )
    search_distance_m = round(search_distance_ft / M_TO_FT, 1)
    st.caption(
        f"≈ {search_distance_m:.0f} m in each direction — "
        f"vessel evaluates candidates every 5 ft along flowline "
        f"(±{search_distance_ft:.0f} ft from reference point)"
    )

    demo_depths   = [round(0.50 + i * 0.25, 2) for i in range(int((12.00 - 0.50) / 0.25) + 1)]
    default_depth = 2.00
    default_index = demo_depths.index(default_depth) if default_depth in demo_depths else len(demo_depths) // 2

    selected_depth = st.selectbox(
        "Select demo water depth (ft)",
        options=demo_depths,
        index=default_index,
        format_func=lambda x: f"{x:.2f} ft",
    )

    st.subheader("Drainage Area Options")
    use_manual_da        = st.checkbox("Override with manual drainage area", value=False)
    drainage_area_manual = st.number_input(
        "Manual drainage area (mi²)",
        min_value=0.0001, value=1.9, step=0.1, format="%.4f",
        disabled=not use_manual_da,
    )

    st.subheader("Demo Turbine Settings")
    turbine_diameter_ft = st.number_input("Turbine diameter (ft)", min_value=0.1, value=1.5, step=0.1)
    cp = st.number_input("Power coefficient, Cp", min_value=0.01, max_value=0.90,
                         value=0.35, step=0.01, format="%.2f")

    st.subheader("Wake Model (Jensen)")
    st.caption("k and η_wake auto-computed from bankfull Froude number after running.")

    show_debug = st.checkbox("Show raw lookup debug output", value=False)
    run_button = st.button("Run Demo Recommendation", use_container_width=True)


if "results" not in st.session_state:
    st.session_state.results = None

if run_button:
    try:
        with st.spinner("Computing demo recommendation..."):
            hydro = build_hydro_context(lat, lon)

            if use_manual_da:
                drainage_area_sqmi   = drainage_area_manual
                drainage_area_source = "Manual override"
            else:
                drainage_area_sqmi   = hydro.drainage_area_sqmi
                drainage_area_source = hydro.source

            if drainage_area_sqmi is None:
                st.error(
                    "Automatic drainage-area lookup failed. "
                    "Check 'Override with manual drainage area' and enter a value."
                )
                st.code(hydro.notes)
                if show_debug:
                    st.code(hydro.debug_nldi_tot_excerpt, language="json")
                    st.code(hydro.debug_streamstats_excerpt, language="json")
                st.stop()

            bankfull = compute_bankfull_metrics(drainage_area_sqmi)

            _g_ft_s2 = 32.2
            _v_bkf   = bankfull["Qbkf"] / bankfull["Abkf"] if bankfull["Abkf"] > 0 else 1.0
            _fr      = _v_bkf / _math.sqrt(_g_ft_s2 * bankfull["Dbkf"]) if bankfull["Dbkf"] > 0 else 0.3
            _k_wake  = round(0.04 + 0.02 * min(_fr, 1.0), 4)

            def _solve_induction(cp_val):
                a = 0.25
                for _ in range(50):
                    f  = 4*a*(1-a)**2 - cp_val
                    fp = 4*(1-a)**2 - 8*a*(1-a)
                    if abs(fp) < 1e-12: break
                    a -= f / fp
                    a  = max(0.01, min(0.49, a))
                return a

            _a_ind = _solve_induction(cp)
            _ct    = 4 * _a_ind * (1 - _a_ind)
            _r_ft  = turbine_diameter_ft / 2.0

            wake_velocity_factor = 1.0 - (1.0 - _math.sqrt(max(0.0, 1.0 - _ct))) * \
                                   (_r_ft / (_r_ft + _k_wake * 4.0))**2
            wake_velocity_factor = round(max(0.50, min(1.00, wake_velocity_factor)), 4)

            recommendation = recommend_action(selected_depth, bankfull)
            est_velocity   = estimate_demo_max_velocity(selected_depth, bankfull)

            est_locations = estimate_demo_locations(
                arc_lat            = lat,
                arc_lon            = lon,
                depth_ft           = selected_depth,
                bankfull           = bankfull,
                turbine_diameter_ft= turbine_diameter_ft,
                reach_elevations   = hydro.reach_elevations,
                reach_distances    = hydro.reach_distances,
                downstream_bearing = hydro.downstream_bearing,
                search_distance_ft = search_distance_ft,
                flowline_coords    = hydro.flowline_coords,
            )

            power = estimate_power_output(
                velocity_ft_s       = float(est_velocity["estimated_max_velocity_ft_s"]),
                turbine_diameter_ft = turbine_diameter_ft,
                cp                  = cp,
                num_rows            = 3,
                turbines_per_row    = 2,
                wake_velocity_factor= wake_velocity_factor,
            )

            deploy_lat     = est_locations["max_velocity_lat"]
            deploy_lon     = est_locations["max_velocity_lon"]
            z_from_surface = round(selected_depth * 0.20, 2)

            st.session_state.results = {
                "hydro":                hydro,
                "drainage_area_sqmi":   drainage_area_sqmi,
                "drainage_area_source": drainage_area_source,
                "bankfull":             bankfull,
                "recommendation":       recommendation,
                "est_velocity":         est_velocity,
                "est_locations":        est_locations,
                "power":                power,
                "selected_depth":       selected_depth,
                "lat":                  lat,
                "lon":                  lon,
                "deploy_lat":           deploy_lat,
                "deploy_lon":           deploy_lon,
                "z_from_surface":       z_from_surface,
                "turbine_diameter_ft":  turbine_diameter_ft,
                "cp":                   cp,
                "wake_velocity_factor": wake_velocity_factor,
                "_fr": _fr, "_k_wake": _k_wake, "_ct": _ct,
                "show_debug":           show_debug,
                "search_distance_m":    search_distance_m,
                "search_distance_ft":   search_distance_ft,
            }

    except Exception as e:
        st.error("The app hit an error.")
        st.code(str(e))


if st.session_state.results:
    r                    = st.session_state.results
    hydro                = r["hydro"]
    drainage_area_sqmi   = r["drainage_area_sqmi"]
    drainage_area_source = r["drainage_area_source"]
    bankfull             = r["bankfull"]
    recommendation       = r["recommendation"]
    est_velocity         = r["est_velocity"]
    est_locations        = r["est_locations"]
    power                = r["power"]
    selected_depth       = r["selected_depth"]
    lat                  = r["lat"]
    lon                  = r["lon"]
    deploy_lat           = r["deploy_lat"]
    deploy_lon           = r["deploy_lon"]
    z_from_surface       = r["z_from_surface"]
    turbine_diameter_ft  = r["turbine_diameter_ft"]
    cp                   = r["cp"]
    wake_velocity_factor = r["wake_velocity_factor"]
    _fr                  = r["_fr"]
    _k_wake              = r["_k_wake"]
    _ct                  = r["_ct"]
    show_debug           = r["show_debug"]
    search_distance_m    = r["search_distance_m"]
    search_distance_ft   = r["search_distance_ft"]

    used_flowline = est_locations.get("used_flowline", False)
    best_dir      = est_locations.get("best_candidate_direction", "unknown")
    best_dist     = abs(est_locations["best_candidate_distance_ft"])

    st.success("Demo recommendation computed successfully.")
    if used_flowline:
        st.info(
            f"✅ Deployment location computed from **NHDPlus flowline geometry** — "
            f"candidates on stream centerline. "
            f"Best site found **{best_dist:.0f} ft {best_dir}** of reference point."
        )
    else:
        st.warning(
            f"⚠️ Flowline geometry unavailable — bearing projection used. "
            f"Best site found **{best_dist:.0f} ft {best_dir}** of reference point. "
            "Verify deployment marker is in channel."
        )

    top1, top2 = st.columns([1, 1])

    with top1:
        st.subheader("Recommendation")
        dv = recommendation["deploy"]
        if dv == "YES":   st.success(f"Deploy now? {dv}")
        elif dv == "MAYBE": st.warning(f"Deploy now? {dv}")
        else:               st.error(f"Deploy now? {dv}")
        st.write(f"**Action:** {recommendation['action']}")
        st.write(f"**Score:** {recommendation['score']}")
        st.write(f"**Reason:** {recommendation['reason']}")
        st.write(f"**ARC navigation note:** {recommendation['nav_note']}")

    with top2:
        st.subheader("Hydro Context")
        st.write(f"**Reference point (lat, lon):** {lat:.6f}, {lon:.6f}")
        st.write(f"**Search range:** ±{search_distance_ft:.0f} ft ({search_distance_m:.0f} m each direction)")
        st.write(f"**COMID:** {hydro.comid or 'N/A'}")
        st.write(f"**ReachCode:** {hydro.reachcode or 'N/A'}")
        st.write(f"**Stream name:** {hydro.stream_name or 'N/A'}")
        st.write(f"**Downstream bearing:** {hydro.downstream_bearing:.1f}°")
        st.write(f"**Flowline vertices:** {len(hydro.flowline_coords) if hydro.flowline_coords else 'N/A'}")
        st.write(f"**Drainage area used:** {drainage_area_sqmi:.3f} mi²")
        st.write(f"**Drainage area source:** {drainage_area_source}")
        st.write(f"**Notes:** {hydro.notes}")

    st.subheader("Estimated Max Velocity and Array Power Output")
    out1, out2, out3 = st.columns(3)

    with out1:
        st.write(f"**Selected Demo Depth:** {selected_depth:.2f} ft")
        st.write(f"**Approach Velocity:** {float(est_velocity['estimated_max_velocity_ft_s']):.2f} ft/s")
        st.write(f"**Velocity confidence:** {est_velocity['confidence_label']}")
        st.write(f"**Turbine diameter:** {turbine_diameter_ft:.2f} ft")
        st.write(f"**Cp:** {cp:.2f}")

    with out2:
        st.markdown("**Array:** 3 stations × 2 turbines (port + starboard) = 6 total")
        st.write(f"**Vessel:** 12 ft × 5 ft | Station spacing: 4 ft")
        st.write(f"**Jensen:** Fr={_fr:.3f} → k={_k_wake:.4f}, Ct={_ct:.3f}, η_wake={wake_velocity_factor:.4f}")
        for i, (p_w, v_ft_s) in enumerate(zip(power["row_powers_watts"], power["row_velocities_ft_s"])):
            st.write(f"**Station {[1,5,9][i]}'**: {v_ft_s:.2f} ft/s → {p_w:.1f} W ({p_w/1000:.4f} kW)")

    with out3:
        st.metric("Total Array Power", f"{power['power_watts']:.1f} W")
        st.metric("Total Array Power (kW)", f"{power['power_kw']:.4f} kW")
        st.write(f"**Single turbine ref:** {power['single_turbine_watts']:.1f} W ({power['single_turbine_kw']:.4f} kW)")

    st.subheader("Estimated Best Deployment Location (x, y, z) — Stream Centerline")
    loc1, loc2 = st.columns([1, 1])
    with loc1:
        st.code(
            f"x = {deploy_lon:.6f}\n"
            f"y = {deploy_lat:.6f}\n"
            f"z = {z_from_surface:.2f} ft below surface",
            language="text",
        )
    with loc2:
        st.write(f"**Direction from reference:** {best_dir.upper()} {best_dist:.0f} ft")
        st.write(f"**Est. depth at best point:** {est_locations['best_candidate_depth_ft']:.2f} ft")
        st.write(f"**Velocity score:** {est_locations['best_candidate_score']:.2f} ft/s")
        st.write(f"**Candidates searched:** {est_locations['candidates_searched']} (every 5 ft over ±{search_distance_ft:.0f} ft)")
        st.write(f"**Position method:** {'NHDPlus flowline ✅' if used_flowline else 'Bearing projection ⚠️'}")
        st.caption(f"📡 {est_locations['elev_method']}")

    st.subheader("Deployment Map")
    st.caption("💡 Hover for live lat/lon (bottom-left). Ruler tool top-left. Orange = max velocity site.")
    try:
        import folium
        import folium.plugins as plugins
        from folium.plugins import MousePosition, MeasureControl
        from streamlit_folium import st_folium

        m = folium.Map(
            location=[lat, lon], zoom_start=19,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
        )
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
            attr="Esri Labels", name="Labels", overlay=True, opacity=0.7,
        ).add_to(m)

        MousePosition(
            position="bottomleft", separator=" | ", prefix="Lat/Lon:",
            lat_formatter="function(num) {return L.Util.formatNum(num, 6);}",
            lng_formatter="function(num) {return L.Util.formatNum(num, 6);}",
        ).add_to(m)
        MeasureControl(position="topleft", primary_length_unit="feet",
                       secondary_length_unit="meters").add_to(m)
        plugins.Fullscreen(position="topright").add_to(m)
        m.get_root().html.add_child(folium.Element("""
        <div style="position:fixed;top:10px;right:50px;z-index:9999;background:white;
            border-radius:50%;width:36px;height:36px;text-align:center;line-height:36px;
            font-size:20px;box-shadow:2px 2px 5px rgba(0,0,0,0.4);pointer-events:none;">🧭</div>
        """))

        # NHDPlus flowline hidden from map — medium-resolution geometry
        # (1:100k scale) does not align with high-resolution satellite imagery.
        # Flowline is still used internally for all hydraulic calculations.

        # Best deployment point
        folium.Marker(
            location=[deploy_lat, deploy_lon],
            popup=folium.Popup(
                f"Max Velocity Site<br>"
                f"{best_dist:.0f} ft {best_dir} of reference<br>"
                f"Lat: {deploy_lat:.6f} | Lon: {deploy_lon:.6f}<br>"
                f"Est. depth: {est_locations['best_candidate_depth_ft']:.2f} ft<br>"
                f"z = {z_from_surface:.2f} ft below surface",
                max_width=280,
            ),
            tooltip=f"Max Velocity — {best_dist:.0f} ft {best_dir}",
            icon=folium.Icon(color="orange", icon="water", prefix="fa"),
        ).add_to(m)

        # Reference point
        folium.Marker(
            location=[lat, lon],
            popup=f"Reference Point — searching ±{search_distance_ft:.0f} ft",
            tooltip="Reference Point",
            icon=folium.Icon(color="green", icon="flag", prefix="fa"),
        ).add_to(m)

        st_folium(m, use_container_width=True, height=500, returned_objects=[])

    except Exception as map_err:
        st.warning(f"Map could not render: {map_err}")

    if show_debug:
        st.subheader("Raw NLDI tot payload")
        st.code(hydro.debug_nldi_tot_excerpt, language="json")
        st.subheader("Raw StreamStats payload")
        st.code(hydro.debug_streamstats_excerpt, language="json")

else:
    st.info("Reference point loaded. Choose a demo depth and click Run Demo Recommendation.")
