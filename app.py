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

DEFAULT_LAT = 35.301845
DEFAULT_LON = -83.183101


st.set_page_config(
    page_title="ARC Hydrokinetic Deployment Advisor",
    page_icon="🌊",
    layout="wide",
)

st.title("ARC Hydrokinetic Deployment Advisor")
st.caption("Stable demo mode with preset water depths from 0.25 ft to 6.50 ft near Cullowhee Creek / Ramsey Center.")

with st.sidebar:
    st.header("Inputs")

    st.write("Default coordinates are preset near Cullowhee Creek by the Ramsey Center area.")
    lat = st.number_input("Latitude", value=DEFAULT_LAT, format="%.6f")
    lon = st.number_input("Longitude", value=DEFAULT_LON, format="%.6f")

    demo_depths = get_demo_depths()
    default_depth = 2.00
    default_index = demo_depths.index(default_depth) if default_depth in demo_depths else len(demo_depths) // 2

    selected_depth = st.selectbox(
        "Select demo water depth (ft)",
        options=demo_depths,
        index=default_index,
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

    st.subheader("Wake Model (Jensen)")
    st.caption(
        "Wake decay coefficient k and η_wake are auto-computed from "
        "regional curve hydraulics (Froude number at bankfull) after running."
    )

    show_debug = st.checkbox("Show raw lookup debug output", value=False)

    run_button = st.button("Run Demo Recommendation", use_container_width=True)

# ── Session state — persists results across map interactions / re-runs ────────
if "results" not in st.session_state:
    st.session_state.results = None

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

            # ── Jensen wake model — auto-computed from regional curve hydraulics ──
            # Froude number at bankfull: Fr = V_bkf / sqrt(g * D_bkf)
            # k = 0.04 + 0.02 * min(Fr, 1.0)  → bounded 0.04–0.06
            # Thrust coefficient Ct derived from Cp via axial induction factor:
            #   Cp = 4a(1-a)²  →  Ct = 4a(1-a)
            import math as _math

            _g_ft_s2   = 32.2
            _v_bkf     = bankfull["Qbkf"] / bankfull["Abkf"] if bankfull["Abkf"] > 0 else 1.0
            _fr        = _v_bkf / _math.sqrt(_g_ft_s2 * bankfull["Dbkf"]) if bankfull["Dbkf"] > 0 else 0.3
            _k_wake    = round(0.04 + 0.02 * min(_fr, 1.0), 4)

            def _solve_induction(cp_val: float) -> float:
                a = 0.25
                for _ in range(50):
                    f  = 4 * a * (1 - a) ** 2 - cp_val
                    fp = 4 * (1 - a) ** 2 - 8 * a * (1 - a)
                    if abs(fp) < 1e-12:
                        break
                    a -= f / fp
                    a  = max(0.01, min(0.49, a))
                return a

            _a_ind  = _solve_induction(cp)
            _ct     = 4 * _a_ind * (1 - _a_ind)
            _r_ft   = turbine_diameter_ft / 2.0
            _x_ft   = 4.0   # station spacing fixed by vessel geometry

            wake_velocity_factor = 1.0 - (1.0 - _math.sqrt(max(0.0, 1.0 - _ct))) * \
                                   (_r_ft / (_r_ft + _k_wake * _x_ft)) ** 2
            wake_velocity_factor = round(max(0.50, min(1.00, wake_velocity_factor)), 4)
            recommendation = recommend_action(selected_depth, bankfull)

            est_velocity = estimate_demo_max_velocity(selected_depth, bankfull)
            est_locations = estimate_demo_locations(
                arc_lat=lat,
                arc_lon=lon,
                depth_ft=selected_depth,
                bankfull=bankfull,
                turbine_diameter_ft=turbine_diameter_ft,
                reach_elevations=hydro.reach_elevations,
                reach_distances=hydro.reach_distances,
                downstream_bearing=hydro.downstream_bearing,
            )
            power = estimate_power_output(
                velocity_ft_s=float(est_velocity["estimated_max_velocity_ft_s"]),
                turbine_diameter_ft=turbine_diameter_ft,
                cp=cp,
                num_rows=3,
                turbines_per_row=2,
                wake_velocity_factor=wake_velocity_factor,
            )

            # ── Persist all results in session state ──────────────────────────
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
                "turbine_diameter_ft":  turbine_diameter_ft,
                "cp":                   cp,
                "wake_velocity_factor": wake_velocity_factor,
                "_fr":                  _fr,
                "_k_wake":              _k_wake,
                "_ct":                  _ct,
                "show_debug":           show_debug,
            }

    except Exception as e:
        st.error("The app hit an error.")
        st.code(str(e))

# ── Display results from session state (survives map re-runs) ─────────────────
if st.session_state.results:
    r                   = st.session_state.results
    hydro               = r["hydro"]
    drainage_area_sqmi  = r["drainage_area_sqmi"]
    drainage_area_source= r["drainage_area_source"]
    bankfull            = r["bankfull"]
    recommendation      = r["recommendation"]
    est_velocity        = r["est_velocity"]
    est_locations       = r["est_locations"]
    power               = r["power"]
    selected_depth      = r["selected_depth"]
    lat                 = r["lat"]
    lon                 = r["lon"]
    turbine_diameter_ft = r["turbine_diameter_ft"]
    cp                  = r["cp"]
    wake_velocity_factor= r["wake_velocity_factor"]
    _fr                 = r["_fr"]
    _k_wake             = r["_k_wake"]
    _ct                 = r["_ct"]
    show_debug          = r["show_debug"]

    z_from_surface = round(selected_depth * 0.20, 2)

    st.success("Demo recommendation computed successfully.")

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
        st.write(f"**COMID:** {hydro.comid or 'N/A'}")
        st.write(f"**ReachCode:** {hydro.reachcode or 'N/A'}")
        st.write(f"**Drainage area used:** {drainage_area_sqmi:.3f} mi²")
        st.write(f"**Drainage area source:** {drainage_area_source}")
        st.write(f"**Hydro lookup notes:** {hydro.notes}")

    # ── Power Output ──────────────────────────────────────────────────────────
    st.subheader("Estimated Max Velocity and Array Power Output")
    out1, out2, out3 = st.columns(3)

    with out1:
        st.write(f"**Selected Demo Depth:** {selected_depth:.2f} ft")
        st.write(f"**Approach Velocity:** {float(est_velocity['estimated_max_velocity_ft_s']):.2f} ft/s")
        st.write(f"**Turbine diameter:** {turbine_diameter_ft:.2f} ft")
        st.write(f"**Cp:** {cp:.2f}")

    with out2:
        st.markdown("**Array:** 3 stations × 2 turbines (port + starboard) = 6 total")
        st.write(f"**Vessel length:** 12 ft | **Beam:** 5 ft | **Station spacing:** 4 ft")
        st.write(f"**Jensen wake model:** Fr={_fr:.3f} → k={_k_wake:.4f}, Ct={_ct:.3f}, η_wake={wake_velocity_factor:.4f}")
        stations = [1, 5, 9]
        for i, (p_w, v_ft_s) in enumerate(
            zip(power["row_powers_watts"], power["row_velocities_ft_s"])
        ):
            st.write(
                f"**Station {stations[i]}'** (port + stbd): "
                f"{v_ft_s:.2f} ft/s → {p_w:.1f} W ({p_w / 1000:.4f} kW)"
            )

    with out3:
        st.metric("Total Array Power", f"{power['power_watts']:.1f} W")
        st.metric("Total Array Power (kW)", f"{power['power_kw']:.4f} kW")
        st.write(
            f"**Single turbine ref:** {power['single_turbine_watts']:.1f} W "
            f"({power['single_turbine_kw']:.4f} kW)"
        )

    # ── Best Deployment Location ──────────────────────────────────────────────
    st.subheader("Estimated Best Deployment Location (x, y, z)")
    loc1, loc2 = st.columns([1, 1])
    with loc1:
        st.code(
            f"x = {est_locations['max_velocity_lon']:.6f}\n"
            f"y = {est_locations['max_velocity_lat']:.6f}\n"
            f"z = {z_from_surface:.2f} ft below surface",
            language="text",
        )
    with loc2:
        st.write(f"**Distance downstream of ARC:** {est_locations['best_candidate_distance_ft']:.0f} ft")
        st.write(f"**Est. depth at best point:** {est_locations['best_candidate_depth_ft']:.2f} ft")
        st.write(f"**Velocity score:** {est_locations['best_candidate_score']:.2f} ft/s")
        st.write(f"**Candidates searched:** {est_locations['candidates_searched']} (every 5 ft over 300 ft)")
        st.caption(f"📡 {est_locations['elev_method']}")

    # ── Deployment map ────────────────────────────────────────────────────────
    st.subheader("Deployment Map")
    try:
        import folium
        from streamlit_folium import st_folium

        m = folium.Map(
            location=[lat, lon],
            zoom_start=17,
            tiles="OpenStreetMap",
        )

        popup_text = (
            f"Best Deployment Point<br>"
            f"{est_locations['best_candidate_distance_ft']:.0f} ft downstream<br>"
            f"Est. depth: {est_locations['best_candidate_depth_ft']:.2f} ft<br>"
            f"z = {z_from_surface:.2f} ft below surface"
        )
        folium.Marker(
            location=[
                est_locations["max_velocity_lat"],
                est_locations["max_velocity_lon"],
            ],
            popup=folium.Popup(popup_text, max_width=250),
            tooltip="Best Deployment Point",
            icon=folium.Icon(color="orange", icon="water", prefix="fa"),
        ).add_to(m)

        folium.Marker(
            location=[lat, lon],
            popup="ARC Vessel Position",
            tooltip="ARC Position",
            icon=folium.Icon(color="blue", icon="ship", prefix="fa"),
        ).add_to(m)

        st_folium(m, use_container_width=True, height=450, returned_objects=[])

    except Exception as map_err:
        st.warning(f"Map could not render: {map_err}")

    if show_debug:
        st.subheader("Raw NLDI tot payload")
        st.code(hydro.debug_nldi_tot_excerpt, language="json")

        st.subheader("Raw StreamStats payload")
        st.code(hydro.debug_streamstats_excerpt, language="json")

else:
    st.info("Default coordinates are already loaded. Choose a demo depth and click Run Demo Recommendation.")
