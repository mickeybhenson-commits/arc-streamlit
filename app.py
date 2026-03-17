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

DEFAULT_LAT = 35.302740
DEFAULT_LON = -83.183447


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
    k_wake = st.number_input(
        "Wake decay coefficient, k",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%.2f",
        help=(
            "Jensen wake decay constant for open-channel flow. "
            "0.04–0.06 is standard for water turbine arrays. "
            "Higher k = faster wake recovery."
        ),
    )

    # ── Jensen wake model ─────────────────────────────────────────────────────
    # Step 1: solve Cp = 4a(1-a)² for axial induction factor a (Newton-Raphson)
    import math as _math
    def _solve_induction(cp_val: float) -> float:
        a = 0.25
        for _ in range(50):
            f  =  4 * a * (1 - a) ** 2 - cp_val
            fp =  4 * (1 - a) ** 2 - 8 * a * (1 - a)
            if abs(fp) < 1e-12:
                break
            a -= f / fp
            a  = max(0.01, min(0.49, a))
        return a

    a_ind = _solve_induction(cp)
    ct    = 4 * a_ind * (1 - a_ind)           # thrust coefficient
    r_ft  = turbine_diameter_ft / 2.0          # rotor radius (ft)
    x_ft  = 4.0                                # station spacing (ft) — fixed by vessel

    # Jensen velocity ratio: V_wake/V_0 = 1 - (1 - √(1-Ct)) · (r / (r + k·x))²
    wake_velocity_factor = 1.0 - (1.0 - _math.sqrt(max(0.0, 1.0 - ct))) * (r_ft / (r_ft + k_wake * x_ft)) ** 2
    wake_velocity_factor = round(max(0.50, min(1.00, wake_velocity_factor)), 4)

    st.caption(
        f"Cp={cp:.2f} → a={a_ind:.3f}, Ct={ct:.3f}\n\n"
        f"**Computed η_wake = {wake_velocity_factor:.4f}**\n\n"
        f"Station 5' power ≈ {wake_velocity_factor**3:.2%} of Station 1'\n\n"
        f"Station 9' power ≈ {wake_velocity_factor**6:.2%} of Station 1'"
    )

    show_debug = st.checkbox("Show raw lookup debug output", value=False)

    run_button = st.button("Run Demo Recommendation", use_container_width=True)



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
                velocity_ft_s=float(est_velocity["estimated_max_velocity_ft_s"]),
                turbine_diameter_ft=turbine_diameter_ft,
                cp=cp,
                num_rows=3,
                turbines_per_row=2,
                wake_velocity_factor=wake_velocity_factor,
            )

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

        # ── Power Output ──────────────────────────────────────────────────────
        st.subheader("Estimated Max Velocity and Array Power Output")
        out1, out2, out3 = st.columns(3)

        # z from water surface: max velocity ~20% of depth below surface
        # per log-law open channel velocity profile (standard hydraulics)
        z_from_surface = round(selected_depth * 0.20, 2)

        with out1:
            st.write(f"**Selected Demo Depth:** {selected_depth:.2f} ft")
            st.write(f"**Approach Velocity:** {float(est_velocity['estimated_max_velocity_ft_s']):.2f} ft/s")
            st.write(f"**Turbine diameter:** {turbine_diameter_ft:.2f} ft")
            st.write(f"**Cp:** {cp:.2f}")

        with out2:
            st.markdown("**Array:** 3 stations × 2 turbines (port + starboard) = 6 total")
            st.write(f"**Vessel length:** 12 ft | **Beam:** 5 ft | **Station spacing:** 4 ft")
            st.write(f"**Wake velocity factor:** {wake_velocity_factor:.2f}")
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

        # ── Max velocity XYZ ──────────────────────────────────────────────────
        st.subheader("Estimated Max Velocity Point (x, y, z)")
        st.code(
            f"x = {est_locations['max_velocity_lon']:.6f}\n"
            f"y = {est_locations['max_velocity_lat']:.6f}\n"
            f"z = {z_from_surface:.2f} ft below surface",
            language="text",
        )

        # ── Deployment map ────────────────────────────────────────────────────
        st.subheader("Deployment Map")
        try:
            import pydeck as pdk
            import os

            MAPBOX_TOKEN = "pk.eyJ1IjoibWJoZW5zb24xOTQ1IiwiYSI6ImNtbXRrM2owaTFzencycnB5dHFvN2J5dW8ifQ.OG5D5ZFCg9XroLargRIGMg"
            os.environ["MAPBOX_API_KEY"] = MAPBOX_TOKEN

            map_points = [
                {
                    "lat":   est_locations["max_velocity_lat"],
                    "lon":   est_locations["max_velocity_lon"],
                    "label": f"Max Velocity Point  z={z_from_surface:.2f} ft below surface",
                    "color": [255, 140, 0, 230],   # orange
                    "radius": 4,
                },
            ]

            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_points,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
                radius_min_pixels=2,
                radius_max_pixels=6,
                pickable=True,
            )

            view = pdk.ViewState(
                latitude=lat,
                longitude=lon,
                zoom=19,
                pitch=0,
                bearing=0,
            )

            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/satellite-streets-v12",
                    initial_view_state=view,
                    layers=[scatter_layer],
                    tooltip={"text": "{label}"},
                ),
                use_container_width=True,
            )

            st.markdown(
                '<span style="color:#FF8C00;font-weight:700;">●</span>'
                "&nbsp; Max Velocity Point",
                unsafe_allow_html=True,
            )

        except Exception as map_err:
            st.warning(f"Map could not render: {map_err}")

        if show_debug:
            st.subheader("Raw NLDI tot payload")
            st.code(hydro.debug_nldi_tot_excerpt, language="json")

            st.subheader("Raw StreamStats payload")
            st.code(hydro.debug_streamstats_excerpt, language="json")

    except Exception as e:
        st.error("The app hit an error.")
        st.code(str(e))

else:
    st.info("Default coordinates are already loaded. Choose a demo depth and click Run Demo Recommendation.")
