"""
NEMO Bathymetric Survey Analyzer
---------------------------------
Accepts Recon ASV depth readings (every 5 ft over 300 ft of stream length)
and produces a full hydraulic analysis:

  - Longitudinal bed elevation profile
  - Water depth profile
  - Manning's velocity at each point (n=0.045, post-Helene WNC calibrated)
  - Best deployment location (max Manning's velocity)
  - Folium map with ARC start, all survey points, and best deployment flagged
  - Exportable CSV of all computed hydraulics

Input format: CSV or manual entry
  Columns: distance_ft, depth_ft, lat, lon
"""

from __future__ import annotations

import io
import math
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="NEMO Bathymetric Survey",
    page_icon="🛥️",
    layout="wide",
)

st.title("🛥️ NEMO Bathymetric Survey Analyzer")
st.caption("Recon ASV depth data → hydraulic analysis → optimal deployment location")

# ── Constants ─────────────────────────────────────────────────────────────────
N_MANNING    = 0.045   # post-Helene WNC calibrated roughness (Henson et al.)
MIN_DEPTH_FT = 0.10    # ignore readings shallower than this

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Survey Settings")

    wse_arc = st.number_input(
        "Water surface elevation at ARC start (ft, NAVD88 or relative)",
        value=0.0,
        step=0.1,
        format="%.2f",
        help=(
            "Enter NAVD88 elevation if known. "
            "If unknown, enter 0 — all bed elevations will be relative."
        ),
    )

    turbine_diameter_ft = st.number_input(
        "Turbine diameter (ft)",
        min_value=0.1,
        value=1.5,
        step=0.1,
    )
    min_deploy_depth = max(turbine_diameter_ft * 1.2, 0.5)

    arc_lat = st.number_input(
        "ARC start latitude",
        value=35.301819,
        format="%.6f",
    )
    arc_lon = st.number_input(
        "ARC start longitude",
        value=-83.183095,
        format="%.6f",
    )
    stream_bearing = st.number_input(
        "Downstream bearing (°)",
        value=342.0,
        step=1.0,
        format="%.1f",
        help="GPS-calibrated bearing for this reach.",
    )

    st.markdown("---")
    st.subheader("Input Method")
    input_method = st.radio(
        "How are you entering depth data?",
        ["Upload CSV", "Manual entry (paste)"],
        index=0,
    )

# ── Data input ────────────────────────────────────────────────────────────────
st.subheader("Depth Data Input")

df_raw = None

if input_method == "Upload CSV":
    st.markdown(
        "Upload a CSV with columns: `distance_ft`, `depth_ft`  \n"
        "Optional columns: `lat`, `lon` (if omitted, computed from bearing)"
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_raw)} rows.")
            st.dataframe(df_raw.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

else:
    st.markdown(
        "Paste two columns separated by commas: `distance_ft, depth_ft`  \n"
        "One reading per line. Example:  \n"
        "`0, 2.10`  \n`5, 2.35`  \n`10, 1.98`"
    )
    pasted = st.text_area(
        "Paste depth readings",
        height=200,
        placeholder="0, 2.10\n5, 2.35\n10, 1.98\n...",
    )
    if pasted.strip():
        try:
            lines = [l.strip() for l in pasted.strip().splitlines() if l.strip()]
            rows  = [list(map(float, l.split(","))) for l in lines]
            df_raw = pd.DataFrame(rows, columns=["distance_ft", "depth_ft"])
            st.success(f"Parsed {len(df_raw)} readings.")
        except Exception as e:
            st.error(f"Could not parse input: {e}")

# ── Demo data button ──────────────────────────────────────────────────────────
if st.button("Load example survey data (61 points, 300 ft)"):
    import numpy as np
    np.random.seed(42)
    distances = [i * 5.0 for i in range(61)]
    # Simulated pool-riffle profile with noise
    depths = [
        max(0.3, 2.5
            + 0.6 * math.sin(2 * math.pi * d / 50.0)
            + 0.3 * math.sin(2 * math.pi * d / 73.0)
            + float(np.random.normal(0, 0.08)))
        for d in distances
    ]
    df_raw = pd.DataFrame({"distance_ft": distances, "depth_ft": depths})
    st.success("Example survey loaded — 61 points over 300 ft.")

# ── Compute hydraulics ────────────────────────────────────────────────────────
if df_raw is not None and len(df_raw) >= 2:

    df = df_raw.copy()

    # Ensure required columns
    if "distance_ft" not in df.columns or "depth_ft" not in df.columns:
        st.error("CSV must contain 'distance_ft' and 'depth_ft' columns.")
        st.stop()

    df = df.sort_values("distance_ft").reset_index(drop=True)
    df["depth_ft"] = df["depth_ft"].clip(lower=MIN_DEPTH_FT)

    # ── Compute lat/lon if not provided ───────────────────────────────────────
    def forward_offset(lat, lon, dist_m, bearing_deg):
        R    = 6371000.0
        lat1 = math.radians(lat)
        lon1 = math.radians(lon)
        brng = math.radians(bearing_deg)
        lat2 = math.asin(
            math.sin(lat1) * math.cos(dist_m / R) +
            math.cos(lat1) * math.sin(dist_m / R) * math.cos(brng)
        )
        lon2 = lon1 + math.atan2(
            math.sin(brng) * math.sin(dist_m / R) * math.cos(lat1),
            math.cos(dist_m / R) - math.sin(lat1) * math.sin(lat2)
        )
        return math.degrees(lat2), math.degrees(lon2)

    if "lat" not in df.columns or "lon" not in df.columns:
        lats, lons = [], []
        for d_ft in df["distance_ft"]:
            plat, plon = forward_offset(arc_lat, arc_lon, d_ft * 0.3048, stream_bearing)
            lats.append(plat)
            lons.append(plon)
        df["lat"] = lats
        df["lon"] = lons

    # ── Bed elevation ─────────────────────────────────────────────────────────
    # WSE drops linearly at average reach slope from ARC WSE
    total_dist = df["distance_ft"].iloc[-1] - df["distance_ft"].iloc[0]
    total_drop = df["depth_ft"].iloc[0] * 0.0   # will compute from WSE

    # WSE at each point: WSE_arc - S_avg * x  (uniform flow assumption)
    # S_avg from first-to-last bed elevation
    # Bed elev = WSE - depth
    # Start: bed_0 = wse_arc - depth_0
    bed_0 = wse_arc - df["depth_ft"].iloc[0]
    bed_last = wse_arc - df["depth_ft"].iloc[-1]  # approx — iterate once
    S_avg = max(0.00001, (bed_0 - bed_last) / total_dist) if total_dist > 0 else 0.001

    df["wse_ft"]      = wse_arc - S_avg * df["distance_ft"]
    df["bed_elev_ft"] = df["wse_ft"] - df["depth_ft"]

    # ── Local slope (centered difference) ────────────────────────────────────
    slopes = []
    for i in range(len(df)):
        if i == 0:
            dx = df["distance_ft"].iloc[1] - df["distance_ft"].iloc[0]
            dz = df["bed_elev_ft"].iloc[0] - df["bed_elev_ft"].iloc[1]
        elif i == len(df) - 1:
            dx = df["distance_ft"].iloc[-1] - df["distance_ft"].iloc[-2]
            dz = df["bed_elev_ft"].iloc[-2] - df["bed_elev_ft"].iloc[-1]
        else:
            dx = df["distance_ft"].iloc[i+1] - df["distance_ft"].iloc[i-1]
            dz = df["bed_elev_ft"].iloc[i-1] - df["bed_elev_ft"].iloc[i+1]
        slopes.append(max(0.00001, dz / dx) if dx > 0 else S_avg)
    df["local_slope"] = slopes

    # ── Manning's velocity ────────────────────────────────────────────────────
    # V = (1.486/n) * R^(2/3) * S^(1/2)   [US customary, R ≈ depth for wide channel]
    df["velocity_ft_s"] = (1.486 / N_MANNING) * \
                          (df["depth_ft"] ** (2.0 / 3.0)) * \
                          (df["local_slope"] ** 0.5)

    # ── Deployable mask ───────────────────────────────────────────────────────
    df["deployable"] = df["depth_ft"] >= min_deploy_depth

    # ── Best deployment point ─────────────────────────────────────────────────
    deployable_df = df[df["deployable"] & (df["distance_ft"] >= 25.0)]
    if len(deployable_df) == 0:
        deployable_df = df[df["distance_ft"] >= 25.0]

    best_idx  = deployable_df["velocity_ft_s"].idxmax()
    best_row  = df.loc[best_idx]

    st.success(
        f"✅ Best deployment: **{best_row['distance_ft']:.0f} ft** downstream | "
        f"depth **{best_row['depth_ft']:.2f} ft** | "
        f"velocity **{best_row['velocity_ft_s']:.2f} ft/s**"
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Bed Elevation Profile",
            "Water Depth",
            "Manning's Velocity (n=0.045)",
        ),
        vertical_spacing=0.08,
    )

    # Bed elevation
    fig.add_trace(go.Scatter(
        x=df["distance_ft"], y=df["bed_elev_ft"],
        mode="lines+markers", name="Bed Elevation",
        line=dict(color="#8B6914", width=2),
        marker=dict(size=4),
        fill="tozeroy", fillcolor="rgba(139,105,20,0.15)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["distance_ft"], y=df["wse_ft"],
        mode="lines", name="Water Surface",
        line=dict(color="#4488FF", width=1.5, dash="dash"),
    ), row=1, col=1)

    # Depth — colored by deployability
    colors = ["#00C8FF" if d else "#FF4444" for d in df["deployable"]]
    fig.add_trace(go.Bar(
        x=df["distance_ft"], y=df["depth_ft"],
        name="Depth", marker_color=colors,
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(
        y=min_deploy_depth, line_dash="dot",
        line_color="orange", annotation_text=f"Min deploy depth ({min_deploy_depth:.1f} ft)",
        row=2, col=1,
    )

    # Velocity
    fig.add_trace(go.Scatter(
        x=df["distance_ft"], y=df["velocity_ft_s"],
        mode="lines+markers", name="Velocity",
        line=dict(color="#FF8C00", width=2),
        marker=dict(size=4),
    ), row=3, col=1)
    fig.add_vline(
        x=best_row["distance_ft"], line_dash="dash",
        line_color="#FF8C00",
        annotation_text=f"Best: {best_row['distance_ft']:.0f} ft",
        row=3, col=1,
    )

    fig.update_layout(
        height=700,
        template="plotly_dark",
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(title_text="Distance Downstream (ft)", row=3, col=1)
    fig.update_yaxes(title_text="Elevation (ft)", row=1, col=1)
    fig.update_yaxes(title_text="Depth (ft)", row=2, col=1)
    fig.update_yaxes(title_text="Velocity (ft/s)", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ── Folium map ────────────────────────────────────────────────────────────
    st.subheader("Survey Map")
    try:
        import folium
        from streamlit_folium import st_folium

        m = folium.Map(
            location=[arc_lat, arc_lon],
            zoom_start=17,
            tiles="OpenStreetMap",
        )

        # Survey points — color by velocity
        vmin = df["velocity_ft_s"].min()
        vmax = df["velocity_ft_s"].max()

        def _vel_color(v):
            if vmax == vmin:
                return "blue"
            t = (v - vmin) / (vmax - vmin)
            if t < 0.33:
                return "blue"
            elif t < 0.66:
                return "green"
            else:
                return "orange"

        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4,
                color=_vel_color(row["velocity_ft_s"]),
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(
                    f"{row['distance_ft']:.0f} ft downstream<br>"
                    f"Depth: {row['depth_ft']:.2f} ft<br>"
                    f"Velocity: {row['velocity_ft_s']:.2f} ft/s",
                    max_width=200,
                ),
            ).add_to(m)

        # ARC start
        folium.Marker(
            location=[arc_lat, arc_lon],
            tooltip="ARC Start",
            icon=folium.Icon(color="blue", icon="ship", prefix="fa"),
        ).add_to(m)

        # Best deployment point
        folium.Marker(
            location=[best_row["lat"], best_row["lon"]],
            tooltip=f"Best Deployment: {best_row['velocity_ft_s']:.2f} ft/s",
            popup=folium.Popup(
                f"<b>Best Deployment Point</b><br>"
                f"Distance: {best_row['distance_ft']:.0f} ft<br>"
                f"Depth: {best_row['depth_ft']:.2f} ft<br>"
                f"Velocity: {best_row['velocity_ft_s']:.2f} ft/s<br>"
                f"Bed elev: {best_row['bed_elev_ft']:.2f} ft",
                max_width=250,
            ),
            icon=folium.Icon(color="orange", icon="water", prefix="fa"),
        ).add_to(m)

        # Legend
        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:white;padding:10px;border-radius:8px;
                    border:1px solid #ccc;font-size:12px;">
        <b>Velocity</b><br>
        <span style="color:blue">●</span> Low<br>
        <span style="color:green">●</span> Medium<br>
        <span style="color:orange">●</span> High<br>
        <span style="font-size:16px">🚢</span> ARC Start<br>
        <span style="font-size:16px">📍</span> Best Deployment
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, use_container_width=True, height=450, returned_objects=[])

    except Exception as map_err:
        st.warning(f"Map could not render: {map_err}")

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Hydraulic Summary")
    summary_cols = ["distance_ft", "depth_ft", "bed_elev_ft",
                    "local_slope", "velocity_ft_s", "deployable"]
    df_display = df[summary_cols].copy()
    df_display.columns = [
        "Distance (ft)", "Depth (ft)", "Bed Elev (ft)",
        "Local Slope", "Velocity (ft/s)", "Deployable"
    ]
    df_display = df_display.round(4)

    # Highlight best row
    def _highlight_best(row):
        if row["Distance (ft)"] == best_row["distance_ft"]:
            return ["background-color: #FF8C0033"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_display.style.apply(_highlight_best, axis=1),
        use_container_width=True,
        height=300,
    )

    # ── CSV export ────────────────────────────────────────────────────────────
    st.subheader("Export")
    csv_out = df[summary_cols + ["lat", "lon"]].round(5).to_csv(index=False)
    st.download_button(
        label="⬇️ Download hydraulic survey CSV",
        data=csv_out,
        file_name="nemo_bathymetric_survey.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Key metrics ───────────────────────────────────────────────────────────
    st.subheader("Key Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Max Velocity", f"{df['velocity_ft_s'].max():.2f} ft/s")
    with m2:
        st.metric("Max Depth", f"{df['depth_ft'].max():.2f} ft")
    with m3:
        st.metric("Avg Velocity", f"{df['velocity_ft_s'].mean():.2f} ft/s")
    with m4:
        st.metric("Deployable Points",
                  f"{df['deployable'].sum()} / {len(df)}")

else:
    st.info(
        "Upload a CSV, paste depth readings, or click **Load example survey data** to begin."
    )
    st.markdown(
        """
**Expected CSV format:**
```
distance_ft,depth_ft
0,2.10
5,2.35
10,1.98
...
300,3.12
```
Optional columns `lat` and `lon` can be included if the Recon ASV logs GPS.
If omitted, coordinates are computed from the ARC start position and bearing.
"""
    )
