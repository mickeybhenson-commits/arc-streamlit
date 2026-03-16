from __future__ import annotations

import folium


def build_decision_map(lat: float, lon: float, depth_ft: float, hydro, recommendation):
    m = folium.Map(location=[lat, lon], zoom_start=14, control_scale=True)

    popup_lines = [
        "<b>Recon / ARC Decision Point</b>",
        f"Lat: {lat:.6f}",
        f"Lon: {lon:.6f}",
        f"Depth: {depth_ft:.2f} ft",
        f"Stream: {hydro.stream_name or 'Unknown'}",
        (
            f"Drainage area: {hydro.drainage_area_sqmi:.3f} mi²"
            if hydro.drainage_area_sqmi is not None
            else "Drainage area: unavailable"
        ),
        f"Action: {recommendation['action']}",
        f"Deploy: {recommendation['deploy']}",
        f"Score: {recommendation['score']}",
    ]

    deploy_status = recommendation["deploy"]
    if deploy_status == "YES":
        color = "green"
    elif deploy_status == "MAYBE":
        color = "orange"
    else:
        color = "red"

    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup("<br>".join(popup_lines), max_width=350),
        tooltip="ARC Recommendation",
        icon=folium.Icon(color=color, icon="info-sign"),
    ).add_to(m)

    return m
