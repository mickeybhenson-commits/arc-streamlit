from __future__ import annotations

import folium


def build_demo_map(
    arc_lat: float,
    arc_lon: float,
    max_lat: float,
    max_lon: float,
    deploy_lat: float,
    deploy_lon: float,
    selected_depth_ft: float,
    estimated_max_velocity_ft_s: float,
):
    center_lat = (arc_lat + max_lat + deploy_lat) / 3.0
    center_lon = (arc_lon + max_lon + deploy_lon) / 3.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=17, control_scale=True)

    folium.Marker(
        [arc_lat, arc_lon],
        tooltip="ARC Reference Point",
        popup=folium.Popup(
            f"<b>ARC Reference Point</b><br>"
            f"Lat: {arc_lat:.6f}<br>"
            f"Lon: {arc_lon:.6f}",
            max_width=300,
        ),
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

    folium.Marker(
        [max_lat, max_lon],
        tooltip="Estimated Max Velocity Point",
        popup=folium.Popup(
            f"<b>Estimated Max Velocity Point</b><br>"
            f"Lat: {max_lat:.6f}<br>"
            f"Lon: {max_lon:.6f}<br>"
            f"z (depth): {selected_depth_ft:.2f} ft<br>"
            f"Estimated vmax: {estimated_max_velocity_ft_s:.2f} ft/s",
            max_width=320,
        ),
        icon=folium.Icon(color="red", icon="flash"),
    ).add_to(m)

    folium.Marker(
        [deploy_lat, deploy_lon],
        tooltip="Estimated ARC Deployment Point",
        popup=folium.Popup(
            f"<b>Estimated ARC Deployment Point</b><br>"
            f"Lat: {deploy_lat:.6f}<br>"
            f"Lon: {deploy_lon:.6f}<br>"
            f"z (depth): {selected_depth_ft:.2f} ft",
            max_width=300,
        ),
        icon=folium.Icon(color="green", icon="ok-sign"),
    ).add_to(m)

    folium.PolyLine(
        locations=[[arc_lat, arc_lon], [max_lat, max_lon]],
        color="red",
        weight=3,
        tooltip="ARC to estimated max velocity point",
    ).add_to(m)

    folium.PolyLine(
        locations=[[max_lat, max_lon], [deploy_lat, deploy_lon]],
        color="green",
        weight=3,
        tooltip="Estimated max velocity point to deployment point",
    ).add_to(m)

    return m
