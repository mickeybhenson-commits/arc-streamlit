from __future__ import annotations

import math
from typing import Dict, Tuple, List, Union
import pandas as pd


REGIONAL_CURVES = {
    "Abkf": {"a": 22.1, "b": 0.67},
    "Wbkf": {"a": 19.9, "b": 0.36},
    "Dbkf": {"a": 1.1, "b": 0.31},
    "Qbkf": {"a": 115.7, "b": 0.73},
}


def regional_curve(a: float, b: float, drainage_area_sqmi: float) -> float:
    return a * (drainage_area_sqmi ** b)


def compute_bankfull_metrics(drainage_area_sqmi: float) -> Dict[str, float]:
    return {
        name: regional_curve(cfg["a"], cfg["b"], drainage_area_sqmi)
        for name, cfg in REGIONAL_CURVES.items()
    }


def compute_hydrokinetic_score(depth_ft: float, dbkf_ft: float) -> Tuple[float, str]:
    if dbkf_ft <= 0:
        return 0.0, "Invalid bankfull depth estimate."

    ratio = depth_ft / dbkf_ft
    score = max(0.0, 100.0 - (abs(ratio - 1.0) * 80.0))

    if ratio < 0.6:
        reason = f"Too shallow relative to estimated bankfull mean depth (depth_ratio={ratio:.2f})."
    elif ratio <= 1.4:
        reason = f"Depth is within a favorable range relative to estimated bankfull mean depth (depth_ratio={ratio:.2f})."
    else:
        reason = (
            f"Depth exceeds the bankfull mean-depth estimate; "
            f"may still be usable, but verify true velocity and position in the active channel (depth_ratio={ratio:.2f})."
        )

    return score, reason


def recommend_action(depth_ft: float, bankfull: Dict[str, float]) -> Dict[str, str]:
    dbkf = bankfull["Dbkf"]
    wbkf = bankfull["Wbkf"]
    qbkf = bankfull["Qbkf"]
    abkf = bankfull["Abkf"]

    score, reason = compute_hydrokinetic_score(depth_ft, dbkf)
    ratio = depth_ft / dbkf if dbkf > 0 else float("inf")

    if ratio < 0.6:
        action = "NAVIGATE TO DEEPER WATER / DO NOT DEPLOY YET"
        deploy = "NO"
        nav_note = (
            "Measured depth is substantially below the estimated bankfull mean depth. "
            "ARC should continue searching for a deeper portion of the main thread or thalweg."
        )
    elif ratio <= 1.4:
        action = "DEPLOY CANDIDATE"
        deploy = "YES"
        nav_note = (
            "Depth is in a favorable range relative to the estimated bankfull mean depth. "
            "This is a reasonable first-pass deployment candidate."
        )
    else:
        action = "POSSIBLE DEPLOYMENT / VERIFY FLOW FIRST"
        deploy = "MAYBE"
        nav_note = (
            "Depth is above the estimated bankfull mean-depth value. "
            "Verify that this is active high-velocity flow rather than pooled or slower water before deploying."
        )

    return {
        "deploy": deploy,
        "action": action,
        "score": f"{score:.1f}/100",
        "reason": reason,
        "nav_note": nav_note,
        "estimated_bankfull_width_ft": f"{wbkf:.2f}",
        "estimated_bankfull_depth_ft": f"{dbkf:.2f}",
        "estimated_bankfull_area_ft2": f"{abkf:.2f}",
        "estimated_bankfull_discharge_cfs": f"{qbkf:.2f}",
    }


def format_summary_table(drainage_area_sqmi: float, bankfull: Dict[str, float], depth_ft: float) -> pd.DataFrame:
    dbkf = bankfull["Dbkf"]
    depth_ratio = depth_ft / dbkf if dbkf > 0 else None

    data = [
        ["Drainage Area", drainage_area_sqmi, "mi²"],
        ["Selected Demo Depth", depth_ft, "ft"],
        ["Estimated Bankfull Area", bankfull["Abkf"], "ft²"],
        ["Estimated Bankfull Width", bankfull["Wbkf"], "ft"],
        ["Estimated Bankfull Depth", bankfull["Dbkf"], "ft"],
        ["Estimated Bankfull Discharge", bankfull["Qbkf"], "cfs"],
        ["Depth / Bankfull Depth Ratio", depth_ratio, "-"],
    ]

    return pd.DataFrame(data, columns=["Metric", "Value", "Units"])


def get_demo_depths() -> List[float]:
    depths = []
    value = 0.25
    while value <= 6.50 + 1e-9:
        depths.append(round(value, 2))
        value += 0.25
    return depths


def estimate_demo_max_velocity(depth_ft: float, bankfull: Dict[str, float]) -> Dict[str, Union[float, str]]:
    dbkf = bankfull["Dbkf"]
    abkf = bankfull["Abkf"]
    qbkf = bankfull["Qbkf"]

    avg_bankfull_velocity = qbkf / abkf if abkf > 0 else 0.0
    ratio = depth_ft / dbkf if dbkf > 0 else 1.0

    peak_factor = 0.85 + 0.45 * math.exp(-((ratio - 1.0) / 0.35) ** 2)
    estimated_max_velocity_ft_s = max(0.1, avg_bankfull_velocity * peak_factor)

    if 0.85 <= ratio <= 1.15:
        confidence = "Higher"
    elif 0.65 <= ratio <= 1.35:
        confidence = "Moderate"
    else:
        confidence = "Lower"

    note = (
        "Estimated from drainage area, regional curves, and selected demo depth. "
        "This is an estimated maximum velocity for demonstration, not a direct field measurement."
    )

    return {
        "estimated_max_velocity_ft_s": estimated_max_velocity_ft_s,
        "confidence_label": confidence,
        "note": note,
    }


def forward_offset(lat: float, lon: float, distance_m: float, bearing_deg: float) -> Tuple[float, float]:
    radius_m = 6371000.0
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(bearing_deg)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_m / radius_m) +
        math.cos(lat1) * math.sin(distance_m / radius_m) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(distance_m / radius_m) * math.cos(lat1),
        math.cos(distance_m / radius_m) - math.sin(lat1) * math.sin(lat2)
    )

    return math.degrees(lat2), math.degrees(lon2)


def estimate_demo_locations(arc_lat: float, arc_lon: float, depth_ft: float, bankfull: Dict[str, float]) -> Dict[str, float]:
    dbkf = bankfull["Dbkf"]
    ratio = depth_ft / dbkf if dbkf > 0 else 1.0

    thalweg_shift_m = min(12.0, max(4.0, 4.0 + abs(1.0 - ratio) * 8.0))
    stream_bearing_deg = 35.0
    cross_channel_bearing_deg = 80.0

    max_lat, max_lon = forward_offset(arc_lat, arc_lon, thalweg_shift_m, cross_channel_bearing_deg)
    deploy_lat, deploy_lon = forward_offset(max_lat, max_lon, 5.0, stream_bearing_deg)

    return {
        "arc_lat": arc_lat,
        "arc_lon": arc_lon,
        "max_velocity_lat": max_lat,
        "max_velocity_lon": max_lon,
        "deployment_lat": deploy_lat,
        "deployment_lon": deploy_lon,
    }


def estimate_power_output(velocity_ft_s: float, turbine_diameter_ft: float, cp: float, rho: float = 1000.0) -> Dict[str, float]:
    velocity_m_s = velocity_ft_s * 0.3048
    diameter_m = turbine_diameter_ft * 0.3048
    swept_area_m2 = math.pi * (diameter_m / 2.0) ** 2
    power_watts = 0.5 * rho * swept_area_m2 * (velocity_m_s ** 3) * cp
    power_kw = power_watts / 1000.0

    return {
        "power_watts": power_watts,
        "power_kw": power_kw,
        "swept_area_m2": swept_area_m2,
    }


def build_demo_scenario_table(
    lat: float,
    lon: float,
    depths_ft: List[float],
    bankfull: Dict[str, float],
    turbine_diameter_ft: float,
    cp: float,
) -> pd.DataFrame:
    rows = []

    for depth in depths_ft:
        rec = recommend_action(depth, bankfull)
        est_v = estimate_demo_max_velocity(depth, bankfull)
        est_loc = estimate_demo_locations(lat, lon, depth, bankfull)
        power = estimate_power_output(
            velocity_ft_s=float(est_v["estimated_max_velocity_ft_s"]),
            turbine_diameter_ft=turbine_diameter_ft,
            cp=cp,
        )

        rows.append({
            "Depth_ft": round(depth, 2),
            "Deploy": rec["deploy"],
            "Score": rec["score"],
            "Estimated_Max_Velocity_ft_s": round(float(est_v["estimated_max_velocity_ft_s"]), 3),
            "Estimated_Power_W": round(power["power_watts"], 2),
            "Estimated_Power_kW": round(power["power_kw"], 4),
            "MaxVel_X_Lon": round(est_loc["max_velocity_lon"], 6),
            "MaxVel_Y_Lat": round(est_loc["max_velocity_lat"], 6),
            "MaxVel_Z_Depth_ft": round(depth, 2),
            "Deploy_X_Lon": round(est_loc["deployment_lon"], 6),
            "Deploy_Y_Lat": round(est_loc["deployment_lat"], 6),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by="Estimated_Power_W", ascending=False).reset_index(drop=True)
    return df
