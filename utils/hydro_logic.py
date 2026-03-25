from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd


REGIONAL_CURVES = {
    "Abkf": {"a": 22.1, "b": 0.67},
    "Wbkf": {"a": 19.9, "b": 0.36},
    "Dbkf": {"a": 1.1,  "b": 0.31},
    "Qbkf": {"a": 115.7, "b": 0.73},
}

# ── Operational depth thresholds (ft) ─────────────────────────────────────────
MIN_DEPLOY_DEPTH_FT  =  2.00   # minimum depth for safe turbine deployment
HIGH_WATER_DEPTH_FT  = 10.00   # above this, verify conditions before deploying
FLOOD_STAGE_DEPTH_FT = 12.00   # at or above flood stage — do not deploy


def regional_curve(a: float, b: float, drainage_area_sqmi: float) -> float:
    return a * (drainage_area_sqmi ** b)


def compute_bankfull_metrics(drainage_area_sqmi: float) -> Dict[str, float]:
    return {
        name: regional_curve(cfg["a"], cfg["b"], drainage_area_sqmi)
        for name, cfg in REGIONAL_CURVES.items()
    }


def compute_hydrokinetic_score(depth_ft: float, dbkf_ft: float) -> Tuple[float, str]:
    """Score the hydrokinetic suitability of a given depth.

    Scoring uses practical operational thresholds rather than bankfull ratio
    alone, because depths well above bankfull are common and desirable for
    energy capture — the concern is only at flood stage.
    """
    if dbkf_ft <= 0:
        return 0.0, "Invalid bankfull depth estimate."

    ratio = depth_ft / dbkf_ft

    # ── Practical threshold scoring ───────────────────────────────────────────
    if depth_ft < MIN_DEPLOY_DEPTH_FT:
        score  = max(0.0, (depth_ft / MIN_DEPLOY_DEPTH_FT) * 40.0)
        reason = (
            f"Too shallow for safe turbine deployment "
            f"(depth={depth_ft:.2f} ft < {MIN_DEPLOY_DEPTH_FT:.2f} ft minimum, "
            f"depth_ratio={ratio:.2f})."
        )
    elif depth_ft <= HIGH_WATER_DEPTH_FT:
        # Best operating window — score 70–100 scaled by depth within the range
        frac   = (depth_ft - MIN_DEPLOY_DEPTH_FT) / (HIGH_WATER_DEPTH_FT - MIN_DEPLOY_DEPTH_FT)
        score  = 70.0 + 30.0 * math.exp(-((frac - 0.4) / 0.5) ** 2)
        reason = (
            f"Depth is within the operational deployment window "
            f"({MIN_DEPLOY_DEPTH_FT:.0f}–{HIGH_WATER_DEPTH_FT:.0f} ft, "
            f"depth_ratio={ratio:.2f})."
        )
    elif depth_ft < FLOOD_STAGE_DEPTH_FT:
        score  = 50.0 - 20.0 * ((depth_ft - HIGH_WATER_DEPTH_FT)
                                 / (FLOOD_STAGE_DEPTH_FT - HIGH_WATER_DEPTH_FT))
        reason = (
            f"Depth is approaching flood stage; verify active high-velocity flow "
            f"and safe navigation before deploying "
            f"(depth_ratio={ratio:.2f})."
        )
    else:
        score  = 10.0
        reason = (
            f"Depth is at or above flood stage ({FLOOD_STAGE_DEPTH_FT:.0f} ft). "
            f"Do not deploy — unsafe conditions (depth_ratio={ratio:.2f})."
        )

    return round(score, 1), reason


def recommend_action(depth_ft: float, bankfull: Dict[str, float]) -> Dict[str, str]:
    """Deployment recommendation based on practical operational depth thresholds.

    Decision logic
    --------------
    depth < 2.00 ft  → NO   — too shallow for turbine clearance
    2.00–10.00 ft    → YES  — good operational window for demo deployment
    10.00–12.00 ft   → MAYBE — approaching flood stage, verify conditions
    ≥ 12.00 ft       → NO   — flood stage, unsafe
    """
    dbkf = bankfull["Dbkf"]
    wbkf = bankfull["Wbkf"]
    qbkf = bankfull["Qbkf"]
    abkf = bankfull["Abkf"]

    score, reason = compute_hydrokinetic_score(depth_ft, dbkf)

    if depth_ft < MIN_DEPLOY_DEPTH_FT:
        action   = "NAVIGATE TO DEEPER WATER / DO NOT DEPLOY YET"
        deploy   = "NO"
        nav_note = (
            f"Measured depth ({depth_ft:.2f} ft) is below the {MIN_DEPLOY_DEPTH_FT:.2f} ft "
            f"minimum required for safe turbine deployment. "
            "ARC should continue searching for deeper water in the main thread or thalweg."
        )
    elif depth_ft <= HIGH_WATER_DEPTH_FT:
        action   = "DEPLOY CANDIDATE"
        deploy   = "YES"
        nav_note = (
            f"Depth ({depth_ft:.2f} ft) is within the operational deployment window "
            f"({MIN_DEPLOY_DEPTH_FT:.0f}–{HIGH_WATER_DEPTH_FT:.0f} ft). "
            "This is a favorable deployment candidate."
        )
    elif depth_ft < FLOOD_STAGE_DEPTH_FT:
        action   = "POSSIBLE DEPLOYMENT / VERIFY CONDITIONS"
        deploy   = "MAYBE"
        nav_note = (
            f"Depth ({depth_ft:.2f} ft) is approaching flood stage "
            f"({FLOOD_STAGE_DEPTH_FT:.0f} ft). Verify that flow is navigable "
            "and velocity is not dangerously high before deploying."
        )
    else:
        action   = "DO NOT DEPLOY — FLOOD STAGE"
        deploy   = "NO"
        nav_note = (
            f"Depth ({depth_ft:.2f} ft) is at or above flood stage "
            f"({FLOOD_STAGE_DEPTH_FT:.0f} ft). Conditions are unsafe for deployment. "
            "Retract and wait for water levels to recede."
        )

    return {
        "deploy":   deploy,
        "action":   action,
        "score":    f"{score:.1f}/100",
        "reason":   reason,
        "nav_note": nav_note,
        "estimated_bankfull_width_ft":        f"{wbkf:.2f}",
        "estimated_bankfull_depth_ft":        f"{dbkf:.2f}",
        "estimated_bankfull_area_ft2":        f"{abkf:.2f}",
        "estimated_bankfull_discharge_cfs":   f"{qbkf:.2f}",
    }


def format_summary_table(
    drainage_area_sqmi: float,
    bankfull: Dict[str, float],
    depth_ft: float,
) -> pd.DataFrame:
    dbkf        = bankfull["Dbkf"]
    depth_ratio = depth_ft / dbkf if dbkf > 0 else None

    data = [
        ["Drainage Area",                  drainage_area_sqmi,    "mi²"],
        ["Selected Demo Depth",            depth_ft,              "ft"],
        ["Estimated Bankfull Area",        bankfull["Abkf"],      "ft²"],
        ["Estimated Bankfull Width",       bankfull["Wbkf"],      "ft"],
        ["Estimated Bankfull Depth",       bankfull["Dbkf"],      "ft"],
        ["Estimated Bankfull Discharge",   bankfull["Qbkf"],      "cfs"],
        ["Depth / Bankfull Depth Ratio",   depth_ratio,           "-"],
    ]

    return pd.DataFrame(data, columns=["Metric", "Value", "Units"])


def get_demo_depths() -> List[float]:
    depths = []
    value  = 0.25
    while value <= 6.50 + 1e-9:
        depths.append(round(value, 2))
        value += 0.25
    return depths


def estimate_demo_max_velocity(
    depth_ft: float,
    bankfull: Dict[str, float],
) -> Dict[str, Union[float, str]]:
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
        "confidence_label":            confidence,
        "note":                        note,
    }


def forward_offset(
    lat: float,
    lon: float,
    distance_m: float,
    bearing_deg: float,
) -> Tuple[float, float]:
    radius_m = 6371000.0
    lat1     = math.radians(lat)
    lon1     = math.radians(lon)
    brng     = math.radians(bearing_deg)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_m / radius_m) +
        math.cos(lat1) * math.sin(distance_m / radius_m) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(distance_m / radius_m) * math.cos(lat1),
        math.cos(distance_m / radius_m) - math.sin(lat1) * math.sin(lat2)
    )

    return math.degrees(lat2), math.degrees(lon2)


def estimate_demo_locations(
    arc_lat: float,
    arc_lon: float,
    depth_ft: float,
    bankfull: Dict[str, float],
    turbine_diameter_ft: float = 1.5,
    reach_elevations: Optional[List[float]] = None,
    reach_distances: Optional[List[float]] = None,
    downstream_bearing: float = 155.0,
    search_distance_ft: float = 300.0,
) -> Dict[str, Union[float, str, list]]:
    """
    Search downstream from the upper boundary for the best deployment location.

    PRIMARY MODEL — Manning's equation with real 3DEP elevations (when available):
        V = (1.486/n) · R^(2/3) · S^(1/2)
        n = 0.045 (post-Helene WNC calibrated, Henson et al.)
        R ≈ local_depth (wide channel)
        S = local bed slope from centered difference on 3DEP elevations
        local_depth = WSE(x) - bed_elev(x)
        WSE(x) = WSE_arc - S_avg · x  (uniform flow along corridor)

    FALLBACK — dual sinusoid estimated bed profile when 3DEP unavailable.

    Parameters
    ----------
    arc_lat, arc_lon     : upper boundary coordinates
    reach_elevations     : list of ft elevations at reach_distances sample points
    reach_distances      : list of ft distances from upper boundary
    search_distance_ft   : how far downstream to search (default 300 ft)
    """
    import hashlib as _hashlib

    dbkf      = bankfull["Dbkf"]
    STREAM_BEARING_DEG = downstream_bearing   # auto-detected from 3DEP elevation probes
    SEARCH_FT = float(search_distance_ft)
    STEP_FT   = 5.0
    FT_TO_M   = 0.3048
    MIN_DEPTH = max(turbine_diameter_ft * 1.2, 0.5)
    N_MANNING = 0.045   # post-Helene WNC calibrated roughness

    use_real_elevations = (
        reach_elevations is not None
        and reach_distances is not None
        and len(reach_elevations) >= 2
        and all(e is not None for e in reach_elevations)
    )

    # ── Elevation interpolation ───────────────────────────────────────────────
    if use_real_elevations:
        def _interp_elev(x_ft: float) -> float:
            if x_ft <= reach_distances[0]:
                return reach_elevations[0]
            if x_ft >= reach_distances[-1]:
                return reach_elevations[-1]
            for i in range(len(reach_distances) - 1):
                if reach_distances[i] <= x_ft <= reach_distances[i + 1]:
                    t = (x_ft - reach_distances[i]) / (reach_distances[i + 1] - reach_distances[i])
                    return reach_elevations[i] + t * (reach_elevations[i + 1] - reach_elevations[i])
            return reach_elevations[-1]

        arc_bed_elev = _interp_elev(0.0)
        end_bed_elev = _interp_elev(SEARCH_FT)
        S_avg        = max(0.0001, (arc_bed_elev - end_bed_elev) / SEARCH_FT)
        wse_arc      = arc_bed_elev + depth_ft   # water surface at upper boundary

        def _local_depth(x_ft: float) -> float:
            wse   = wse_arc - S_avg * x_ft        # WSE drops at average slope
            bed   = _interp_elev(x_ft)
            return max(0.05, wse - bed)

        def _local_slope(x_ft: float) -> float:
            dx    = 15.0
            hi    = _interp_elev(max(0.0, x_ft - dx))
            lo    = _interp_elev(min(SEARCH_FT, x_ft + dx))
            return max(0.0001, (hi - lo) / (2.0 * dx))

        def _score(x_ft: float) -> float:
            d = _local_depth(x_ft)
            if d < MIN_DEPTH:
                return 0.0
            S = _local_slope(x_ft)
            # Manning's: V = (1.486/n) * R^(2/3) * S^(1/2), wide channel R ≈ d
            return (1.486 / N_MANNING) * (d ** (2.0 / 3.0)) * (S ** 0.5)

        elev_method = "USGS 3DEP 1m lidar — Manning's equation (n=0.045)"

    else:
        # ── Sinusoidal fallback ───────────────────────────────────────────────
        _seed_str  = f"{arc_lat:.5f}{arc_lon:.5f}"
        _seed_hash = int(_hashlib.md5(_seed_str.encode()).hexdigest(), 16)
        _phase1    = (_seed_hash % 1000) / 1000.0 * 2.0 * math.pi
        _phase2    = ((_seed_hash // 1000) % 1000) / 1000.0 * 2.0 * math.pi
        _phase3    = ((_seed_hash // 1000000) % 1000) / 1000.0 * 2.0 * math.pi

        _ratio_st  = depth_ft / dbkf if dbkf > 0 else 1.0
        _alpha     = 1.0 / (1.0 + math.exp(-4.0 * (_ratio_st - 1.0)))
        _v_bkf     = (bankfull["Qbkf"] / bankfull["Abkf"]) if bankfull["Abkf"] > 0 else 1.0

        BED_AMP1 = 0.35
        BED_AMP2 = 0.18
        BED_AMP3 = 0.10 * max(0.0, _ratio_st - 0.5)
        WLEN1, WLEN2, WLEN3 = 50.0, 73.0, 31.0

        _bed0 = (
            BED_AMP1 * math.sin(_phase1)
            + BED_AMP2 * math.sin(_phase2)
            + BED_AMP3 * math.sin(_phase3)
        )

        def _local_depth(x_ft: float) -> float:
            bed = (
                BED_AMP1 * math.sin(2.0 * math.pi * x_ft / WLEN1 + _phase1)
                + BED_AMP2 * math.sin(2.0 * math.pi * x_ft / WLEN2 + _phase2)
                + BED_AMP3 * math.sin(2.0 * math.pi * x_ft / WLEN3 + _phase3)
            )
            return max(0.10, depth_ft - (bed - _bed0))

        def _score(x_ft: float) -> float:
            d = _local_depth(x_ft)
            if d < MIN_DEPTH:
                return 0.0
            v_est     = _v_bkf * (dbkf / d)
            depth_fav = math.exp(-((d / dbkf - 1.0) / 0.45) ** 2) if dbkf > 0 else 1.0
            long_bias = 1.0 + 0.003 * _alpha * x_ft
            return v_est * (depth_fav ** (1.0 - _alpha)) * long_bias

        elev_method = "Estimated sinusoidal bed profile (3DEP unavailable)"

    # ── Search corridor ───────────────────────────────────────────────────────
    candidates = []
    n_steps = int(SEARCH_FT / STEP_FT) + 1

    for i in range(n_steps):
        x_ft  = i * STEP_FT
        x_m   = x_ft * FT_TO_M
        c_lat, c_lon = forward_offset(arc_lat, arc_lon, x_m, STREAM_BEARING_DEG)
        d     = _local_depth(x_ft)
        sc    = _score(x_ft)
        candidates.append({
            "distance_ft": x_ft,
            "lat":         c_lat,
            "lon":         c_lon,
            "depth_ft":    round(d, 3),
            "score":       round(sc, 4),
        })

    best = max(
        (c for c in candidates if c["distance_ft"] >= 25.0),
        key=lambda c: c["score"],
        default=candidates[-1],
    )

    max_lat, max_lon       = best["lat"], best["lon"]
    deploy_lat, deploy_lon = forward_offset(max_lat, max_lon, 2.0, STREAM_BEARING_DEG)

    return {
        "arc_lat":                    arc_lat,
        "arc_lon":                    arc_lon,
        "max_velocity_lat":           max_lat,
        "max_velocity_lon":           max_lon,
        "deployment_lat":             deploy_lat,
        "deployment_lon":             deploy_lon,
        "best_candidate_distance_ft": best["distance_ft"],
        "best_candidate_depth_ft":    best["depth_ft"],
        "best_candidate_score":       best["score"],
        "candidates_searched":        len(candidates),
        "elev_method":                elev_method,
        "search_note": (
            f"Best location {best['distance_ft']:.0f} ft downstream of upper boundary "
            f"(est. depth {best['depth_ft']:.2f} ft, "
            f"velocity {best['score']:.2f} ft/s). "
            f"{len(candidates)} candidates over {SEARCH_FT:.0f} ft. "
            f"Method: {elev_method}."
        ),
    }


def estimate_power_output(
    velocity_ft_s: float,
    turbine_diameter_ft: float,
    cp: float,
    num_rows: int = 3,
    turbines_per_row: int = 2,
    wake_velocity_factor: float = 0.85,
) -> Dict[str, Union[float, int, list]]:
    """
    Estimate total array power output for the NEMO 6-turbine hydrokinetic array.

    Vessel configuration:
        - 12 ft long × 5 ft wide
        - 3 stations at 1', 5', 9' from bow (upstream end)
        - Port and starboard turbine at each station (2 per row)
        - Approach flow runs along the vessel long axis;
          wake cascades station 1' → 5' → 9'

    Wake model:
        Each downstream station sees velocity reduced by wake_velocity_factor.
        Power scales as v³, so:
            Station 1' power factor = 1.000
            Station 5' power factor = wake_velocity_factor³
            Station 9' power factor = wake_velocity_factor⁶

    Parameters
    ----------
    velocity_ft_s          : approach velocity at Station 1' (ft/s)
    turbine_diameter_ft    : rotor diameter (ft)
    cp                     : power coefficient (dimensionless)
    num_rows               : number of stations along vessel length (default 3)
    turbines_per_row       : turbines per station — port + starboard (default 2)
    wake_velocity_factor   : fractional velocity retained by each successive
                             downstream station (default 0.85)

    Returns
    -------
    dict
        power_watts              : total array power (W)
        power_kw                 : total array power (kW)
        num_rows                 : number of stations
        turbines_per_row         : turbines per station
        total_turbines           : total turbine count (num_rows × turbines_per_row)
        wake_velocity_factor     : wake factor used
        row_powers_watts         : per-station total power list (W), length = num_rows
        row_velocities_ft_s      : per-station velocity list (ft/s), length = num_rows
        single_turbine_watts     : reference power for one turbine at approach velocity (W)
        single_turbine_kw        : same, in kW
        swept_area_m2            : rotor swept area (m²) — backward-compatible key
    """
    rho_water_kg_m3 = 999.7           # kg/m³ freshwater

    velocity_ms = velocity_ft_s * 0.3048
    diameter_m  = turbine_diameter_ft * 0.3048
    area_m2     = math.pi / 4.0 * diameter_m ** 2

    def _single_turbine_watts(v_ms: float) -> float:
        """Actuator-disk power: P = 0.5 · Cp · ρ · A · v³"""
        return 0.5 * cp * rho_water_kg_m3 * area_m2 * v_ms ** 3

    row_powers_w:      List[float] = []
    row_velocities_fs: List[float] = []

    for row_idx in range(num_rows):
        v_row_ms   = velocity_ms * (wake_velocity_factor ** row_idx)
        v_row_ft_s = v_row_ms / 0.3048
        p_row_w    = turbines_per_row * _single_turbine_watts(v_row_ms)
        row_powers_w.append(p_row_w)
        row_velocities_fs.append(v_row_ft_s)

    total_w  = sum(row_powers_w)
    total_kw = total_w / 1000.0

    return {
        # ── Scalar totals ─────────────────────────────────────────────────────
        "power_watts": total_w,
        "power_kw":    total_kw,
        # ── Array configuration ───────────────────────────────────────────────
        "num_rows":             num_rows,
        "turbines_per_row":     turbines_per_row,
        "total_turbines":       num_rows * turbines_per_row,
        "wake_velocity_factor": wake_velocity_factor,
        # ── Per-station breakdown ─────────────────────────────────────────────
        "row_powers_watts":    row_powers_w,
        "row_velocities_ft_s": row_velocities_fs,
        # ── Single-turbine reference (at approach velocity) ───────────────────
        "single_turbine_watts": _single_turbine_watts(velocity_ms),
        "single_turbine_kw":    _single_turbine_watts(velocity_ms) / 1000.0,
        # ── Backward-compatible key ───────────────────────────────────────────
        "swept_area_m2": area_m2,
    }


def build_demo_scenario_table(
    lat: float,
    lon: float,
    depths_ft: List[float],
    bankfull: Dict[str, float],
    turbine_diameter_ft: float,
    cp: float,
    num_rows: int = 3,
    turbines_per_row: int = 2,
    wake_velocity_factor: float = 0.85,
) -> pd.DataFrame:
    """
    Build a scenario table across all demo depths.

    Power columns reflect the full 6-turbine array with wake losses applied.
    Per-station breakdown is not included in the table (kept flat for export).
    """
    rows = []

    for depth in depths_ft:
        rec     = recommend_action(depth, bankfull)
        est_v   = estimate_demo_max_velocity(depth, bankfull)
        est_loc = estimate_demo_locations(lat, lon, depth, bankfull)
        power   = estimate_power_output(
            velocity_ft_s=float(est_v["estimated_max_velocity_ft_s"]),
            turbine_diameter_ft=turbine_diameter_ft,
            cp=cp,
            num_rows=num_rows,
            turbines_per_row=turbines_per_row,
            wake_velocity_factor=wake_velocity_factor,
        )

        rows.append({
            "Depth_ft":                    round(depth, 2),
            "Deploy":                      rec["deploy"],
            "Score":                       rec["score"],
            "Estimated_Max_Velocity_ft_s": round(float(est_v["estimated_max_velocity_ft_s"]), 3),
            "Array_Total_Power_W":         round(power["power_watts"], 2),
            "Array_Total_Power_kW":        round(power["power_kw"], 4),
            "Single_Turbine_Power_W":      round(power["single_turbine_watts"], 2),
            "Station1_Power_W":            round(power["row_powers_watts"][0], 2),
            "Station5_Power_W":            round(power["row_powers_watts"][1], 2),
            "Station9_Power_W":            round(power["row_powers_watts"][2], 2),
            "MaxVel_X_Lon":                round(est_loc["max_velocity_lon"], 6),
            "MaxVel_Y_Lat":                round(est_loc["max_velocity_lat"], 6),
            "MaxVel_Z_Depth_ft":           round(depth, 2),
            "Deploy_X_Lon":                round(est_loc["deployment_lon"], 6),
            "Deploy_Y_Lat":                round(est_loc["deployment_lat"], 6),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by="Array_Total_Power_W", ascending=False).reset_index(drop=True)
    return df
