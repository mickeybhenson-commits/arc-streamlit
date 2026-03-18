from __future__ import annotations

import math
from typing import Dict, Tuple, List, Union
import pandas as pd


REGIONAL_CURVES = {
    "Abkf": {"a": 22.1, "b": 0.67},
    "Wbkf": {"a": 19.9, "b": 0.36},
    "Dbkf": {"a": 1.1,  "b": 0.31},
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
        action   = "NAVIGATE TO DEEPER WATER / DO NOT DEPLOY YET"
        deploy   = "NO"
        nav_note = (
            "Measured depth is substantially below the estimated bankfull mean depth. "
            "ARC should continue searching for a deeper portion of the main thread or thalweg."
        )
    elif ratio <= 1.4:
        action   = "DEPLOY CANDIDATE"
        deploy   = "YES"
        nav_note = (
            "Depth is in a favorable range relative to the estimated bankfull mean depth. "
            "This is a reasonable first-pass deployment candidate."
        )
    else:
        action   = "POSSIBLE DEPLOYMENT / VERIFY FLOW FIRST"
        deploy   = "MAYBE"
        nav_note = (
            "Depth is above the estimated bankfull mean-depth value. "
            "Verify that this is active high-velocity flow rather than pooled or slower water before deploying."
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
) -> Dict[str, Union[float, str, list]]:
    """
    Search a 100 ft downstream corridor (21 candidate points at 5 ft intervals)
    for the highest-velocity deployment location.

    At each candidate point, local depth is estimated using a sinusoidal
    pool-riffle model (wavelength = 50 ft, amplitude = 20% of bankfull depth)
    anchored to the measured depth at the ARC position.  The candidate score
    is the estimated max velocity derived from that local depth via the same
    regional-curve peak-factor model used elsewhere.

    A cross-channel thalweg shift (4–12 m) is applied at the best-scoring
    point to place the returned coordinates in the highest-velocity thread.

    All estimates are based on regional curves only.  The Recon ASV
    bathymetric survey will replace this with measured hydraulics.

    Parameters
    ----------
    arc_lat, arc_lon : float
        ARC vessel GPS position (decimal degrees).
    depth_ft : float
        Selected / measured water depth at the ARC position (ft).
    bankfull : dict
        Bankfull metrics from compute_bankfull_metrics().

    Returns
    -------
    dict
        arc_lat, arc_lon            : ARC position
        max_velocity_lat/lon        : best estimated deployment coordinates
        deployment_lat/lon          : 5 m downstream of max velocity point
        best_candidate_distance_ft  : distance downstream of ARC to best point
        best_candidate_depth_ft     : estimated depth at best point
        best_candidate_score        : velocity score at best point (ft/s)
        candidates_searched         : total candidates evaluated
        search_note                 : description of method used
    """
    dbkf  = bankfull["Dbkf"]

    stream_bearing_deg        = 315.0   # Cullowhee Creek flows SE→NW
    cross_channel_bearing_deg = 45.0    # perpendicular toward NE (away from parking lot)

    # ── Reach depth profile ───────────────────────────────────────────────────
    # Bed elevation profile: two fixed sinusoids seeded by coordinates.
    # Fixed physical amplitudes (not scaled by stage) represent real bed relief.
    # Water surface at ARC position = depth_ft above local bed.
    # local_depth(x) = depth_ft + bed(0) - bed(x)
    #
    # Composite score = velocity × depth_favorability^(1-alpha)
    # alpha = sigmoid(depth_ft / dbkf) — low stage weights depth favorability,
    # high stage weights velocity.  This creates a genuinely stage-dependent
    # optimal location because the two objectives peak at different downstream
    # distances.

    import hashlib as _hashlib

    _seed_str  = f"{arc_lat:.5f}{arc_lon:.5f}"
    _seed_hash = int(_hashlib.md5(_seed_str.encode()).hexdigest(), 16)
    _phase1    = (_seed_hash % 1000) / 1000.0 * 2.0 * math.pi
    _phase2    = ((_seed_hash // 1000) % 1000) / 1000.0 * 2.0 * math.pi

    WAVELENGTH1_FT = 50.0    # primary pool-riffle spacing
    WAVELENGTH2_FT = 73.0    # secondary harmonic — incommensurate
    BED_AMP1       = 0.35    # ft — fixed physical bed relief, primary
    BED_AMP2       = 0.18    # ft — fixed physical bed relief, secondary

    MIN_DEPLOY_DEPTH_FT = max(turbine_diameter_ft * 1.2, 0.5)

    # Bed elevation relative to ARC position bed (positive = higher bed = shallower)
    def _bed_rel(downstream_ft: float) -> float:
        return (
            BED_AMP1 * math.sin(2.0 * math.pi * downstream_ft / WAVELENGTH1_FT + _phase1)
            + BED_AMP2 * math.sin(2.0 * math.pi * downstream_ft / WAVELENGTH2_FT + _phase2)
            - BED_AMP1 * math.sin(_phase1)
            - BED_AMP2 * math.sin(_phase2)
        )

    def _local_depth(downstream_ft: float) -> float:
        return max(0.05, depth_ft - _bed_rel(downstream_ft))

    # Stage-dependent blending weight (sigmoid centered at bankfull)
    _ratio_stage = depth_ft / dbkf if dbkf > 0 else 1.0
    _alpha       = 1.0 / (1.0 + math.exp(-4.0 * (_ratio_stage - 1.0)))

    _v_bkf = (bankfull["Qbkf"] / bankfull["Abkf"]) if bankfull["Abkf"] > 0 else 1.0

    def _velocity_score(local_depth: float) -> float:
        """
        Composite score:
          velocity component  : v = v_bkf * (dbkf / local_depth)  [continuity]
          depth favorability  : Gaussian peak at local_depth = dbkf
          blend weight alpha  : sigmoid of depth_ft/dbkf
            alpha→0 at low stage  → depth favorability dominates
            alpha→1 at high stage → velocity dominates
        Hard cutoff below min deployable depth.
        """
        if local_depth < MIN_DEPLOY_DEPTH_FT:
            return 0.0
        v_est        = _v_bkf * (dbkf / local_depth)
        depth_ratio  = local_depth / dbkf if dbkf > 0 else 1.0
        depth_favor  = math.exp(-((depth_ratio - 1.0) / 0.45) ** 2)
        return v_est * (depth_favor ** (1.0 - _alpha))

    # ── Search corridor: 0 ft to 100 ft downstream at 5 ft intervals ─────────
    SEARCH_DISTANCE_FT = 300.0
    STEP_FT            = 5.0
    FT_TO_M            = 0.3048

    candidates = []
    steps = int(SEARCH_DISTANCE_FT / STEP_FT) + 1   # 0, 5, 10 … 100 → 21 points

    for i in range(steps):
        dist_ft = i * STEP_FT
        dist_m  = dist_ft * FT_TO_M

        cand_lat, cand_lon = forward_offset(
            arc_lat, arc_lon, dist_m, stream_bearing_deg
        )

        local_depth = _local_depth(dist_ft)
        score       = _velocity_score(local_depth)

        candidates.append({
            "distance_ft": dist_ft,
            "lat":         cand_lat,
            "lon":         cand_lon,
            "depth_ft":    round(local_depth, 3),
            "score":       round(score, 4),
        })

    # ── Select best candidate ─────────────────────────────────────────────────
    best = max(candidates, key=lambda c: c["score"])

    # Cross-channel thalweg shift at best point
    best_ratio         = best["depth_ft"] / dbkf if dbkf > 0 else 1.0
    thalweg_shift_m    = min(12.0, max(4.0, 4.0 + abs(1.0 - best_ratio) * 8.0))

    max_lat, max_lon   = forward_offset(
        best["lat"], best["lon"], thalweg_shift_m, cross_channel_bearing_deg
    )
    deploy_lat, deploy_lon = forward_offset(max_lat, max_lon, 5.0, stream_bearing_deg)

    return {
        "arc_lat":                   arc_lat,
        "arc_lon":                   arc_lon,
        "max_velocity_lat":          max_lat,
        "max_velocity_lon":          max_lon,
        "deployment_lat":            deploy_lat,
        "deployment_lon":            deploy_lon,
        "best_candidate_distance_ft": best["distance_ft"],
        "best_candidate_depth_ft":   best["depth_ft"],
        "best_candidate_score":      best["score"],
        "candidates_searched":       len(candidates),
        "search_note": (
            f"Best location found {best['distance_ft']:.0f} ft downstream of ARC position "
            f"(est. depth {best['depth_ft']:.2f} ft, velocity score {best['score']:.2f} ft/s). "
            f"Searched {len(candidates)} candidates over {SEARCH_DISTANCE_FT:.0f} ft corridor "
            f"using dual-sinusoid reach profile (λ1={WAVELENGTH1_FT:.0f} ft, "
            f"λ2={WAVELENGTH2_FT:.0f} ft). Regional curves only — Recon ASV survey will supersede."
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
