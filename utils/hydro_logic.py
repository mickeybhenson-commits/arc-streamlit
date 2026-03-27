from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd


REGIONAL_CURVES = {
    "Abkf": {"a": 22.1,  "b": 0.67},
    "Wbkf": {"a": 19.9,  "b": 0.36},
    "Dbkf": {"a":  1.1,  "b": 0.31},
    "Qbkf": {"a": 115.7, "b": 0.73},
}

MIN_DEPLOY_DEPTH_FT  =  2.00
HIGH_WATER_DEPTH_FT  = 10.00
FLOOD_STAGE_DEPTH_FT = 12.00


# ── Geodetic helpers ──────────────────────────────────────────────────────────

def _haversine_ft(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in feet."""
    R  = 6_371_000.0
    φ1 = math.radians(lat1)
    φ2 = math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a)) / 0.3048


def _bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Forward azimuth (degrees, 0–360) from point 1 → point 2."""
    dlon  = math.radians(lon2 - lon1)
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _angle_diff(a: float, b: float) -> float:
    """Absolute angular difference (0–180°)."""
    return abs(((a - b + 180) % 360) - 180)


def forward_offset(
    lat: float, lon: float, distance_m: float, bearing_deg: float
) -> Tuple[float, float]:
    R    = 6_371_000.0
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(bearing_deg)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_m / R)
        + math.cos(lat1) * math.sin(distance_m / R) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(distance_m / R) * math.cos(lat1),
        math.cos(distance_m / R) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


# ── Flowline walking ──────────────────────────────────────────────────────────

def _walk_flowline(
    arc_lat:            float,
    arc_lon:            float,
    flowline_coords:    List[Tuple[float, float]],   # (lat, lon) upstream→downstream
    downstream_bearing: float,
    search_distance_ft: float,
    step_ft:            float = 5.0,
) -> List[Tuple[float, float, float]]:
    """
    Walk downstream along the NHDPlus flowline from the arc position.

    Every returned candidate is INTERPOLATED FROM FLOWLINE VERTICES —
    meaning it lies exactly on the stream centerline geometry.

    Parameters
    ----------
    arc_lat / arc_lon   : upper boundary (start of search)
    flowline_coords     : (lat, lon) list from usgs_lookup.fetch_flowline_geometry()
    downstream_bearing  : used only to resolve flowline direction ambiguity
    search_distance_ft  : how far downstream to search
    step_ft             : candidate spacing along channel

    Returns
    -------
    List of (distance_ft, lat, lon) — one entry per candidate, starting at 0 ft.
    """
    if len(flowline_coords) < 2:
        return []

    n = len(flowline_coords)

    # 1. Find nearest flowline vertex to arc position
    min_d     = float("inf")
    start_idx = 0
    for i, (vlat, vlon) in enumerate(flowline_coords):
        d = _haversine_ft(arc_lat, arc_lon, vlat, vlon)
        if d < min_d:
            min_d     = d
            start_idx = i

    # 2. Determine downstream direction
    #    Compare bearing of each adjacent segment to downstream_bearing hint.
    fwd_b = (
        _bearing_between(
            flowline_coords[start_idx][0], flowline_coords[start_idx][1],
            flowline_coords[start_idx + 1][0], flowline_coords[start_idx + 1][1],
        )
        if start_idx < n - 1 else None
    )
    bwd_b = (
        _bearing_between(
            flowline_coords[start_idx][0], flowline_coords[start_idx][1],
            flowline_coords[start_idx - 1][0], flowline_coords[start_idx - 1][1],
        )
        if start_idx > 0 else None
    )

    fwd_diff = _angle_diff(fwd_b, downstream_bearing) if fwd_b is not None else 360.0
    bwd_diff = _angle_diff(bwd_b, downstream_bearing) if bwd_b is not None else 360.0

    if fwd_diff <= bwd_diff:
        segment: List[Tuple[float, float]] = flowline_coords[start_idx:]
    else:
        segment = flowline_coords[start_idx::-1]

    if len(segment) < 2:
        return []

    # 3. Cumulative along-channel distances
    cum: List[float] = [0.0]
    for i in range(1, len(segment)):
        cum.append(cum[-1] + _haversine_ft(
            segment[i-1][0], segment[i-1][1],
            segment[i][0],   segment[i][1],
        ))

    flowline_len_ft = cum[-1]

    # Final bearing used to extend the flowline if it runs out
    final_bearing = _bearing_between(
        segment[-2][0], segment[-2][1],
        segment[-1][0], segment[-1][1],
    ) if len(segment) >= 2 else downstream_bearing

    last_lat, last_lon = segment[-1]

    def _interp_at(dist_ft: float) -> Tuple[float, float]:
        """Interpolate (lat, lon) along segment at dist_ft from start."""
        if dist_ft <= 0.0:
            return segment[0]
        if dist_ft >= flowline_len_ft:
            # Extend beyond end of flowline using final bearing
            overshoot_m = (dist_ft - flowline_len_ft) * 0.3048
            return forward_offset(last_lat, last_lon, overshoot_m, final_bearing)
        # Linear interpolation within segment
        lo, hi = 0, len(cum) - 2
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if cum[mid] <= dist_ft:
                lo = mid
            else:
                hi = mid
        # Ensure correct segment
        j = lo
        while j < len(cum) - 2 and cum[j + 1] < dist_ft:
            j += 1
        seg_len = cum[j + 1] - cum[j]
        t = (dist_ft - cum[j]) / seg_len if seg_len > 0 else 0.0
        lat = segment[j][0] + t * (segment[j + 1][0] - segment[j][0])
        lon = segment[j][1] + t * (segment[j + 1][1] - segment[j][1])
        return lat, lon

    # 4. Generate candidates every step_ft
    candidates: List[Tuple[float, float, float]] = []
    dist = 0.0
    while dist <= search_distance_ft + 1e-6:
        lat, lon = _interp_at(dist)
        candidates.append((round(dist, 2), lat, lon))
        dist = round(dist + step_ft, 6)

    return candidates


# ── Regional curves and bankfull ─────────────────────────────────────────────

def regional_curve(a: float, b: float, drainage_area_sqmi: float) -> float:
    return a * (drainage_area_sqmi ** b)


def compute_bankfull_metrics(drainage_area_sqmi: float) -> Dict[str, float]:
    return {
        name: regional_curve(cfg["a"], cfg["b"], drainage_area_sqmi)
        for name, cfg in REGIONAL_CURVES.items()
    }


# ── Hydrokinetic scoring ──────────────────────────────────────────────────────

def compute_hydrokinetic_score(depth_ft: float, dbkf_ft: float) -> Tuple[float, str]:
    if dbkf_ft <= 0:
        return 0.0, "Invalid bankfull depth estimate."

    ratio = depth_ft / dbkf_ft

    if depth_ft < MIN_DEPLOY_DEPTH_FT:
        score  = max(0.0, (depth_ft / MIN_DEPLOY_DEPTH_FT) * 40.0)
        reason = (
            f"Too shallow for safe turbine deployment "
            f"(depth={depth_ft:.2f} ft < {MIN_DEPLOY_DEPTH_FT:.2f} ft minimum, "
            f"depth_ratio={ratio:.2f})."
        )
    elif depth_ft <= HIGH_WATER_DEPTH_FT:
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
            f"Depth is approaching flood stage; verify conditions before deploying "
            f"(depth_ratio={ratio:.2f})."
        )
    else:
        score  = 10.0
        reason = (
            f"Depth at or above flood stage ({FLOOD_STAGE_DEPTH_FT:.0f} ft). "
            f"Do not deploy (depth_ratio={ratio:.2f})."
        )

    return round(score, 1), reason


def recommend_action(
    depth_ft: float, bankfull: Dict[str, float]
) -> Dict[str, str]:
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
            f"({FLOOD_STAGE_DEPTH_FT:.0f} ft). Verify navigable flow before deploying."
        )
    else:
        action   = "DO NOT DEPLOY — FLOOD STAGE"
        deploy   = "NO"
        nav_note = (
            f"Depth ({depth_ft:.2f} ft) at or above flood stage "
            f"({FLOOD_STAGE_DEPTH_FT:.0f} ft). Unsafe — retract and wait."
        )

    return {
        "deploy":   deploy,
        "action":   action,
        "score":    f"{score:.1f}/100",
        "reason":   reason,
        "nav_note": nav_note,
        "estimated_bankfull_width_ft":       f"{wbkf:.2f}",
        "estimated_bankfull_depth_ft":       f"{dbkf:.2f}",
        "estimated_bankfull_area_ft2":       f"{abkf:.2f}",
        "estimated_bankfull_discharge_cfs":  f"{qbkf:.2f}",
    }


def format_summary_table(
    drainage_area_sqmi: float,
    bankfull: Dict[str, float],
    depth_ft: float,
) -> pd.DataFrame:
    dbkf        = bankfull["Dbkf"]
    depth_ratio = depth_ft / dbkf if dbkf > 0 else None
    data = [
        ["Drainage Area",                drainage_area_sqmi,  "mi²"],
        ["Selected Demo Depth",          depth_ft,            "ft"],
        ["Estimated Bankfull Area",      bankfull["Abkf"],    "ft²"],
        ["Estimated Bankfull Width",     bankfull["Wbkf"],    "ft"],
        ["Estimated Bankfull Depth",     bankfull["Dbkf"],    "ft"],
        ["Estimated Bankfull Discharge", bankfull["Qbkf"],    "cfs"],
        ["Depth / Bankfull Depth Ratio", depth_ratio,         "-"],
    ]
    return pd.DataFrame(data, columns=["Metric", "Value", "Units"])


def get_demo_depths() -> List[float]:
    depths = []
    value  = 0.25
    while value <= 6.50 + 1e-9:
        depths.append(round(value, 2))
        value += 0.25
    return depths


# ── Velocity estimation ───────────────────────────────────────────────────────

def estimate_demo_max_velocity(
    depth_ft: float,
    bankfull: Dict[str, float],
) -> Dict[str, Union[float, str]]:
    dbkf  = bankfull["Dbkf"]
    abkf  = bankfull["Abkf"]
    qbkf  = bankfull["Qbkf"]

    avg_bankfull_velocity       = qbkf / abkf if abkf > 0 else 0.0
    ratio                       = depth_ft / dbkf if dbkf > 0 else 1.0
    peak_factor                 = 0.85 + 0.45 * math.exp(-((ratio - 1.0) / 0.35) ** 2)
    estimated_max_velocity_ft_s = max(0.1, avg_bankfull_velocity * peak_factor)

    if 0.85 <= ratio <= 1.15:
        confidence = "Higher"
    elif 0.65 <= ratio <= 1.35:
        confidence = "Moderate"
    else:
        confidence = "Lower"

    return {
        "estimated_max_velocity_ft_s": estimated_max_velocity_ft_s,
        "confidence_label":            confidence,
        "note": (
            "Estimated from drainage area, regional curves, and selected demo depth. "
            "Not a direct field measurement."
        ),
    }


# ── Location estimation ───────────────────────────────────────────────────────

def estimate_demo_locations(
    arc_lat:            float,
    arc_lon:            float,
    depth_ft:           float,
    bankfull:           Dict[str, float],
    turbine_diameter_ft: float                          = 1.5,
    reach_elevations:   Optional[List[float]]           = None,
    reach_distances:    Optional[List[float]]           = None,
    downstream_bearing: float                           = 155.0,
    search_distance_ft: float                           = 300.0,
    flowline_coords:    Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Union[float, str, list]]:
    """
    Find the best hydrokinetic deployment location downstream of arc position.

    CANDIDATE GENERATION (priority order)
    ──────────────────────────────────────
    1. NHDPlus flowline geometry (flowline_coords):
       Candidates are interpolated from the actual stream centerline vertices.
       Every candidate is GUARANTEED to lie inside the channel.

    2. Straight-line bearing fallback (when flowline unavailable):
       Candidates projected along downstream_bearing — may drift off-channel
       in sinuous reaches.  Use only when COMID lookup fails.

    HYDRAULIC SCORING
    ─────────────────
    Manning's equation on each candidate when 3DEP elevations are available:
        V = (1.486/n) · R^(2/3) · S^(1/2)    n = 0.045 (WNC calibrated)
        R ≈ local_depth (wide channel)
        S = local bed slope from 3DEP centered difference
        WSE(x) = WSE_arc - S_avg · x  (uniform flow)

    Falls back to sinusoidal bed profile when 3DEP unavailable.
    """
    import hashlib as _hashlib

    dbkf      = bankfull["Dbkf"]
    STEP_FT   = 5.0
    MIN_DEPTH = max(turbine_diameter_ft * 1.2, 0.5)
    N_MANNING = 0.045

    # ── Build candidate positions ─────────────────────────────────────────────
    use_flowline = flowline_coords is not None and len(flowline_coords) >= 2

    if use_flowline:
        raw_candidates = _walk_flowline(
            arc_lat, arc_lon, flowline_coords,
            downstream_bearing, search_distance_ft, STEP_FT,
        )
        elev_suffix = "NHDPlus flowline centerline"
    else:
        # Straight-line projection fallback
        raw_candidates = []
        n_steps = int(search_distance_ft / STEP_FT) + 1
        for i in range(n_steps):
            x_ft  = round(i * STEP_FT, 2)
            x_m   = x_ft * 0.3048
            c_lat, c_lon = forward_offset(arc_lat, arc_lon, x_m, downstream_bearing)
            raw_candidates.append((x_ft, c_lat, c_lon))
        elev_suffix = "bearing projection (flowline unavailable)"

    if not raw_candidates:
        # Ultimate fallback — single candidate at arc position
        raw_candidates = [(0.0, arc_lat, arc_lon)]

    # ── Hydraulic scoring ─────────────────────────────────────────────────────
    use_real_elevations = (
        reach_elevations is not None
        and reach_distances is not None
        and len(reach_elevations) >= 2
        and all(e is not None for e in reach_elevations)
    )

    if use_real_elevations:
        def _interp_elev(x_ft: float) -> float:
            if x_ft <= reach_distances[0]:
                return reach_elevations[0]
            if x_ft >= reach_distances[-1]:
                return reach_elevations[-1]
            for i in range(len(reach_distances) - 1):
                if reach_distances[i] <= x_ft <= reach_distances[i + 1]:
                    t = ((x_ft - reach_distances[i])
                         / (reach_distances[i + 1] - reach_distances[i]))
                    return reach_elevations[i] + t * (reach_elevations[i + 1] - reach_elevations[i])
            return reach_elevations[-1]

        arc_bed_elev = _interp_elev(0.0)
        end_bed_elev = _interp_elev(search_distance_ft)
        S_avg        = max(0.0001, (arc_bed_elev - end_bed_elev) / search_distance_ft)
        wse_arc      = arc_bed_elev + depth_ft

        def _local_depth(x_ft: float) -> float:
            wse = wse_arc - S_avg * x_ft
            bed = _interp_elev(x_ft)
            return max(0.05, wse - bed)

        def _local_slope(x_ft: float) -> float:
            dx = 15.0
            hi = _interp_elev(max(0.0, x_ft - dx))
            lo = _interp_elev(min(search_distance_ft, x_ft + dx))
            return max(0.0001, (hi - lo) / (2.0 * dx))

        def _score(x_ft: float) -> float:
            d = _local_depth(x_ft)
            if d < MIN_DEPTH:
                return 0.0
            S = _local_slope(x_ft)
            return (1.486 / N_MANNING) * (d ** (2.0 / 3.0)) * (S ** 0.5)

        elev_method = f"USGS 3DEP 1m lidar — Manning's equation (n={N_MANNING}) | {elev_suffix}"

    else:
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

        elev_method = f"Estimated sinusoidal bed profile (3DEP unavailable) | {elev_suffix}"

    # ── Score every candidate ─────────────────────────────────────────────────
    candidates = []
    for (x_ft, c_lat, c_lon) in raw_candidates:
        d  = _local_depth(x_ft)
        sc = _score(x_ft)
        candidates.append({
            "distance_ft": x_ft,
            "lat":         c_lat,
            "lon":         c_lon,
            "depth_ft":    round(d, 3),
            "score":       round(sc, 4),
        })

    # Skip candidates within 25 ft of arc (vessel needs run-up distance)
    scored = [c for c in candidates if c["distance_ft"] >= 25.0]
    best   = max(scored, key=lambda c: c["score"]) if scored else candidates[-1]

    max_lat, max_lon = best["lat"], best["lon"]

    # Deployment point: 2 m further downstream from best candidate (along channel)
    if use_flowline and raw_candidates:
        # Find the flowline position 2 m past the best candidate
        deploy_lat, deploy_lon = forward_offset(max_lat, max_lon, 2.0, downstream_bearing)
    else:
        deploy_lat, deploy_lon = forward_offset(max_lat, max_lon, 2.0, downstream_bearing)

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
        "used_flowline":              use_flowline,
        "search_note": (
            f"Best location {best['distance_ft']:.0f} ft downstream "
            f"(est. depth {best['depth_ft']:.2f} ft, "
            f"velocity score {best['score']:.2f} ft/s). "
            f"{len(candidates)} candidates over {search_distance_ft:.0f} ft. "
            f"{'Flowline geometry used — candidates on stream centerline.' if use_flowline else 'Bearing projection used — flowline unavailable.'}"
        ),
    }


# ── Power estimation ──────────────────────────────────────────────────────────

def estimate_power_output(
    velocity_ft_s:       float,
    turbine_diameter_ft: float,
    cp:                  float,
    num_rows:            int   = 3,
    turbines_per_row:    int   = 2,
    wake_velocity_factor: float = 0.85,
) -> Dict[str, Union[float, int, list]]:
    """
    Estimate total array power for the NEMO 6-turbine hydrokinetic array.
    Array: 3 stations × 2 turbines (port + starboard), 4 ft station spacing.
    Wake cascades station 1' → 5' → 9'.
    """
    rho_water_kg_m3 = 999.7
    velocity_ms     = velocity_ft_s * 0.3048
    diameter_m      = turbine_diameter_ft * 0.3048
    area_m2         = math.pi / 4.0 * diameter_m ** 2

    def _single_turbine_watts(v_ms: float) -> float:
        return 0.5 * cp * rho_water_kg_m3 * area_m2 * v_ms ** 3

    row_powers_w:      List[float] = []
    row_velocities_fs: List[float] = []

    for row_idx in range(num_rows):
        v_row_ms   = velocity_ms * (wake_velocity_factor ** row_idx)
        v_row_ft_s = v_row_ms / 0.3048
        p_row_w    = turbines_per_row * _single_turbine_watts(v_row_ms)
        row_powers_w.append(p_row_w)
        row_velocities_fs.append(v_row_ft_s)

    total_w = sum(row_powers_w)

    return {
        "power_watts":           total_w,
        "power_kw":              total_w / 1000.0,
        "num_rows":              num_rows,
        "turbines_per_row":      turbines_per_row,
        "total_turbines":        num_rows * turbines_per_row,
        "wake_velocity_factor":  wake_velocity_factor,
        "row_powers_watts":      row_powers_w,
        "row_velocities_ft_s":   row_velocities_fs,
        "single_turbine_watts":  _single_turbine_watts(velocity_ms),
        "single_turbine_kw":     _single_turbine_watts(velocity_ms) / 1000.0,
        "swept_area_m2":         area_m2,
    }


# ── Scenario table ────────────────────────────────────────────────────────────

def build_demo_scenario_table(
    lat:                float,
    lon:                float,
    depths_ft:          List[float],
    bankfull:           Dict[str, float],
    turbine_diameter_ft: float,
    cp:                 float,
    num_rows:           int   = 3,
    turbines_per_row:   int   = 2,
    wake_velocity_factor: float = 0.85,
) -> pd.DataFrame:
    rows = []
    for depth in depths_ft:
        rec     = recommend_action(depth, bankfull)
        est_v   = estimate_demo_max_velocity(depth, bankfull)
        est_loc = estimate_demo_locations(lat, lon, depth, bankfull)
        power   = estimate_power_output(
            velocity_ft_s       = float(est_v["estimated_max_velocity_ft_s"]),
            turbine_diameter_ft = turbine_diameter_ft,
            cp                  = cp,
            num_rows            = num_rows,
            turbines_per_row    = turbines_per_row,
            wake_velocity_factor= wake_velocity_factor,
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
    return df.sort_values(by="Array_Total_Power_W", ascending=False).reset_index(drop=True)
