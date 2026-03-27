from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import math
import requests


REQUEST_TIMEOUT = 25
SQKM_TO_SQMI   = 0.3861021585424458
FT_TO_M         = 0.3048


@dataclass
class HydroContext:
    drainage_area_sqmi:       Optional[float]
    comid:                    Optional[str]
    reachcode:                Optional[str]
    stream_name:              Optional[str]
    source:                   str
    notes:                    str
    debug_nldi_tot_excerpt:   str
    debug_streamstats_excerpt: str
    reach_elevations:         Optional[List[float]] = None
    reach_distances:          Optional[List[float]] = None
    reach_slope:              Optional[float]       = None
    downstream_bearing:       float                 = 155.0
    flowline_coords:          Optional[List[Tuple[float, float]]] = None
    # flowline_coords: (lat, lon) pairs — stream centerline vertices


# ── Utilities ──────────────────────────────────────────────────────────────────

def safe_get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT,
                         headers={"Accept": "application/json"})
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def json_excerpt(data: Optional[Dict[str, Any]], max_chars: int = 2500) -> str:
    if data is None:
        return "None"
    try:
        text = json.dumps(data, indent=2)
        return text[:max_chars] + "\n... [truncated]" if len(text) > max_chars else text
    except Exception:
        return str(data)


def _haversine_ft(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R  = 6_371_000.0
    φ1 = math.radians(lat1);  φ2 = math.radians(lat2)
    dφ = math.radians(lat2 - lat1);  dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * R * math.asin(math.sqrt(a)) / FT_TO_M


def _bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlon  = math.radians(lon2 - lon1)
    lat1r = math.radians(lat1);  lat2r = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r)*math.sin(lat2r) - math.sin(lat1r)*math.cos(lat2r)*math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _angle_diff(a: float, b: float) -> float:
    return abs(((a - b + 180) % 360) - 180)


def _forward_offset(lat: float, lon: float, distance_m: float, bearing_deg: float) -> Tuple[float, float]:
    R    = 6_371_000.0
    lat1 = math.radians(lat);  lon1 = math.radians(lon)
    brng = math.radians(bearing_deg)
    lat2 = math.asin(math.sin(lat1)*math.cos(distance_m/R)
                     + math.cos(lat1)*math.sin(distance_m/R)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(distance_m/R)*math.cos(lat1),
                              math.cos(distance_m/R) - math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


# ── Elevation ──────────────────────────────────────────────────────────────────

def get_elevation_ft(lat: float, lon: float) -> Optional[float]:
    """Query USGS 3DEP for ground elevation at (lat, lon). Returns feet."""
    url    = "https://epqs.nationalmap.gov/v1/json"
    params = {"x": lon, "y": lat, "wkid": 4326, "units": "Feet", "includeDate": False}
    try:
        r = requests.get(url, params=params, timeout=10, headers={"Accept": "application/json"})
        r.raise_for_status()
        data = r.json()
        val  = data.get("value") or data.get("elevation")
        return float(val) if val is not None else None
    except Exception:
        return None


def _verify_bearing_downhill(
    arc_lat: float,
    arc_lon: float,
    bearing: float,
    probe_ft: float = 150.0,
) -> float:
    """
    Definitive downstream direction check.

    Probes elevation at probe_ft in both `bearing` and `bearing + 180°`.
    Returns whichever direction is DOWNHILL (lower elevation).
    If 3DEP is unavailable, returns bearing unchanged.

    This is the final authority on direction — overrides any flowline
    ordering ambiguity or probe heuristic.
    """
    arc_elev = get_elevation_ft(arc_lat, arc_lon)
    if arc_elev is None:
        return bearing

    fwd_lat, fwd_lon = _forward_offset(arc_lat, arc_lon, probe_ft * FT_TO_M, bearing)
    rev_lat, rev_lon = _forward_offset(arc_lat, arc_lon, probe_ft * FT_TO_M, (bearing + 180) % 360)

    fwd_elev = get_elevation_ft(fwd_lat, fwd_lon)
    rev_elev = get_elevation_ft(rev_lat, rev_lon)

    if fwd_elev is None and rev_elev is None:
        return bearing
    if fwd_elev is None:
        return (bearing + 180) % 360
    if rev_elev is None:
        return bearing

    # Pick the direction that goes DOWNHILL from the arc position
    fwd_drop = arc_elev - fwd_elev   # positive = downhill
    rev_drop = arc_elev - rev_elev

    if rev_drop > fwd_drop:
        # Reverse direction is more downhill — flip bearing 180°
        return (bearing + 180) % 360
    return bearing


# ── NHDPlus flowline geometry ──────────────────────────────────────────────────

def fetch_flowline_geometry(comid: str) -> Optional[List[Tuple[float, float]]]:
    """
    Fetch NHDPlus flowline centerline vertices for a COMID.
    Returns (lat, lon) list in stored order.
    Direction is ambiguous at this stage — resolved later by _verify_bearing_downhill().
    """
    url  = f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}"
    data = safe_get_json(url)
    if not data:
        return None
    try:
        coords = None
        geom   = data.get("geometry") or {}
        if geom.get("type") == "LineString":
            coords = geom["coordinates"]
        else:
            for feat in data.get("features", []):
                g = feat.get("geometry", {})
                if g.get("type") == "LineString":
                    coords = g["coordinates"]
                    break
        if coords and len(coords) >= 2:
            return [(float(c[1]), float(c[0])) for c in coords]  # [lon,lat] → (lat,lon)
    except Exception:
        pass
    return None


def bearing_from_flowline(
    flowline_coords: List[Tuple[float, float]],
    arc_lat: float,
    arc_lon: float,
) -> float:
    """
    Extract the local bearing of the flowline at the arc position.
    Returns the raw bearing of the nearest adjacent segment —
    direction correctness is NOT guaranteed here.
    Call _verify_bearing_downhill() after this to resolve upstream vs downstream.
    """
    if len(flowline_coords) < 2:
        return 180.0

    n = len(flowline_coords)

    # Nearest vertex
    min_d, idx = float("inf"), 0
    for i, (vlat, vlon) in enumerate(flowline_coords):
        d = _haversine_ft(arc_lat, arc_lon, vlat, vlon)
        if d < min_d:
            min_d, idx = d, i

    # Use a multi-vertex window for a more stable bearing estimate
    # Take the bearing of the segment spanning ±1 vertex around the nearest
    i1 = max(0, idx - 1)
    i2 = min(n - 1, idx + 1)
    return _bearing_between(
        flowline_coords[i1][0], flowline_coords[i1][1],
        flowline_coords[i2][0], flowline_coords[i2][1],
    )


# ── NLDI / StreamStats lookups ─────────────────────────────────────────────────

def get_nldi_comid(lat: float, lon: float) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    url    = "https://api.water.usgs.gov/nldi/linked-data/comid/position"
    params = {"coords": f"POINT({lon} {lat})", "f": "json"}
    data   = safe_get_json(url, params=params)
    if not data:
        return None, None, None, "NLDI position lookup failed"
    try:
        features = data.get("features", [])
        if not features:
            return None, None, None, "NLDI returned no matching features"
        props     = features[0].get("properties", {})
        comid     = str(props.get("identifier") or props.get("comid") or "")
        reachcode = str(props.get("reachcode") or "")
        name      = props.get("name") or props.get("gnis_name") or ""
        return comid or None, reachcode or None, name or None, "NLDI position lookup succeeded"
    except Exception:
        return None, None, None, "NLDI response parse failed"


def lookup_stream_name_from_comid(comid: str) -> str:
    url  = f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}"
    data = safe_get_json(url)
    if not data:
        return "Unnamed stream"
    try:
        props = data.get("properties", {})
        name  = (props.get("gnis_name") or props.get("name") or "").strip()
        if name:
            return name
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            name  = (props.get("gnis_name") or props.get("name") or "").strip()
            if name:
                return name
    except Exception:
        pass
    return "Unnamed stream"


def _extract_number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _convert_key_value_to_sqmi(key: str, value: Any) -> Optional[float]:
    number = _extract_number(value)
    if number is None:
        return None
    k = key.lower()
    if k in {"drnarea","drainage_area","areasqmi","da","da_sqmi","totdasqmi",
             "tot_drainage_area_sqmi","tot_basin_area"}:
        return number
    if k in {"totdasqkm","areasqkm","da_sqkm","tot_drainage_area_sqkm","catchmentareasqkm"}:
        return number * SQKM_TO_SQMI
    return None


def extract_drainage_area_from_payload(data: Dict[str, Any]) -> Optional[float]:
    direct_keys = [
        "DRNAREA","drnarea","drainage_area","AreaSqMi","areasqmi","DA","da_sqmi",
        "TOTDASQKM","totdasqkm","AreaSqKm","areasqkm","da_sqkm","TOT_BASIN_AREA","tot_basin_area",
    ]
    for key in direct_keys:
        if key in data:
            v = _convert_key_value_to_sqmi(key, data[key])
            if v is not None:
                return v
    features = data.get("features")
    if isinstance(features, list):
        for feat in features:
            props = feat.get("properties", {})
            for key in direct_keys:
                if key in props:
                    v = _convert_key_value_to_sqmi(key, props[key])
                    if v is not None:
                        return v
            prop_name  = str(props.get("name") or props.get("characteristic_id") or props.get("id") or "").lower()
            prop_value = props.get("value") if props.get("value") is not None else props.get("characteristic_value")
            v = _convert_key_value_to_sqmi(prop_name, prop_value)
            if v is not None:
                return v
    for top_key in ["parameters","parametersList","results","workspace","messages","characteristics"]:
        block = data.get(top_key)
        if isinstance(block, list):
            for item in block:
                if not isinstance(item, dict):
                    continue
                code = str(item.get("code") or item.get("name") or item.get("characteristic_id") or item.get("id") or "")
                item_value = item.get("value") if item.get("value") is not None else item.get("characteristic_value")
                v = _convert_key_value_to_sqmi(code, item_value)
                if v is not None:
                    return v
                for key in direct_keys:
                    if key in item:
                        v = _convert_key_value_to_sqmi(key, item[key])
                        if v is not None:
                            return v
        elif isinstance(block, dict):
            for key in direct_keys:
                if key in block:
                    v = _convert_key_value_to_sqmi(key, block[key])
                    if v is not None:
                        return v
    return None


def get_drainage_area_from_nldi_tot(comid: str) -> Tuple[Optional[float], str, str]:
    url    = f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}/tot"
    data   = safe_get_json(url, params={"f": "json"})
    excerpt = json_excerpt(data)
    if not data:
        return None, "NLDI accumulated characteristics lookup failed", excerpt
    da = extract_drainage_area_from_payload(data)
    if da is not None:
        return da, f"Drainage area found from NLDI: {da:.3f} mi²", excerpt
    return None, "NLDI accumulated characteristics did not contain a recognized drainage-area field", excerpt


def get_streamstats_drainage_area(lat: float, lon: float) -> Tuple[Optional[float], str, str]:
    calls = [
        ("https://streamstats.usgs.gov/streamstatsservices/watershed.geojson",
         {"rcode": "NC", "xlocation": lon, "ylocation": lat, "crs": 4326, "includeparameters": "true"}),
        ("https://streamstats.usgs.gov/streamstatsservices/parameters.json",
         {"rcode": "NC", "xlocation": lon, "ylocation": lat, "crs": 4326}),
    ]
    notes, excerpts = [], []
    for url, params in calls:
        data = safe_get_json(url, params=params)
        excerpts.append(f"URL: {url}\n{json_excerpt(data, max_chars=1500)}")
        if not data:
            notes.append(f"Failed: {url}"); continue
        da = extract_drainage_area_from_payload(data)
        if da is not None:
            return da, f"Drainage area from StreamStats: {da:.3f} mi²", "\n\n".join(excerpts)
        notes.append(f"No drainage-area field: {url}")
    return None, " | ".join(notes) or "StreamStats lookup failed", "\n\n".join(excerpts)


# ── Elevation corridor sampling ────────────────────────────────────────────────

def sample_elevations_along_flowline(
    arc_lat:           float,
    arc_lon:           float,
    flowline_coords:   Optional[List[Tuple[float, float]]],
    downstream_bearing: float,
    n_samples:         int   = 13,
    total_ft:          float = 300.0,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Sample 3DEP elevations at n_samples points walking DOWNSTREAM along
    the NHDPlus flowline.  downstream_bearing has already been verified
    downhill by _verify_bearing_downhill() before this is called.
    """
    step_ft = total_ft / (n_samples - 1)
    sample_points: List[Tuple[float, float, float]] = []

    if flowline_coords and len(flowline_coords) >= 2:
        n = len(flowline_coords)

        # Nearest vertex to arc position
        min_d, start_idx = float("inf"), 0
        for i, (vlat, vlon) in enumerate(flowline_coords):
            d = _haversine_ft(arc_lat, arc_lon, vlat, vlon)
            if d < min_d:
                min_d, start_idx = d, i

        # Resolve downstream direction using verified bearing
        fwd_b = (_bearing_between(
                    flowline_coords[start_idx][0], flowline_coords[start_idx][1],
                    flowline_coords[start_idx+1][0], flowline_coords[start_idx+1][1])
                 if start_idx < n-1 else None)
        bwd_b = (_bearing_between(
                    flowline_coords[start_idx][0], flowline_coords[start_idx][1],
                    flowline_coords[start_idx-1][0], flowline_coords[start_idx-1][1])
                 if start_idx > 0 else None)

        fwd_diff = _angle_diff(fwd_b, downstream_bearing) if fwd_b is not None else 360.0
        bwd_diff = _angle_diff(bwd_b, downstream_bearing) if bwd_b is not None else 360.0

        segment: List[Tuple[float, float]] = (
            flowline_coords[start_idx:] if fwd_diff <= bwd_diff
            else flowline_coords[start_idx::-1]
        )

        cum: List[float] = [0.0]
        for i in range(1, len(segment)):
            cum.append(cum[-1] + _haversine_ft(
                segment[i-1][0], segment[i-1][1],
                segment[i][0],   segment[i][1],
            ))
        flowline_len_ft = cum[-1]
        last_bearing    = (_bearing_between(
                               segment[-2][0], segment[-2][1],
                               segment[-1][0], segment[-1][1])
                           if len(segment) >= 2 else downstream_bearing)
        last_lat, last_lon = segment[-1]

        for i in range(n_samples):
            dist_ft = round(i * step_ft, 2)
            if dist_ft <= flowline_len_ft:
                j = 0
                while j < len(cum)-2 and cum[j+1] < dist_ft:
                    j += 1
                seg_len = cum[j+1] - cum[j]
                t = (dist_ft - cum[j]) / seg_len if seg_len > 0 else 0.0
                slat = segment[j][0] + t*(segment[j+1][0] - segment[j][0])
                slon = segment[j][1] + t*(segment[j+1][1] - segment[j][1])
            else:
                overshoot_m = (dist_ft - flowline_len_ft) * FT_TO_M
                slat, slon  = _forward_offset(last_lat, last_lon, overshoot_m, last_bearing)
            sample_points.append((dist_ft, slat, slon))
    else:
        for i in range(n_samples):
            dist_ft = round(i * step_ft, 2)
            slat, slon = _forward_offset(arc_lat, arc_lon, dist_ft*FT_TO_M, downstream_bearing)
            sample_points.append((dist_ft, slat, slon))

    results: Dict[int, Optional[float]] = {}

    def _fetch(idx: int, s_lat: float, s_lon: float) -> Tuple[int, Optional[float]]:
        return idx, get_elevation_ft(s_lat, s_lon)

    with ThreadPoolExecutor(max_workers=n_samples) as executor:
        futures = {executor.submit(_fetch, i, pt[1], pt[2]): i
                   for i, pt in enumerate(sample_points)}
        for future in as_completed(futures):
            idx, elev = future.result()
            results[idx] = elev

    distances  = [sample_points[i][0] for i in range(n_samples)]
    elevations = [results.get(i) for i in range(n_samples)]
    if any(e is None for e in elevations):
        return None, None
    return distances, elevations


# ── Main entry point ───────────────────────────────────────────────────────────

def build_hydro_context(lat: float, lon: float) -> HydroContext:
    notes = []

    # 1. COMID lookup
    comid, reachcode, stream_name_raw, nldi_note = get_nldi_comid(lat, lon)
    notes.append(nldi_note)

    # 2. Stream name
    if comid:
        stream_name = lookup_stream_name_from_comid(comid)
        if stream_name == "Unnamed stream" and stream_name_raw:
            stream_name = stream_name_raw
    else:
        stream_name = stream_name_raw or "Unnamed stream"

    # 3. Drainage area
    drainage_area_sqmi        = None
    source                    = "No automatic drainage area available"
    debug_nldi_tot_excerpt    = "Not requested"
    debug_streamstats_excerpt = "Not requested"

    if comid:
        drainage_area_sqmi, tot_note, debug_nldi_tot_excerpt = get_drainage_area_from_nldi_tot(comid)
        notes.append(tot_note)
        if drainage_area_sqmi is not None:
            source = "USGS NLDI accumulated characteristics"

    if drainage_area_sqmi is None:
        ss_da, ss_note, debug_streamstats_excerpt = get_streamstats_drainage_area(lat, lon)
        notes.append(ss_note)
        if ss_da is not None:
            drainage_area_sqmi = ss_da
            source             = "USGS StreamStats"

    # 4. NHDPlus flowline geometry
    flowline_coords: Optional[List[Tuple[float, float]]] = None
    if comid:
        flowline_coords = fetch_flowline_geometry(comid)
        if flowline_coords:
            notes.append(f"Flowline geometry: {len(flowline_coords)} vertices")
        else:
            notes.append("Flowline geometry unavailable")

    # 5. Raw bearing from flowline (direction ambiguous — may be up or downstream)
    if flowline_coords:
        raw_bearing = bearing_from_flowline(flowline_coords, lat, lon)
        notes.append(f"Raw flowline bearing: {raw_bearing:.1f}°")
    else:
        raw_bearing = 180.0   # placeholder — will be corrected by elevation check

    # 6. DEFINITIVE downstream direction via elevation sanity check
    #    Probes 150 ft in both directions; picks whichever goes DOWNHILL.
    #    This is the only reliable method when flowline ordering is ambiguous.
    downstream_bearing = _verify_bearing_downhill(lat, lon, raw_bearing, probe_ft=150.0)
    notes.append(
        f"Downstream bearing after elevation verification: {downstream_bearing:.1f}° "
        f"({'flipped' if abs(_angle_diff(downstream_bearing, raw_bearing)) > 90 else 'confirmed'})"
    )

    # 7. Sample 3DEP elevations along correctly-oriented flowline
    reach_distances, reach_elevations = sample_elevations_along_flowline(
        arc_lat            = lat,
        arc_lon            = lon,
        flowline_coords    = flowline_coords,
        downstream_bearing = downstream_bearing,
        n_samples          = 13,
        total_ft           = 300.0,
    )

    reach_slope = None
    if reach_elevations is not None and reach_distances is not None:
        total_drop  = reach_elevations[0] - reach_elevations[-1]
        reach_slope = max(0.0, total_drop / reach_distances[-1]) if reach_distances[-1] > 0 else None
        notes.append(f"3DEP elevation slope along flowline: {reach_slope:.5f} ft/ft")
    else:
        notes.append("3DEP elevation sampling failed — using estimated bed profile")

    return HydroContext(
        drainage_area_sqmi        = drainage_area_sqmi,
        comid                     = comid,
        reachcode                 = reachcode,
        stream_name               = stream_name,
        source                    = source,
        notes                     = " | ".join(notes),
        debug_nldi_tot_excerpt    = debug_nldi_tot_excerpt,
        debug_streamstats_excerpt = debug_streamstats_excerpt,
        reach_elevations          = reach_elevations,
        reach_distances           = reach_distances,
        reach_slope               = reach_slope,
        downstream_bearing        = downstream_bearing,
        flowline_coords           = flowline_coords,
    )
