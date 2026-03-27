from __future__ import annotations

from dataclasses import dataclass, field
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
    drainage_area_sqmi:     Optional[float]
    comid:                  Optional[str]
    reachcode:              Optional[str]
    stream_name:            Optional[str]
    source:                 str
    notes:                  str
    debug_nldi_tot_excerpt:      str
    debug_streamstats_excerpt:   str
    reach_elevations:       Optional[List[float]] = None   # ft, sampled ALONG flowline
    reach_distances:        Optional[List[float]] = None   # ft, distance from arc position
    reach_slope:            Optional[float]       = None   # ft/ft, average reach slope
    downstream_bearing:     float                 = 155.0  # degrees
    flowline_coords:        Optional[List[Tuple[float, float]]] = None
    # flowline_coords: (lat, lon) pairs ordered upstream→downstream
    # Every candidate produced by estimate_demo_locations() is interpolated
    # from these vertices — guaranteed to lie on the stream centerline.


# ── Utilities ─────────────────────────────────────────────────────────────────

def safe_get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        return response.json()
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
    """Haversine distance in feet between two geodetic points."""
    R   = 6_371_000.0
    φ1  = math.radians(lat1)
    φ2  = math.radians(lat2)
    dφ  = math.radians(lat2 - lat1)
    dλ  = math.radians(lon2 - lon1)
    a   = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a)) / FT_TO_M


def _bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Forward azimuth (degrees, 0–360) from point 1 → point 2."""
    dlon  = math.radians(lon2 - lon1)
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _forward_offset_local(
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


# ── NHDPlus flowline geometry ─────────────────────────────────────────────────

def fetch_flowline_geometry(comid: str) -> Optional[List[Tuple[float, float]]]:
    """
    Fetch NHDPlus flowline geometry for a COMID from the NLDI feature endpoint.

    Returns a list of (lat, lon) tuples representing the stream centerline
    vertices in upstream → downstream order, or None if unavailable.

    GeoJSON coordinates are stored as [lon, lat]; this function converts them
    to (lat, lon) for consistency with the rest of the codebase.
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
            # GeoJSON: [lon, lat] → convert to (lat, lon)
            return [(float(c[1]), float(c[0])) for c in coords]

    except Exception:
        pass

    return None


def bearing_from_flowline(
    flowline_coords: List[Tuple[float, float]],
    arc_lat: float,
    arc_lon: float,
) -> float:
    """
    Derive the downstream bearing at the arc position from flowline geometry.

    Algorithm:
      1. Find the nearest flowline vertex to (arc_lat, arc_lon).
      2. Use the next two downstream vertices to compute the local bearing.
      3. Falls back to south (180°) if geometry is degenerate.
    """
    if len(flowline_coords) < 2:
        return 180.0

    # Nearest vertex
    min_d = float("inf")
    idx   = 0
    for i, (vlat, vlon) in enumerate(flowline_coords):
        d = _haversine_ft(arc_lat, arc_lon, vlat, vlon)
        if d < min_d:
            min_d = d
            idx   = i

    n = len(flowline_coords)

    # Try both directions; pick the one pointing generally downstream
    # (NHDPlus ordering is ambiguous — we resolve by taking the larger
    # southward component, since the Cullowhee Creek reach runs NNW).
    def _try_bearing(i: int, j: int) -> Optional[float]:
        if 0 <= i < n and 0 <= j < n:
            return _bearing_between(
                flowline_coords[i][0], flowline_coords[i][1],
                flowline_coords[j][0], flowline_coords[j][1],
            )
        return None

    fwd = _try_bearing(idx, idx + 1)
    bwd = _try_bearing(idx, idx - 1)

    # Pick whichever departs from due-north (0°/360°) more toward a
    # southward or stream-like direction; use southward half-plane test.
    def _southward_score(b: Optional[float]) -> float:
        if b is None:
            return -1.0
        # Score = cos(b - 180); peaks at south (180°), negative at north
        return math.cos(math.radians(b - 180))

    candidates = [(fwd, _southward_score(fwd)), (bwd, _southward_score(bwd))]
    best = max(candidates, key=lambda x: x[1])
    return best[0] if best[0] is not None else 180.0


# ── NLDI / StreamStats lookups ────────────────────────────────────────────────

def get_nldi_comid(
    lat: float, lon: float
) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
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
    number    = _extract_number(value)
    if number is None:
        return None
    key_lower = key.lower()
    if key_lower in {"drnarea","drainage_area","areasqmi","da","da_sqmi","totdasqmi",
                     "tot_drainage_area_sqmi","tot_basin_area"}:
        return number
    if key_lower in {"totdasqkm","areasqkm","da_sqkm","tot_drainage_area_sqkm","catchmentareasqkm"}:
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


def get_drainage_area_from_nldi_tot(
    comid: str,
) -> Tuple[Optional[float], str, str]:
    url    = f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}/tot"
    params = {"f": "json"}
    data   = safe_get_json(url, params=params)
    excerpt = json_excerpt(data)
    if not data:
        return None, "NLDI accumulated characteristics lookup failed", excerpt
    da = extract_drainage_area_from_payload(data)
    if da is not None:
        return da, f"Drainage area found from NLDI accumulated characteristics: {da:.3f} mi²", excerpt
    return None, "NLDI accumulated characteristics did not contain a recognized drainage-area field", excerpt


def get_streamstats_drainage_area(
    lat: float, lon: float
) -> Tuple[Optional[float], str, str]:
    candidate_calls = [
        (
            "https://streamstats.usgs.gov/streamstatsservices/watershed.geojson",
            {"rcode": "NC", "xlocation": lon, "ylocation": lat, "crs": 4326, "includeparameters": "true"},
        ),
        (
            "https://streamstats.usgs.gov/streamstatsservices/parameters.json",
            {"rcode": "NC", "xlocation": lon, "ylocation": lat, "crs": 4326},
        ),
    ]
    notes, excerpts = [], []
    for url, params in candidate_calls:
        data = safe_get_json(url, params=params)
        excerpts.append(f"URL: {url}\n{json_excerpt(data, max_chars=1500)}")
        if not data:
            notes.append(f"Failed: {url}")
            continue
        da = extract_drainage_area_from_payload(data)
        if da is not None:
            return da, f"Drainage area found from StreamStats: {da:.3f} mi²", "\n\n".join(excerpts)
        notes.append(f"No recognized drainage-area field in response: {url}")
    return None, " | ".join(notes) if notes else "StreamStats lookup failed", "\n\n".join(excerpts)


# ── Elevation sampling ─────────────────────────────────────────────────────────

def get_elevation_ft(lat: float, lon: float) -> Optional[float]:
    """Query USGS 3DEP for ground elevation at (lat, lon). Returns feet."""
    url    = "https://epqs.nationalmap.gov/v1/json"
    params = {"x": lon, "y": lat, "wkid": 4326, "units": "Feet", "includeDate": False}
    try:
        resp = requests.get(url, params=params, timeout=10, headers={"Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        val  = data.get("value") or data.get("elevation")
        return float(val) if val is not None else None
    except Exception:
        return None


def sample_elevations_along_flowline(
    arc_lat:          float,
    arc_lon:          float,
    flowline_coords:  List[Tuple[float, float]],
    downstream_bearing: float,
    n_samples:        int   = 13,
    total_ft:         float = 300.0,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Sample 3DEP elevations at n_samples points along the NHDPlus flowline
    starting from (arc_lat, arc_lon) and walking downstream.

    When flowline_coords is available the sample points are interpolated
    along the actual stream centerline.  If the flowline is too short or
    unavailable, points are projected along downstream_bearing as a fallback.

    Returns
    -------
    (distances_ft, elevations_ft) — both lists of length n_samples, or
    (None, None) if any elevation query fails.
    """
    step_ft = total_ft / (n_samples - 1)

    # ── Build sample positions ────────────────────────────────────────────────
    sample_points: List[Tuple[float, float, float]] = []   # (dist_ft, lat, lon)

    if flowline_coords and len(flowline_coords) >= 2:
        # Find nearest vertex to arc position
        min_d = float("inf")
        start_idx = 0
        for i, (vlat, vlon) in enumerate(flowline_coords):
            d = _haversine_ft(arc_lat, arc_lon, vlat, vlon)
            if d < min_d:
                min_d = d
                start_idx = i

        n = len(flowline_coords)

        # Determine downstream direction using bearing hint
        def _try_b(i: int, j: int) -> Optional[float]:
            if 0 <= i < n and 0 <= j < n:
                return _bearing_between(
                    flowline_coords[i][0], flowline_coords[i][1],
                    flowline_coords[j][0], flowline_coords[j][1],
                )
            return None

        fwd_b = _try_b(start_idx, start_idx + 1)
        bwd_b = _try_b(start_idx, start_idx - 1)

        def _diff(b: Optional[float]) -> float:
            if b is None:
                return 360.0
            return abs(((b - downstream_bearing + 180) % 360) - 180)

        if (_diff(fwd_b) <= _diff(bwd_b)):
            segment = flowline_coords[start_idx:]
        else:
            segment = flowline_coords[start_idx::-1]

        # Build cumulative distance along segment
        cum: List[float] = [0.0]
        for i in range(1, len(segment)):
            cum.append(cum[-1] + _haversine_ft(
                segment[i-1][0], segment[i-1][1],
                segment[i][0],   segment[i][1],
            ))

        seg_len_ft = cum[-1]
        last_bearing = (
            _bearing_between(
                segment[-2][0], segment[-2][1],
                segment[-1][0], segment[-1][1],
            )
            if len(segment) >= 2 else downstream_bearing
        )

        for i in range(n_samples):
            dist_ft = round(i * step_ft, 2)
            if dist_ft <= seg_len_ft:
                # Interpolate along flowline
                j = 0
                while j < len(cum) - 2 and cum[j + 1] < dist_ft:
                    j += 1
                if cum[j + 1] > cum[j]:
                    t   = (dist_ft - cum[j]) / (cum[j + 1] - cum[j])
                    lat = segment[j][0] + t * (segment[j + 1][0] - segment[j][0])
                    lon = segment[j][1] + t * (segment[j + 1][1] - segment[j][1])
                else:
                    lat, lon = segment[j]
            else:
                # Flowline exhausted — extend along final bearing
                overshoot_m = (dist_ft - seg_len_ft) * FT_TO_M
                lat, lon = _forward_offset_local(
                    segment[-1][0], segment[-1][1], overshoot_m, last_bearing
                )
            sample_points.append((dist_ft, lat, lon))
    else:
        # Fallback: straight-line bearing projection
        for i in range(n_samples):
            dist_ft = round(i * step_ft, 2)
            dist_m  = dist_ft * FT_TO_M
            s_lat, s_lon = _forward_offset_local(arc_lat, arc_lon, dist_m, downstream_bearing)
            sample_points.append((dist_ft, s_lat, s_lon))

    # ── Parallel 3DEP queries ─────────────────────────────────────────────────
    results: Dict[int, Optional[float]] = {}

    def _fetch(idx: int, s_lat: float, s_lon: float) -> Tuple[int, Optional[float]]:
        return idx, get_elevation_ft(s_lat, s_lon)

    with ThreadPoolExecutor(max_workers=n_samples) as executor:
        futures = {
            executor.submit(_fetch, i, pt[1], pt[2]): i
            for i, pt in enumerate(sample_points)
        }
        for future in as_completed(futures):
            idx, elev = future.result()
            results[idx] = elev

    distances  = [sample_points[i][0] for i in range(n_samples)]
    elevations = [results.get(i) for i in range(n_samples)]

    if any(e is None for e in elevations):
        return None, None

    return distances, elevations


def get_downstream_bearing(
    arc_lat:       float,
    arc_lon:       float,
    comid:         Optional[str] = None,
    n_candidates:  int   = 8,
    probe_dist_ft: float = 150.0,
) -> float:
    """
    Kept for backward compatibility.
    Prefers NHD flowline geometry; falls back to elevation probes.
    Use build_hydro_context() which also exposes flowline_coords.
    """
    if comid:
        coords = fetch_flowline_geometry(comid)
        if coords:
            return bearing_from_flowline(coords, arc_lat, arc_lon)

    # Fallback: elevation probes constrained to southward arc
    dist_m   = probe_dist_ft * FT_TO_M
    arc_elev = get_elevation_ft(arc_lat, arc_lon)
    if arc_elev is None:
        return 180.0

    bearings = [90.0 + i * (180.0 / (n_candidates - 1)) for i in range(n_candidates)]

    def _probe(bearing: float) -> Tuple[float, Optional[float]]:
        p_lat, p_lon = _forward_offset_local(arc_lat, arc_lon, dist_m, bearing)
        return bearing, get_elevation_ft(p_lat, p_lon)

    drops: List[Tuple[float, float]] = []
    with ThreadPoolExecutor(max_workers=n_candidates) as ex:
        futures = {ex.submit(_probe, b): b for b in bearings}
        for future in as_completed(futures):
            bearing, elev = future.result()
            if elev is not None:
                drops.append((bearing, arc_elev - elev))

    return max(drops, key=lambda x: x[1])[0] if drops else 180.0


# ── Main entry point ──────────────────────────────────────────────────────────

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
    drainage_area_sqmi = None
    source             = "No automatic drainage area available"
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

    # 4. NHDPlus flowline geometry (single API call — reused for bearing + elevation sampling)
    flowline_coords: Optional[List[Tuple[float, float]]] = None
    downstream_bearing = 180.0   # default south

    if comid:
        flowline_coords = fetch_flowline_geometry(comid)
        if flowline_coords:
            downstream_bearing = bearing_from_flowline(flowline_coords, lat, lon)
            notes.append(
                f"Flowline geometry fetched: {len(flowline_coords)} vertices. "
                f"Downstream bearing from flowline: {downstream_bearing:.1f}°"
            )
        else:
            notes.append("Flowline geometry unavailable — falling back to elevation probe bearing")
            downstream_bearing = get_downstream_bearing(lat, lon, comid=None)
            notes.append(f"Elevation-probe bearing: {downstream_bearing:.1f}°")
    else:
        notes.append("No COMID — falling back to elevation probe bearing")
        downstream_bearing = get_downstream_bearing(lat, lon, comid=None)
        notes.append(f"Elevation-probe bearing: {downstream_bearing:.1f}°")

    # 5. Sample 3DEP elevations ALONG the flowline (not along straight bearing)
    reach_distances, reach_elevations = sample_elevations_along_flowline(
        arc_lat           = lat,
        arc_lon           = lon,
        flowline_coords   = flowline_coords,
        downstream_bearing= downstream_bearing,
        n_samples         = 13,
        total_ft          = 300.0,
    )

    reach_slope = None
    if reach_elevations is not None and reach_distances is not None:
        total_drop  = reach_elevations[0] - reach_elevations[-1]
        reach_slope = max(0.0, total_drop / reach_distances[-1]) if reach_distances[-1] > 0 else None
        notes.append(f"3DEP elevation sampled along flowline: slope={reach_slope:.5f} ft/ft")
    else:
        notes.append("3DEP elevation sampling failed — using estimated bed profile")

    return HydroContext(
        drainage_area_sqmi       = drainage_area_sqmi,
        comid                    = comid,
        reachcode                = reachcode,
        stream_name              = stream_name,
        source                   = source,
        notes                    = " | ".join(notes),
        debug_nldi_tot_excerpt   = debug_nldi_tot_excerpt,
        debug_streamstats_excerpt= debug_streamstats_excerpt,
        reach_elevations         = reach_elevations,
        reach_distances          = reach_distances,
        reach_slope              = reach_slope,
        downstream_bearing       = downstream_bearing,
        flowline_coords          = flowline_coords,
    )
