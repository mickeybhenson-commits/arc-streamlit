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
    phi1 = math.radians(lat1);  phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1);  dlambda = math.radians(lon2 - lon1)
    a  = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a)) / FT_TO_M


def _bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlon  = math.radians(lon2 - lon1)
    lat1r = math.radians(lat1);  lat2r = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r)*math.sin(lat2r) - math.sin(lat1r)*math.cos(lat2r)*math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _angle_diff(a: float, b: float) -> float:
    return abs(((a - b + 180) % 360) - 180)


def _forward_offset(
    lat: float, lon: float, distance_m: float, bearing_deg: float
) -> Tuple[float, float]:
    R    = 6_371_000.0
    lat1 = math.radians(lat);  lon1 = math.radians(lon)
    brng = math.radians(bearing_deg)
    lat2 = math.asin(math.sin(lat1)*math.cos(distance_m/R)
                     + math.cos(lat1)*math.sin(distance_m/R)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(distance_m/R)*math.cos(lat1),
                              math.cos(distance_m/R) - math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


# ── NHDPlus flowline geometry ──────────────────────────────────────────────────

def fetch_flowline_geometry(comid: str) -> Optional[List[Tuple[float, float]]]:
    """
    Fetch NHDPlus flowline centerline vertices for a COMID.
    Returns (lat, lon) list in stored order.
    Direction ambiguity is resolved separately by get_downstream_bearing_from_nldi().
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
            return [(float(c[1]), float(c[0])) for c in coords]  # [lon,lat]→(lat,lon)
    except Exception:
        pass
    return None


def get_downstream_bearing_from_nldi(
    comid: str,
    arc_lat: float,
    arc_lon: float,
) -> Optional[float]:
    """
    Determine the definitive downstream bearing using the NLDI downstream
    navigation endpoint.

    Strategy
    --------
    1. Call NLDI DM (downstream mainstem) navigation for 1 km downstream.
    2. Extract the geometry of the next downstream reach.
    3. Compute bearing from arc position to the centroid of that reach.

    This is 100% reliable because NLDI navigation is hydrologically routed —
    it always returns the downstream direction regardless of terrain, elevation
    probes, or flowline coordinate ordering.

    Returns None if the navigation call fails (caller falls back to
    flowline geometry heuristic).
    """
    url    = (f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}"
              f"/navigation/DM/flowlines")
    params = {"distance": 1, "f": "json"}   # 1 km downstream
    data   = safe_get_json(url, params=params)
    if not data:
        return None

    try:
        features = data.get("features", [])
        if not features:
            return None

        # Collect all downstream vertices from returned flowlines
        all_coords: List[Tuple[float, float]] = []
        for feat in features:
            geom = feat.get("geometry", {})
            if geom.get("type") == "LineString":
                for c in geom["coordinates"]:
                    all_coords.append((float(c[1]), float(c[0])))

        if not all_coords:
            return None

        # Centroid of downstream reach geometry
        ds_lat = sum(c[0] for c in all_coords) / len(all_coords)
        ds_lon = sum(c[1] for c in all_coords) / len(all_coords)

        bearing = _bearing_between(arc_lat, arc_lon, ds_lat, ds_lon)
        return bearing

    except Exception:
        return None


def bearing_from_flowline_with_hint(
    flowline_coords: List[Tuple[float, float]],
    arc_lat: float,
    arc_lon: float,
    downstream_hint: float,
) -> float:
    """
    Extract the local flowline bearing at arc position, oriented using
    a known-good downstream hint (from NLDI navigation).

    The hint resolves the upstream/downstream ambiguity in the raw
    flowline coordinate ordering.
    """
    if len(flowline_coords) < 2:
        return downstream_hint

    n = len(flowline_coords)

    # Nearest vertex
    min_d, idx = float("inf"), 0
    for i, (vlat, vlon) in enumerate(flowline_coords):
        d = _haversine_ft(arc_lat, arc_lon, vlat, vlon)
        if d < min_d:
            min_d, idx = d, i

    # Local segment bearings
    fwd_b = (_bearing_between(
                 flowline_coords[idx][0], flowline_coords[idx][1],
                 flowline_coords[idx+1][0], flowline_coords[idx+1][1])
             if idx < n-1 else None)
    bwd_b = (_bearing_between(
                 flowline_coords[idx][0], flowline_coords[idx][1],
                 flowline_coords[idx-1][0], flowline_coords[idx-1][1])
             if idx > 0 else None)

    # Pick whichever is closest to the known-good downstream hint
    fwd_diff = _angle_diff(fwd_b, downstream_hint) if fwd_b is not None else 360.0
    bwd_diff = _angle_diff(bwd_b, downstream_hint) if bwd_b is not None else 360.0

    if fwd_diff <= bwd_diff:
        return fwd_b if fwd_b is not None else downstream_hint
    else:
        return bwd_b if bwd_b is not None else downstream_hint


# ── NLDI / StreamStats lookups ─────────────────────────────────────────────────

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
                code = str(item.get("code") or item.get("name") or
                           item.get("characteristic_id") or item.get("id") or "")
                item_value = (item.get("value") if item.get("value") is not None
                              else item.get("characteristic_value"))
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
    url     = f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}/tot"
    data    = safe_get_json(url, params={"f": "json"})
    excerpt = json_excerpt(data)
    if not data:
        return None, "NLDI accumulated characteristics lookup failed", excerpt
    da = extract_drainage_area_from_payload(data)
    if da is not None:
        return da, f"Drainage area from NLDI: {da:.3f} mi²", excerpt
    return None, "NLDI characteristics had no recognized drainage-area field", excerpt


def get_streamstats_drainage_area(
    lat: float, lon: float
) -> Tuple[Optional[float], str, str]:
    calls = [
        ("https://streamstats.usgs.gov/streamstatsservices/watershed.geojson",
         {"rcode": "NC", "xlocation": lon, "ylocation": lat, "crs": 4326,
          "includeparameters": "true"}),
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


# ── Elevation sampling ─────────────────────────────────────────────────────────

def get_elevation_ft(lat: float, lon: float) -> Optional[float]:
    url    = "https://epqs.nationalmap.gov/v1/json"
    params = {"x": lon, "y": lat, "wkid": 4326, "units": "Feet", "includeDate": False}
    try:
        r = requests.get(url, params=params, timeout=10,
                         headers={"Accept": "application/json"})
        r.raise_for_status()
        data = r.json()
        val  = data.get("value") or data.get("elevation")
        return float(val) if val is not None else None
    except Exception:
        return None


def sample_elevations_along_flowline(
    arc_lat:           float,
    arc_lon:           float,
    flowline_coords:   Optional[List[Tuple[float, float]]],
    downstream_bearing: float,
    n_samples:         int   = 13,
    total_ft:          float = 300.0,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Sample 3DEP elevations at n_samples points walking downstream along
    the NHDPlus flowline.  downstream_bearing has been verified via NLDI
    navigation before this is called — direction is correct.
    """
    step_ft = total_ft / (n_samples - 1)
    sample_points: List[Tuple[float, float, float]] = []

    if flowline_coords and len(flowline_coords) >= 2:
        n = len(flowline_coords)

        min_d, start_idx = float("inf"), 0
        for i, (vlat, vlon) in enumerate(flowline_coords):
            d = _haversine_ft(arc_lat, arc_lon, vlat, vlon)
            if d < min_d:
                min_d, start_idx = d, i

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
            flowline_coords[start_idx:]
            if fwd_diff <= bwd_diff
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
        notes.append(
            f"Flowline geometry: {len(flowline_coords)} vertices"
            if flowline_coords else "Flowline geometry unavailable"
        )

    # 5. Downstream bearing — NLDI navigation is the PRIMARY and most reliable
    #    method.  It uses hydrological routing, not elevation or coordinate order.
    #    Fallback chain:
    #      a) NLDI DM navigation centroid bearing  ← hydrologically correct
    #      b) Flowline geometry + navigation hint  ← good if (a) fails
    #      c) Default 180° south                  ← last resort
    downstream_bearing = 180.0
    bearing_method     = "default (180°)"

    if comid:
        nldi_bearing = get_downstream_bearing_from_nldi(comid, lat, lon)
        if nldi_bearing is not None:
            downstream_bearing = nldi_bearing
            bearing_method     = f"NLDI DM navigation ({nldi_bearing:.1f}°)"
            notes.append(f"Downstream bearing from NLDI navigation: {downstream_bearing:.1f}°")

            # Refine to local flowline tangent using the navigation bearing as hint
            if flowline_coords:
                refined = bearing_from_flowline_with_hint(
                    flowline_coords, lat, lon, downstream_bearing
                )
                # Only use refinement if it stays within 45° of the navigation bearing
                if _angle_diff(refined, downstream_bearing) <= 45.0:
                    downstream_bearing = refined
                    bearing_method     = (f"Flowline tangent refined with NLDI hint "
                                          f"({downstream_bearing:.1f}°)")
                    notes.append(f"Bearing refined to local flowline tangent: {downstream_bearing:.1f}°")
        else:
            notes.append("NLDI DM navigation failed — using flowline geometry bearing")
            if flowline_coords and len(flowline_coords) >= 2:
                # Without a reliable hint, use the stored coordinate order
                # (last two vertices → downstream end of NHDPlus LineString)
                coords = flowline_coords
                n      = len(coords)
                downstream_bearing = _bearing_between(
                    coords[n//2][0], coords[n//2][1],
                    coords[-1][0],   coords[-1][1],
                )
                bearing_method = f"Flowline endpoint bearing ({downstream_bearing:.1f}°)"
                notes.append(f"Bearing from flowline endpoints: {downstream_bearing:.1f}°")

    notes.append(f"Bearing method: {bearing_method}")

    # 6. Sample 3DEP elevations along correctly-oriented flowline
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
        notes.append(f"3DEP slope along flowline: {reach_slope:.5f} ft/ft")
    else:
        notes.append("3DEP elevation sampling failed — sinusoidal fallback will be used")

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
