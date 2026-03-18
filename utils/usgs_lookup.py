from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import math
import requests


REQUEST_TIMEOUT = 25
SQKM_TO_SQMI = 0.3861021585424458


@dataclass
class HydroContext:
    drainage_area_sqmi: Optional[float]
    comid: Optional[str]
    reachcode: Optional[str]
    stream_name: Optional[str]
    source: str
    notes: str
    debug_nldi_tot_excerpt: str
    debug_streamstats_excerpt: str
    reach_elevations: Optional[List[float]] = None   # ft, sampled along downstream corridor
    reach_distances: Optional[List[float]] = None    # ft, distance from ARC position
    reach_slope: Optional[float] = None              # ft/ft, average reach slope
    downstream_bearing: float = 155.0                # degrees, auto-detected from 3DEP


def safe_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
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
        if len(text) > max_chars:
            return text[:max_chars] + "\n... [truncated]"
        return text
    except Exception:
        return str(data)


def get_nldi_comid(lat: float, lon: float) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    url = "https://api.water.usgs.gov/nldi/linked-data/comid/position"
    params = {
        "coords": f"POINT({lon} {lat})",
        "f": "json",
    }

    data = safe_get_json(url, params=params)
    if not data:
        return None, None, None, "NLDI position lookup failed"

    try:
        features = data.get("features", [])
        if not features:
            return None, None, None, "NLDI returned no matching features"

        props = features[0].get("properties", {})
        comid = str(props.get("identifier") or props.get("comid") or "")
        reachcode = str(props.get("reachcode") or "")

        # Best-effort name from position endpoint — may be blank for small tributaries.
        # build_hydro_context will follow up with a dedicated COMID name lookup.
        stream_name = props.get("name") or props.get("gnis_name") or ""

        return comid or None, reachcode or None, stream_name or None, "NLDI position lookup succeeded"
    except Exception:
        return None, None, None, "NLDI response parse failed"


def lookup_stream_name_from_comid(comid: str) -> str:
    """
    Fetch the official NHD GNIS stream name for a known COMID.

    Calls the NLDI feature endpoint:
        GET https://api.water.usgs.gov/nldi/linked-data/comid/{comid}

    The returned GeoJSON feature carries a 'gnis_name' property that is
    populated whenever the NHD has an official name for the reach.  Small
    tributaries and ditches that have never been assigned a GNIS name will
    return an empty string — in that case 'Unnamed stream' is returned so
    the UI always shows a human-readable label.

    Parameters
    ----------
    comid : str
        NHDPlus COMID as a string.

    Returns
    -------
    str
        GNIS stream name, or 'Unnamed stream' if blank / lookup fails.
    """
    url = f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}"
    data = safe_get_json(url)
    if not data:
        return "Unnamed stream"

    try:
        # The endpoint returns a single GeoJSON feature (not a FeatureCollection)
        props = data.get("properties", {})
        name = (props.get("gnis_name") or props.get("name") or "").strip()
        if name:
            return name

        # Some responses nest the feature inside a features array
        features = data.get("features", [])
        if features:
            props = features[0].get("properties", {})
            name = (props.get("gnis_name") or props.get("name") or "").strip()
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

    key_lower = key.lower()

    # Already square miles
    if key_lower in {
        "drnarea",
        "drainage_area",
        "areasqmi",
        "da",
        "da_sqmi",
        "totdasqmi",
        "tot_drainage_area_sqmi",
        "tot_basin_area",
    }:
        return number

    # Square kilometers -> convert to square miles
    if key_lower in {
        "totdasqkm",
        "areasqkm",
        "da_sqkm",
        "tot_drainage_area_sqkm",
        "catchmentareasqkm",
    }:
        return number * SQKM_TO_SQMI

    return None


def extract_drainage_area_from_payload(data: Dict[str, Any]) -> Optional[float]:
    direct_keys = [
        "DRNAREA",
        "drnarea",
        "drainage_area",
        "AreaSqMi",
        "areasqmi",
        "DA",
        "da_sqmi",
        "TOTDASQKM",
        "totdasqkm",
        "AreaSqKm",
        "areasqkm",
        "da_sqkm",
        "TOT_BASIN_AREA",
        "tot_basin_area",
    ]

    # Direct top-level keys
    for key in direct_keys:
        if key in data:
            converted = _convert_key_value_to_sqmi(key, data[key])
            if converted is not None:
                return converted

    # GeoJSON feature properties
    features = data.get("features")
    if isinstance(features, list):
        for feat in features:
            props = feat.get("properties", {})
            for key in direct_keys:
                if key in props:
                    converted = _convert_key_value_to_sqmi(key, props[key])
                    if converted is not None:
                        return converted

            prop_name = str(
                props.get("name")
                or props.get("characteristic_id")
                or props.get("id")
                or ""
            ).lower()
            prop_value = (
                props.get("value")
                if props.get("value") is not None
                else props.get("characteristic_value")
            )
            converted = _convert_key_value_to_sqmi(prop_name, prop_value)
            if converted is not None:
                return converted

    # Nested list/dict blocks
    for top_key in ["parameters", "parametersList", "results", "workspace", "messages", "characteristics"]:
        block = data.get(top_key)

        if isinstance(block, list):
            for item in block:
                if not isinstance(item, dict):
                    continue

                code = str(
                    item.get("code")
                    or item.get("name")
                    or item.get("characteristic_id")
                    or item.get("id")
                    or ""
                )

                item_value = (
                    item.get("value")
                    if item.get("value") is not None
                    else item.get("characteristic_value")
                )

                converted = _convert_key_value_to_sqmi(code, item_value)
                if converted is not None:
                    return converted

                for key in direct_keys:
                    if key in item:
                        converted = _convert_key_value_to_sqmi(key, item[key])
                        if converted is not None:
                            return converted

        elif isinstance(block, dict):
            for key in direct_keys:
                if key in block:
                    converted = _convert_key_value_to_sqmi(key, block[key])
                    if converted is not None:
                        return converted

    return None


def get_drainage_area_from_nldi_tot(comid: str) -> Tuple[Optional[float], str, str]:
    url = f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}/tot"
    params = {"f": "json"}

    data = safe_get_json(url, params=params)
    excerpt = json_excerpt(data)

    if not data:
        return None, "NLDI accumulated characteristics lookup failed", excerpt

    drainage_area_sqmi = extract_drainage_area_from_payload(data)
    if drainage_area_sqmi is not None:
        return (
            drainage_area_sqmi,
            f"Drainage area found from NLDI accumulated characteristics: {drainage_area_sqmi:.3f} mi²",
            excerpt,
        )

    return None, "NLDI accumulated characteristics did not contain a recognized drainage-area field", excerpt


def get_streamstats_drainage_area(lat: float, lon: float) -> Tuple[Optional[float], str, str]:
    candidate_calls = [
        (
            "https://streamstats.usgs.gov/streamstatsservices/watershed.geojson",
            {
                "rcode": "NC",
                "xlocation": lon,
                "ylocation": lat,
                "crs": 4326,
                "includeparameters": "true",
            },
        ),
        (
            "https://streamstats.usgs.gov/streamstatsservices/parameters.json",
            {
                "rcode": "NC",
                "xlocation": lon,
                "ylocation": lat,
                "crs": 4326,
            },
        ),
    ]

    notes = []
    excerpts = []

    for url, params in candidate_calls:
        data = safe_get_json(url, params=params)
        excerpts.append(f"URL: {url}\n{json_excerpt(data, max_chars=1500)}")

        if not data:
            notes.append(f"Failed: {url}")
            continue

        drainage_area_sqmi = extract_drainage_area_from_payload(data)
        if drainage_area_sqmi is not None:
            return (
                drainage_area_sqmi,
                f"Drainage area found from StreamStats: {drainage_area_sqmi:.3f} mi²",
                "\n\n".join(excerpts),
            )

        notes.append(f"No recognized drainage-area field in response: {url}")

    return None, " | ".join(notes) if notes else "StreamStats lookup failed", "\n\n".join(excerpts)


def _forward_offset_local(lat: float, lon: float, distance_m: float, bearing_deg: float) -> Tuple[float, float]:
    """Haversine forward offset — duplicated here to avoid circular import."""
    R = 6371000.0
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(bearing_deg)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_m / R) +
        math.cos(lat1) * math.sin(distance_m / R) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(distance_m / R) * math.cos(lat1),
        math.cos(distance_m / R) - math.sin(lat1) * math.sin(lat2)
    )
    return math.degrees(lat2), math.degrees(lon2)


def get_elevation_ft(lat: float, lon: float) -> Optional[float]:
    """
    Query USGS 3DEP Elevation Point Query Service for ground elevation.
    Returns elevation in feet, or None on failure.
    WNC has excellent 1m lidar coverage post-Hurricane Helene.
    """
    url = "https://epqs.nationalmap.gov/v1/json"
    params = {
        "x":           lon,
        "y":           lat,
        "wkid":        4326,
        "units":       "Feet",
        "includeDate": False,
    }
    try:
        resp = requests.get(url, params=params, timeout=10,
                            headers={"Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        val = data.get("value") or data.get("elevation")
        return float(val) if val is not None else None
    except Exception:
        return None


def sample_corridor_elevations(
    arc_lat: float,
    arc_lon: float,
    bearing_deg: float = 155.0,
    n_samples: int = 13,
    total_ft: float = 300.0,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Sample ground elevations at n_samples points along the downstream corridor.

    Points are spaced evenly from 0 to total_ft along bearing_deg.
    Queries run in parallel via ThreadPoolExecutor for speed (~2–4 seconds).

    Returns
    -------
    (distances_ft, elevations_ft) — both lists of length n_samples
    Returns (None, None) if any elevation query fails.
    """
    FT_TO_M   = 0.3048
    step_ft   = total_ft / (n_samples - 1)

    sample_points: List[Tuple[float, float, float]] = []
    for i in range(n_samples):
        dist_ft = round(i * step_ft, 2)
        dist_m  = dist_ft * FT_TO_M
        s_lat, s_lon = _forward_offset_local(arc_lat, arc_lon, dist_m, bearing_deg)
        sample_points.append((dist_ft, s_lat, s_lon))

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

    distances   = [sample_points[i][0] for i in range(n_samples)]
    elevations  = [results.get(i) for i in range(n_samples)]

    if any(e is None for e in elevations):
        return None, None

    return distances, elevations


def get_downstream_bearing(
    arc_lat: float,
    arc_lon: float,
    comid: Optional[str] = None,
    n_candidates: int = 8,
    probe_dist_ft: float = 150.0,
) -> float:
    """
    Determine the downstream bearing using NHD flowline geometry (primary)
    or 3DEP elevation probes constrained to ±90° of south (fallback).

    NHD flowline: the LineString coordinates run UPSTREAM → DOWNSTREAM,
    so we take the last two vertices and compute the bearing between them.
    """
    # ── Primary: NHD flowline geometry from COMID ─────────────────────────────
    if comid:
        try:
            url  = f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}"
            resp = requests.get(url, timeout=15,
                                headers={"Accept": "application/json"})
            resp.raise_for_status()
            data = resp.json()

            # Extract LineString coordinates — NHD stores upstream→downstream
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
                # Last two points = downstream end of reach
                lon1, lat1 = coords[-2][0], coords[-2][1]
                lon2, lat2 = coords[-1][0], coords[-1][1]

                # Bearing from second-to-last → last coordinate
                dlon  = math.radians(lon2 - lon1)
                lat1r = math.radians(lat1)
                lat2r = math.radians(lat2)
                x     = math.sin(dlon) * math.cos(lat2r)
                y     = math.cos(lat1r) * math.sin(lat2r) - \
                        math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
                bearing = (math.degrees(math.atan2(x, y)) + 360 + 180) % 360
                return bearing
        except Exception:
            pass

    # ── Fallback: elevation probes constrained to southward arc ──────────────
    # Constrain to 90°–270° (southward half) to avoid highway embankment
    FT_TO_M  = 0.3048
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

    if not drops:
        return 180.0

    return max(drops, key=lambda x: x[1])[0]


def build_hydro_context(lat: float, lon: float) -> HydroContext:
    notes = []

    comid, reachcode, stream_name_raw, nldi_note = get_nldi_comid(lat, lon)
    notes.append(nldi_note)

    if comid:
        stream_name = lookup_stream_name_from_comid(comid)
        if stream_name == "Unnamed stream" and stream_name_raw:
            stream_name = stream_name_raw
    else:
        stream_name = stream_name_raw or "Unnamed stream"

    drainage_area_sqmi = None
    source = "No automatic drainage area available"
    debug_nldi_tot_excerpt = "Not requested"
    debug_streamstats_excerpt = "Not requested"

    if comid:
        drainage_area_sqmi, tot_note, debug_nldi_tot_excerpt = get_drainage_area_from_nldi_tot(comid)
        notes.append(tot_note)
        if drainage_area_sqmi is not None:
            source = "USGS NLDI accumulated characteristics"

    if drainage_area_sqmi is None:
        ss_drainage_area, ss_note, debug_streamstats_excerpt = get_streamstats_drainage_area(lat, lon)
        notes.append(ss_note)
        if ss_drainage_area is not None:
            drainage_area_sqmi = ss_drainage_area
            source = "USGS StreamStats"

    # ── Auto-determine downstream bearing from NHD flowline geometry ──────────
    downstream_bearing = get_downstream_bearing(lat, lon, comid=comid)
    notes.append(f"Downstream bearing auto-detected: {downstream_bearing:.1f}°")

    # ── 3DEP elevation corridor — 13 points over 300 ft downstream ───────────
    reach_distances, reach_elevations = sample_corridor_elevations(
        arc_lat=lat,
        arc_lon=lon,
        bearing_deg=downstream_bearing,
        n_samples=13,
        total_ft=300.0,
    )

    reach_slope = None
    if reach_elevations is not None and reach_distances is not None:
        total_drop = reach_elevations[0] - reach_elevations[-1]
        reach_slope = max(0.0, total_drop / reach_distances[-1]) if reach_distances[-1] > 0 else None
        notes.append(f"3DEP elevation sampled: slope={reach_slope:.5f} ft/ft")
    else:
        notes.append("3DEP elevation sampling failed — using estimated bed profile")

    return HydroContext(
        drainage_area_sqmi=drainage_area_sqmi,
        comid=comid,
        reachcode=reachcode,
        stream_name=stream_name,
        source=source,
        notes=" | ".join(notes),
        debug_nldi_tot_excerpt=debug_nldi_tot_excerpt,
        debug_streamstats_excerpt=debug_streamstats_excerpt,
        reach_elevations=reach_elevations,
        reach_distances=reach_distances,
        reach_slope=reach_slope,
        downstream_bearing=downstream_bearing,
    )
