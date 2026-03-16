from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


REQUEST_TIMEOUT = 20


@dataclass
class HydroContext:
    drainage_area_sqmi: Optional[float]
    comid: Optional[str]
    reachcode: Optional[str]
    stream_name: Optional[str]
    source: str
    notes: str


def safe_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def get_nldi_comid(lat: float, lon: float) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Attempts to snap a coordinate to the hydrologic network using an NLDI-style endpoint.
    If the service changes or fails, the rest of the app still works.
    """
    url = "https://labs.waterdata.usgs.gov/api/nldi/linked-data/comid/position"
    params = {"coords": f"POINT({lon} {lat})"}

    data = safe_get_json(url, params=params)
    if not data:
        return None, None, None

    try:
        features = data.get("features", [])
        if not features:
            return None, None, None

        props = features[0].get("properties", {})
        comid = str(props.get("identifier") or props.get("comid") or "")
        reachcode = str(props.get("reachcode") or "")
        stream_name = props.get("name") or props.get("gnis_name") or "Unnamed stream"
        return comid or None, reachcode or None, stream_name or None
    except Exception:
        return None, None, None


def extract_drainage_area_from_streamstats_payload(data: Dict[str, Any]) -> Optional[float]:
    common_keys = ["drainage_area", "DRNAREA", "AreaSqMi", "areasqmi", "DA", "da_sqmi"]

    for key in common_keys:
        if key in data:
            try:
                return float(data[key])
            except Exception:
                pass

    for top_key in ["parameters", "parametersList", "results", "workspace"]:
        block = data.get(top_key)

        if isinstance(block, list):
            for item in block:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or item.get("code") or "").upper()
                if name in {"DRNAREA", "DRAINAGE_AREA", "DA"}:
                    try:
                        return float(item.get("value"))
                    except Exception:
                        pass

        elif isinstance(block, dict):
            for key in common_keys:
                if key in block:
                    try:
                        return float(block[key])
                    except Exception:
                        pass

    features = data.get("features")
    if isinstance(features, list):
        for feat in features:
            props = feat.get("properties", {})
            for key in common_keys:
                if key in props:
                    try:
                        return float(props[key])
                    except Exception:
                        pass

    return None


def get_streamstats_drainage_area(lat: float, lon: float) -> Optional[float]:
    """
    Defensive attempt to retrieve drainage area from StreamStats-style services.
    """
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

    for url, params in candidate_calls:
        data = safe_get_json(url, params=params)
        if not data:
            continue

        drainage_area_sqmi = extract_drainage_area_from_streamstats_payload(data)
        if drainage_area_sqmi is not None:
            return drainage_area_sqmi

    return None


def build_hydro_context(lat: float, lon: float) -> HydroContext:
    comid, reachcode, stream_name = get_nldi_comid(lat, lon)
    drainage_area_sqmi = get_streamstats_drainage_area(lat, lon)

    notes = []
    if comid:
        notes.append(f"NLDI/flowline match found (COMID={comid})")
    else:
        notes.append("No NLDI COMID found automatically")

    if drainage_area_sqmi is not None:
        notes.append(f"USGS drainage area found: {drainage_area_sqmi:.3f} mi^2")
        source = "USGS online services"
    else:
        notes.append("USGS drainage area lookup failed; manual drainage area entry required")
        source = "manual fallback needed"

    return HydroContext(
        drainage_area_sqmi=drainage_area_sqmi,
        comid=comid,
        reachcode=reachcode,
        stream_name=stream_name,
        source=source,
        notes=" | ".join(notes),
    )
