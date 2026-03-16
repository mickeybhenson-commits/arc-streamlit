from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


REQUEST_TIMEOUT = 25


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
        stream_name = props.get("name") or props.get("gnis_name") or "Unnamed stream"

        return comid or None, reachcode or None, stream_name or None, "NLDI position lookup succeeded"
    except Exception:
        return None, None, None, "NLDI response parse failed"


def _extract_number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def extract_drainage_area_from_payload(data: Dict[str, Any]) -> Optional[float]:
    candidate_keys = [
        "DRNAREA",
        "drnarea",
        "drainage_area",
        "AreaSqMi",
        "areasqmi",
        "DA",
        "da_sqmi",
    ]

    for key in candidate_keys:
        if key in data:
            value = _extract_number(data[key])
            if value is not None:
                return value

    features = data.get("features")
    if isinstance(features, list):
        for feat in features:
            props = feat.get("properties", {})
            for key in candidate_keys:
                if key in props:
                    value = _extract_number(props[key])
                    if value is not None:
                        return value

    for top_key in ["parameters", "parametersList", "results", "workspace", "messages"]:
        block = data.get(top_key)

        if isinstance(block, list):
            for item in block:
                if not isinstance(item, dict):
                    continue
                code = str(item.get("code") or item.get("name") or "").upper()
                if code in {"DRNAREA", "DRAINAGE_AREA", "DA"}:
                    value = _extract_number(item.get("value"))
                    if value is not None:
                        return value

        elif isinstance(block, dict):
            for key in candidate_keys:
                if key in block:
                    value = _extract_number(block[key])
                    if value is not None:
                        return value

    return None


def get_drainage_area_from_nldi_tot(comid: str) -> Tuple[Optional[float], str]:
    url = f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}/tot"
    params = {"f": "json"}

    data = safe_get_json(url, params=params)
    if not data:
        return None, "NLDI total characteristics lookup failed"

    drainage_area_sqmi = extract_drainage_area_from_payload(data)
    if drainage_area_sqmi is not None:
        return drainage_area_sqmi, "Drainage area found from NLDI accumulated characteristics"

    return None, "NLDI accumulated characteristics did not contain drainage area"


def get_streamstats_drainage_area(lat: float, lon: float) -> Tuple[Optional[float], str]:
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

    for url, params in candidate_calls:
        data = safe_get_json(url, params=params)
        if not data:
            notes.append(f"Failed: {url}")
            continue

        drainage_area_sqmi = extract_drainage_area_from_payload(data)
        if drainage_area_sqmi is not None:
            return drainage_area_sqmi, f"Drainage area found from StreamStats service: {url}"

        notes.append(f"No drainage area in response: {url}")

    return None, " | ".join(notes) if notes else "StreamStats lookup failed"


def build_hydro_context(lat: float, lon: float) -> HydroContext:
    notes = []

    comid, reachcode, stream_name, nldi_note = get_nldi_comid(lat, lon)
    notes.append(nldi_note)

    drainage_area_sqmi = None
    source = "No automatic drainage area available"

    if comid:
        drainage_area_sqmi, tot_note = get_drainage_area_from_nldi_tot(comid)
        notes.append(tot_note)
        if drainage_area_sqmi is not None:
            source = "USGS NLDI accumulated characteristics"

    if drainage_area_sqmi is None:
        ss_drainage_area, ss_note = get_streamstats_drainage_area(lat, lon)
        notes.append(ss_note)
        if ss_drainage_area is not None:
            drainage_area_sqmi = ss_drainage_area
            source = "USGS StreamStats"

    return HydroContext(
        drainage_area_sqmi=drainage_area_sqmi,
        comid=comid,
        reachcode=reachcode,
        stream_name=stream_name,
        source=source,
        notes=" | ".join(notes),
    )
