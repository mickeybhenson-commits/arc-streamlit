from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import json
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
        stream_name = props.get("name") or props.get("gnis_name") or "Unnamed stream"

        return comid or None, reachcode or None, stream_name or None, "NLDI position lookup succeeded"
    except Exception:
        return None, None, None, "NLDI response parse failed"


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

    if key_lower in {
        "drnarea",
        "drainage_area",
        "areasqmi",
        "da",
        "da_sqmi",
        "totdasqmi",
        "tot_drainage_area_sqmi",
    }:
        return number

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
    ]

    for key in direct_keys:
        if key in data:
            converted = _convert_key_value_to_sqmi(key, data[key])
            if converted is not None:
                return converted

    features = data.get("features")
    if isinstance(features, list):
        for feat in features:
            props = feat.get("properties", {})
            for key in direct_keys:
                if key in props:
                    converted = _convert_key_value_to_sqmi(key, props[key])
                    if converted is not None:
                        return converted

            prop_name = str(props.get("name") or props.get("characteristic_id") or props.get("id") or "").lower()
            prop_value = props.get("value")
            converted = _convert_key_value_to_sqmi(prop_name, prop_value)
            if converted is not None:
                return converted

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

                converted = _convert_key_value_to_sqmi(code, item.get("value"))
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
        return drainage_area_sqmi, f"Drainage area found from NLDI accumulated characteristics: {drainage_area_sqmi:.3f} mi²", excerpt

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
            return drainage_area_sqmi, f"Drainage area found from StreamStats: {drainage_area_sqmi:.3f} mi²", "\n\n".join(excerpts)

        notes.append(f"No recognized drainage-area field in response: {url}")

    return None, " | ".join(notes) if notes else "StreamStats lookup failed", "\n\n".join(excerpts)


def build_hydro_context(lat: float, lon: float) -> HydroContext:
    notes = []

    comid, reachcode, stream_name, nldi_note = get_nldi_comid(lat, lon)
    notes.append(nldi_note)

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

    return HydroContext(
        drainage_area_sqmi=drainage_area_sqmi,
        comid=comid,
        reachcode=reachcode,
        stream_name=stream_name,
        source=source,
        notes=" | ".join(notes),
        debug_nldi_tot_excerpt=debug_nldi_tot_excerpt,
        debug_streamstats_excerpt=debug_streamstats_excerpt,
    )
