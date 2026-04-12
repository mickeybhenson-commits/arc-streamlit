from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import requests

# --- CONSTANTS ---
REQUEST_TIMEOUT = 25
SQKM_TO_SQMI    = 0.386102
FT_TO_M         = 0.3048

@dataclass
class HydroContext:
    drainage_area_sqmi:        Optional[float]
    comid:                     Optional[str]
    reachcode:                 Optional[str]
    stream_name:               Optional[str]
    source:                    str
    notes:                     str
    debug_nldi_tot_excerpt:    str
    debug_streamstats_excerpt: str
    reach_elevations:          Optional[List[float]] = None
    reach_distances:           Optional[List[float]] = None
    reach_slope:               Optional[float]       = None
    downstream_bearing:        float                 = 155.0
    flowline_coords:           Optional[List[Tuple[float, float]]] = None

# ── API Utilities ─────────────────────────────────────────────────────────────

def safe_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT, headers={"Accept": "application/json"})
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def json_excerpt(data: Optional[Dict[str, Any]], max_chars: int = 1500) -> str:
    if data is None: return "None"
    text = json.dumps(data, indent=2)
    return text[:max_chars] + "\n..." if len(text) > max_chars else text

# ── Spatial Helpers ───────────────────────────────────────────────────────────

def _haversine_ft(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a)) / FT_TO_M

def _bearing_between(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r)*math.sin(lat2r) - math.sin(lat1r)*math.cos(lat2r)*math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

# ── NLDI/StreamStats Core ─────────────────────────────────────────────────────

def get_nldi_comid(lat: float, lon: float) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    url = "https://api.water.usgs.gov/nldi/linked-data/comid/position"
    params = {"coords": f"POINT({lon} {lat})", "f": "json"}
    data = safe_get_json(url, params=params)
    if not data or not data.get("features"): return None, None, None, "NLDI lookup failed"
    props = data["features"][0].get("properties", {})
    return str(props.get("identifier")), str(props.get("reachcode")), props.get("name"), "NLDI Success"

def get_drainage_area_from_nldi_tot(comid: str) -> Tuple[Optional[float], str, str]:
    url = f"https://api.water.usgs.gov/nldi/linked-data/comid/{comid}/tot"
    data = safe_get_json(url, params={"f": "json"})
    if not data: return None, "NLDI DA failed", "None"
    # Search for characteristic ID 'TOT_BASIN_AREA' or similar
    for feat in data.get("features", []):
        p = feat.get("properties", {})
        if "tot_drainage_area_sqkm" in str(p.get("characteristic_id", "")).lower():
            val = float(p.get("characteristic_value", 0)) * SQKM_TO_SQMI
            return val, f"DA: {val:.2f} mi²", json_excerpt(data)
    return None, "DA Field not found", json_excerpt(data)

# ── Elevation/Lidar Engine (3DEP) ─────────────────────────────────────────────

def get_elevation_ft(lat: float, lon: float) -> Optional[float]:
    url = "https://epqs.nationalmap.gov/v1/json"
    params = {"x": lon, "y": lat, "wkid": 4326, "units": "Feet"}
    data = safe_get_json(url, params=params)
    if data:
        val = data.get("value") or data.get("elevation")
        return float(val) if val and float(val) > -1000 else None
    return None

def sample_elevations_along_flowline(lat, lon, bearing, n=13, dist_ft=300.0):
    """Samples Lidar along the reach to calculate the energy gradient (slope)."""
    step = dist_ft / (n - 1)
    dists, elevs = [], []
    
    def _fetch(i):
        d = i * step
        # Simple projection for demo; production uses NHDPlus geometry vertices
        rad_b = math.radians(bearing)
        R = 6371000.0
        d_m = d * FT_TO_M
        l1, r1 = math.radians(lat), math.radians(lon)
        l2 = math.asin(math.sin(l1)*math.cos(d_m/R) + math.cos(l1)*math.sin(d_m/R)*math.cos(rad_b))
        r2 = r1 + math.atan2(math.sin(rad_b)*math.sin(d_m/R)*math.cos(l1), math.cos(d_m/R)-math.sin(l1)*math.sin(l2))
        return d, get_elevation_ft(math.degrees(l2), math.degrees(r2))

    with ThreadPoolExecutor(max_workers=n) as exec:
        futures = [exec.submit(_fetch, i) for i in range(n)]
        for f in as_completed(futures):
            d, e = f.result()
            if e: dists.append(d); elevs.append(e)
            
    # Sort by distance
    res = sorted(zip(dists, elevs))
    return [x[0] for x in res], [x[1] for x in res]

# ── Context Builder ───────────────────────────────────────────────────────────

def build_hydro_context(lat: float, lon: float) -> HydroContext:
    comid, reach, name, note = get_nldi_comid(lat, lon)
    da, da_note, excerpt = (None, "N/A", "")
    if comid:
        da, da_note, excerpt = get_drainage_area_from_nldi_tot(comid)
    
    # Primary inputs for Neural Logic: Slope and DA
    dists, elevs = sample_elevations_along_flowline(lat, lon, 155.0) # Default bearing
    slope = (elevs[0] - elevs[-1]) / dists[-1] if len(elevs) > 1 else 0.001
    
    return HydroContext(
        drainage_area_sqmi=da, comid=comid, reachcode=reach,
        stream_name=name or "Cullowhee Creek", source="USGS NLDI/3DEP",
        notes=f"{note} | {da_note}", debug_nldi_tot_excerpt=excerpt,
        debug_streamstats_excerpt="N/A", reach_elevations=elevs,
        reach_distances=dists, reach_slope=max(0.0001, slope),
        downstream_bearing=155.0, flowline_coords=None
    )
