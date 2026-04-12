from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import requests

# --- CONSTANTS ---
REQUEST_TIMEOUT = 25
SQKM_TO_SQMI    = 0.3861021585424458
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

# [REUSE YOUR UTILITIES: safe_get_json, json_excerpt, _haversine_ft, _bearing_between, _forward_offset]

# ── NLDI Navigation & Flowline ──────────────────────────────────────────────────

# [REUSE YOUR GEOMETRY FUNCTIONS: fetch_flowline_geometry, get_downstream_bearing_from_nldi, 
#  bearing_from_flowline_with_hint]
# Note: These are critical because the Neural Logic depends on the orientation 
# of the flowline to determine 'Upstream' vs 'Downstream' candidates.

# ── Drainage Area Retrieval (The Regional Curve Key) ───────────────────────────

# [REUSE YOUR DRAINAGE AREA FUNCTIONS: get_nldi_comid, get_drainage_area_from_nldi_tot, 
#  get_streamstats_drainage_area, extract_drainage_area_from_payload]
# Note: This provides the 'A' value for your Q = a * A^b Regional Curve equations.

# ── Elevation sampling (The Neural Logic Key) ───────────────────────────────────

def get_elevation_ft(lat: float, lon: float) -> Optional[float]:
    """Retrieves 1-meter 3DEP Lidar elevation from the National Map."""
    url    = "https://epqs.nationalmap.gov/v1/json"
    params = {"x": lon, "y": lat, "wkid": 4326, "units": "Feet", "includeDate": False}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        val  = data.get("value") or data.get("elevation")
        # National Map sometimes returns -1000000 for no-data
        return float(val) if (val is not None and val > -1000) else None
    except Exception:
        return None

# [REUSE YOUR SAMPLE FUNCTION: sample_elevations_along_flowline]

# ── Integrated Hydro Context Builder ───────────────────────────────────────────

def build_hydro_context(lat: float, lon: float) -> HydroContext:
    """
    Constructs the full spatial context for the ARC Neural Logic.
    Pairs hydrological COMID data with high-res lidar slope analysis.
    """
    notes = []

    # 1. Identity: Find the COMID and Stream Name
    comid, reachcode, stream_name_raw, nldi_note = get_nldi_comid(lat, lon)
    notes.append(nldi_note)
    stream_name = lookup_stream_name_from_comid(comid) if comid else (stream_name_raw or "Unnamed")

    # 2. Drainage Area: Primary input for NC Regional Curves
    da_sqmi, source = None, "No data"
    excerpt_nldi, excerpt_ss = "", ""
    
    if comid:
        da_sqmi, tot_note, excerpt_nldi = get_drainage_area_from_nldi_tot(comid)
        if da_sqmi: 
            source = "USGS NLDI (Preferred)"
            notes.append(tot_note)

    if da_sqmi is None:
        da_sqmi, ss_note, excerpt_ss = get_streamstats_drainage_area(lat, lon)
        if da_sqmi:
            source = "USGS StreamStats"
            notes.append(ss_note)

    # 3. Geometry: Fetch flowline for spatial search
    flowline = fetch_flowline_geometry(comid) if comid else None

    # 4. Hydraulics: Determine downstream bearing (Hydrologic Routing)
    ds_bearing = 180.0
    if comid:
        nldi_b = get_downstream_bearing_from_nldi(comid, lat, lon)
        if nldi_b is not None:
            ds_bearing = nldi_b
            if flowline: # Refine with local tangent
                ds_bearing = bearing_from_flowline_with_hint(flowline, lat, lon, ds_bearing)

    # 5. Physics: Sample Lidar Slope (Neural Net Input)
    dist, elevs = sample_elevations_along_flowline(lat, lon, flowline, ds_bearing)
    slope = None
    if elevs and dist:
        drop = elevs[0] - elevs[-1]
        slope = max(0.0001, drop / dist[-1]) # Minimum slope to prevent zero-velocity errors
        notes.append(f"Lidar Slope: {slope:.5f} ft/ft")

    return HydroContext(
        drainage_area_sqmi=da_sqmi, comid=comid, reachcode=reachcode,
        stream_name=stream_name, source=source, notes=" | ".join(notes),
        debug_nldi_tot_excerpt=excerpt_nldi, debug_streamstats_excerpt=excerpt_ss,
        reach_elevations=elevs, reach_distances=dist, reach_slope=slope,
        downstream_bearing=ds_bearing, flowline_coords=flowline
    )
