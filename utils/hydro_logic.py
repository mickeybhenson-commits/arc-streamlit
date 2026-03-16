
---

## 4) `arc-streamlit/utils/hydro_logic.py`

```python
from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd


# Western North Carolina / NC mountain regional curves
# Commonly cited form:
#   Abkf = 22.1 * DA^0.67
#   Wbkf = 19.9 * DA^0.36
#   Dbkf = 1.1  * DA^0.31
#   Qbkf = 115.7 * DA^0.73
# where DA is drainage area in square miles

REGIONAL_CURVES = {
    "Abkf": {"a": 22.1, "b": 0.67},    # ft^2
    "Wbkf": {"a": 19.9, "b": 0.36},    # ft
    "Dbkf": {"a": 1.1, "b": 0.31},     # ft
    "Qbkf": {"a": 115.7, "b": 0.73},   # cfs
}


def regional_curve(a: float, b: float, drainage_area_sqmi: float) -> float:
    return a * (drainage_area_sqmi ** b)


def compute_bankfull_metrics(drainage_area_sqmi: float) -> Dict[str, float]:
    return {
        name: regional_curve(cfg["a"], cfg["b"], drainage_area_sqmi)
        for name, cfg in REGIONAL_CURVES.items()
    }


def compute_hydrokinetic_score(depth_ft: float, dbkf_ft: float) -> Tuple[float, str]:
    """
    First-pass decision score using measured depth relative to estimated bankfull mean depth.
    """
    if dbkf_ft <= 0:
        return 0.0, "Invalid bankfull depth estimate."

    ratio = depth_ft / dbkf_ft
    score = max(0.0, 100.0 - (abs(ratio - 1.0) * 80.0))

    if ratio < 0.6:
        reason = f"Too shallow relative to estimated bankfull mean depth (depth_ratio={ratio:.2f})."
    elif ratio <= 1.4:
        reason = f"Depth is within a favorable range relative to estimated bankfull mean depth (depth_ratio={ratio:.2f})."
    else:
        reason = (
            f"Depth exceeds the bankfull mean-depth estimate; "
            f"may still be usable, but verify true velocity and position in the active channel (depth_ratio={ratio:.2f})."
        )

    return score, reason


def recommend_action(depth_ft: float, bankfull: Dict[str, float]) -> Dict[str, str]:
    dbkf = bankfull["Dbkf"]
    wbkf = bankfull["Wbkf"]
    qbkf = bankfull["Qbkf"]
    abkf = bankfull["Abkf"]

    score, reason = compute_hydrokinetic_score(depth_ft, dbkf)
    ratio = depth_ft / dbkf if dbkf > 0 else float("inf")

    if ratio < 0.6:
        action = "NAVIGATE TO DEEPER WATER / DO NOT DEPLOY YET"
        deploy = "NO"
        nav_note = (
            "Measured depth is substantially below the estimated bankfull mean depth. "
            "ARC should continue searching for a deeper portion of the main thread or thalweg."
        )
    elif ratio <= 1.4:
        action = "DEPLOY CANDIDATE"
        deploy = "YES"
        nav_note = (
            "Measured depth is in a favorable range relative to the estimated bankfull mean depth. "
            "This is a reasonable first-pass deployment candidate."
        )
    else:
        action = "POSSIBLE DEPLOYMENT / VERIFY FLOW FIRST"
        deploy = "MAYBE"
        nav_note = (
            "Depth is above the estimated bankfull mean-depth value. "
            "Verify that this is active high-velocity flow rather than pooled or slower water before deploying."
        )

    return {
        "deploy": deploy,
        "action": action,
        "score": f"{score:.1f}/100",
        "reason": reason,
        "nav_note": nav_note,
        "estimated_bankfull_width_ft": f"{wbkf:.2f}",
        "estimated_bankfull_depth_ft": f"{dbkf:.2f}",
        "estimated_bankfull_area_ft2": f"{abkf:.2f}",
        "estimated_bankfull_discharge_cfs": f"{qbkf:.2f}",
    }


def format_summary_table(drainage_area_sqmi: float, bankfull: Dict[str, float], depth_ft: float) -> pd.DataFrame:
    dbkf = bankfull["Dbkf"]
    depth_ratio = depth_ft / dbkf if dbkf > 0 else None

    data = [
        ["Drainage Area", drainage_area_sqmi, "mi²"],
        ["Measured Depth", depth_ft, "ft"],
        ["Estimated Bankfull Area", bankfull["Abkf"], "ft²"],
        ["Estimated Bankfull Width", bankfull["Wbkf"], "ft"],
        ["Estimated Bankfull Depth", bankfull["Dbkf"], "ft"],
        ["Estimated Bankfull Discharge", bankfull["Qbkf"], "cfs"],
        ["Depth / Bankfull Depth Ratio", depth_ratio, "-"],
    ]

    df = pd.DataFrame(data, columns=["Metric", "Value", "Units"])
    return df
