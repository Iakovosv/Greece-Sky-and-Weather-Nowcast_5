from __future__ import annotations
from typing import Any, Dict, List, Tuple

def sum_rain_last_minutes(buffer_rows: List[Dict[str, Any]], minutes: int) -> float:
    # expects rows are 1-minute cadence
    if minutes <= 0 or len(buffer_rows) < minutes+1:
        return 0.0
    tail = buffer_rows[-minutes:]
    return float(sum(float(r.get("rain_1m_mm", 0.0) or 0.0) for r in tail))

def snapshot_for_minutes_ago(buffer_rows: List[Dict[str, Any]], minutes: int) -> Dict[str, Any] | None:
    if len(buffer_rows) < minutes+1:
        return None
    return buffer_rows[-(minutes+1)]
