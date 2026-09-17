from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from core.units import parse_float

def _list_to_map(lst: Any, key_field: str = "id") -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(lst, list):
        return out
    for item in lst:
        if not isinstance(item, dict):
            continue
        k = item.get(key_field)
        if k is None:
            continue
        out[str(k)] = item
    return out

def parse_ecowitt(raw: Dict[str, Any]) -> Dict[str, Any]:
    # raw has keys: common_list, rain, wh25, debug
    common = _list_to_map(raw.get("common_list", []), "id")
    rain = _list_to_map(raw.get("rain", []), "id")

    # Typical IDs (may vary by model)
    t_out = parse_float(common.get("0x02", {}).get("val"))  # outdoor temp C
    rh_out = parse_float(common.get("0x07", {}).get("val"))  # outdoor RH %
    wind_avg = parse_float(common.get("0x0B", {}).get("val"))  # m/s
    wind_gust = parse_float(common.get("0x0C", {}).get("val"))  # m/s
    wind_max = parse_float(common.get("0x19", {}).get("val"))  # m/s
    lux = parse_float(common.get("0x15", {}).get("val"))  # Klux
    uv = parse_float(common.get("0x17", {}).get("val"))  # UV index
    solar_wm2 = parse_float(common.get("0x6D", {}).get("val"))  # W/m2
    wind_dir = parse_float(common.get("0x0A", {}).get("val"))  # degrees

    # Rain
    rain_daily_mm = parse_float(rain.get("0x0D", {}).get("val"))  # daily mm (cumulative)
    rain_rate_mmh = parse_float(rain.get("0x0E", {}).get("val"))  # mm/hr
    rain_event_mm = parse_float(rain.get("0x7C", {}).get("val"))  # event mm (cumulative)
    rain_hourly_mm = parse_float(rain.get("0x10", {}).get("val"))  # hourly mm (last hour)

    # Indoor/pressure from wh25
    wh25 = raw.get("wh25")
    p_rel = None
    p_abs = None
    t_in = None
    rh_in = None
    if isinstance(wh25, list) and wh25 and isinstance(wh25[0], dict):
        t_in = parse_float(wh25[0].get("intemp"))
        rh_in = parse_float(wh25[0].get("inhumi"))
        p_abs = parse_float(wh25[0].get("abs"))
        p_rel = parse_float(wh25[0].get("rel"))

    return {
        "t_out_c": t_out,
        "rh_out_pct": rh_out,
        "wind_avg_ms": wind_avg,
        "wind_gust_ms": wind_gust,
        "wind_max_ms": wind_max,
        "wind_dir_deg": wind_dir,
        "lux_klux": lux,
        "uv_index": uv,
        "solar_wm2": solar_wm2,
        "t_in_c": t_in,
        "rh_in_pct": rh_in,
        "p_abs_hpa": p_abs,
        "p_rel_hpa": p_rel,
        "rain_daily_mm": rain_daily_mm,
        "rain_event_mm": rain_event_mm,
        "rain_hourly_mm": rain_hourly_mm,
        "rain_rate_mmh": rain_rate_mmh,
    }
