"""Enhanced feature engineering for pre-rain detection.

Every feature here is a *leading* indicator: it must be computable from the
buffer strictly before rain starts. The model reads these alongside the
instantaneous base features.

All lookbacks are taken from the ring buffer by minute offset, not by raw
index, so the window names match the data they actually contain.
"""
import math
from typing import Any, Dict, List

# Typical magnitudes used to bring deltas onto a comparable scale.
P_TYP = 3.0    # hPa
H_TYP = 10.0   # %
T_TYP = 2.0    # C
W_TYP = 5.0    # m/s
S_TYP = 200.0  # W/m2


def _f(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = d.get(key)
        if v is None:
            return default
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _row_at(buffer_recent: List[Dict[str, Any]], minutes_ago: int) -> Dict[str, Any] | None:
    """Row `minutes_ago` minutes before the end of buffer_recent (1 row/minute).

    buffer_recent[-1] is the current minute, so minutes_ago=1 is the previous
    row and minutes_ago=10 is ten rows back.
    """
    if minutes_ago <= 0 or len(buffer_recent) < minutes_ago + 1:
        return None
    return buffer_recent[-(minutes_ago + 1)]


def _delta(current: Dict[str, Any], buffer_recent: List[Dict[str, Any]],
           key: str, minutes_ago: int) -> float:
    prev = _row_at(buffer_recent, minutes_ago)
    if prev is None:
        return 0.0
    return _f(current, key) - _f(prev, key)


def compute_derived_features(
    current: Dict[str, Any],
    buffer_recent: List[Dict[str, Any]],
    buffer_30m: List[Dict[str, Any]],
    now_utc: Any,
) -> Dict[str, float]:
    features: Dict[str, float] = {}

    # BASE FEATURES
    features['temperature'] = _f(current, 'temperature')
    features['humidity'] = _f(current, 'humidity')
    features['pressure'] = _f(current, 'pressure')
    features['windspeed'] = _f(current, 'windspeed')
    features['windgust'] = _f(current, 'windgust')
    features['solarradiation'] = _f(current, 'solarradiation')
    features['rainrate'] = _f(current, 'rainrate')
    features['rain_1m_mm'] = _f(current, 'rain_1m_mm')

    # 10-minute deltas (scaled) -- genuinely 10 minutes back
    features['dP_10m_scaled'] = _delta(current, buffer_recent, 'pressure', 10) / P_TYP
    features['dRH_10m_scaled'] = _delta(current, buffer_recent, 'humidity', 10) / H_TYP
    features['dT_10m_scaled'] = _delta(current, buffer_recent, 'temperature', 10) / T_TYP
    features['dWind_10m_scaled'] = _delta(current, buffer_recent, 'windspeed', 10) / W_TYP
    features['dSolar_10m_scaled'] = _delta(current, buffer_recent, 'solarradiation', 10) / S_TYP

    # 30-minute deltas (scaled) -- genuinely 30 minutes back
    features['dP_30m_scaled'] = _delta(current, buffer_recent, 'pressure', 30) / (P_TYP * 2)
    features['dRH_30m_scaled'] = _delta(current, buffer_recent, 'humidity', 30) / (H_TYP * 2)
    features['dT_30m_scaled'] = _delta(current, buffer_recent, 'temperature', 30) / (T_TYP * 2)

    # Rolling statistics over the trailing 30 minutes
    if buffer_30m:
        pressures = [_f(r, 'pressure') for r in buffer_30m]
        humidities = [_f(r, 'humidity') for r in buffer_30m]
        n = len(pressures)

        features['p_mean_30m_scaled'] = (sum(pressures) / n - 1013.0) / P_TYP
        features['rh_mean_30m_scaled'] = (sum(humidities) / n - 60.0) / H_TYP

        if n >= 5:
            features['p_trend_30m'] = (pressures[-1] - pressures[0]) / n * 10 / P_TYP
            features['h_trend_30m'] = (humidities[-1] - humidities[0]) / n * 10 / H_TYP
            recent_5 = pressures[-5:]
            features['p_volatility_5m'] = (max(recent_5) - min(recent_5)) / 2 / P_TYP
        else:
            features['p_trend_30m'] = 0.0
            features['h_trend_30m'] = 0.0
            features['p_volatility_5m'] = 0.0
    else:
        features['p_mean_30m_scaled'] = 0.0
        features['rh_mean_30m_scaled'] = 0.0
        features['p_trend_30m'] = 0.0
        features['h_trend_30m'] = 0.0
        features['p_volatility_5m'] = 0.0

    # Dew point spread: small spread means saturated air
    temp = features['temperature']
    rh = features['humidity']
    if rh > 0.0:
        gamma = (17.27 * temp) / (237.7 + temp) + math.log(max(1e-6, min(100.0, rh)) / 100.0)
        td = (237.7 * gamma) / (17.27 - gamma) if abs(17.27 - gamma) > 1e-9 else temp
        td = td if math.isfinite(td) else temp
    else:
        td = temp
    features['spread_td'] = temp - td
    features['dewpoint_c'] = td
    features['dSpread_10m'] = _delta(current, buffer_recent, 'spread_td', 10)

    # Composite indicators: rising humidity + falling pressure + falling solar
    features['pre_rain_index'] = (
        -features.get('dP_30m_scaled', 0.0) * 0.4
        + features.get('dRH_30m_scaled', 0.0) * 0.4
        - features.get('dSolar_10m_scaled', 0.0) * 0.2
    )
    features['instability_index'] = (
        features.get('rh_mean_30m_scaled', 0.0) * 0.3
        - features.get('dP_10m_scaled', 0.0) * 0.4
        + features.get('dWind_10m_scaled', 0.0) * 0.3
    )

    # Cyclical time encodings
    hour = now_utc.hour + now_utc.minute / 60.0
    features['hour_sin'] = math.sin(2.0 * math.pi * (hour / 24.0))
    features['hour_cos'] = math.cos(2.0 * math.pi * (hour / 24.0))
    doy = now_utc.timetuple().tm_yday
    features['doy_sin'] = math.sin(2.0 * math.pi * (doy / 365.25))
    features['doy_cos'] = math.cos(2.0 * math.pi * (doy / 365.25))

    # Day/Night: solar radiation is the most reliable discriminator here
    solar = _f(current, 'solarradiation', -1.0)
    features['is_day'] = 1.0 if solar >= 10.0 else 0.0

    return features