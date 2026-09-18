# loop.py
import time
import math
from datetime import datetime

from core.config import load_config
from core.logger import Logger
from core.poller import poll
from core.ecowitt_parser import parse_ecowitt
from core.rain_delta import RainDeltaState, choose_cumulative, compute_rain_1m
from core.buffer import RingBuffer
from core.features import compute_derived_features
from model.engine import ModelEngine
from mqtt.publisher import MqttPublisher

# A reading is treated as stale once this many minutes have passed without a
# fresh one. Beyond this the trend features describe an outage, not the weather.
STALE_AFTER_MIN = 30
# A pressure move of this size inside the delta window is not weather; it is a
# sensor fault, and is rejected instead of being fed to the model.
DP_FAULT_HPA = 12.0
DP_FAULT_WINDOW_MIN = 10


def _to_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _parse_ts(value):
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _compute_vpd_kpa(temp_c: float, rh_pct: float) -> float:
    es = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = es * (rh_pct / 100.0)
    return max(0.0, es - ea)


def _compute_dewpoint_c(temp_c: float, rh_pct: float) -> float:
    a = 17.27
    b = 237.7
    rh = max(1e-6, min(100.0, rh_pct))
    gamma = (a * temp_c) / (b + temp_c) + math.log(rh / 100.0)
    denom = a - gamma
    if abs(denom) < 1e-9:
        return temp_c
    td = (b * gamma) / denom
    if not math.isfinite(td):
        return temp_c
    return td


def _mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else 0.0


def _max(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return max(vals) if vals else 0.0


def _is_day(row: dict) -> float:
    lux_klux = _to_float(row.get("lux_klux"), None)
    solar_wm2 = _to_float(row.get("solar_wm2"), None)
    uv = _to_float(row.get("uv_index"), 0.0) or 0.0

    if lux_klux is not None:
        return 1.0 if lux_klux > 0.1 else 0.0

    if solar_wm2 is not None:
        if uv <= 0.0 and solar_wm2 < 10.0:
            return 0.0
        return 1.0 if solar_wm2 >= 10.0 else 0.0

    h = datetime.utcnow().hour
    return 1.0 if 6 <= h <= 18 else 0.0


def _last_good(buffer, key):
    """Most recent non-missing value of `key`, or None if there is none."""
    for row in reversed(buffer.get_all()):
        v = _to_float(row.get(key), None)
        if v is not None:
            return v
    return None


def _forward_fill(value, key, buffer):
    """Hold the last known reading when a sensor reports nothing this minute.

    Substituting 0.0 for a missing reading makes the model see a simultaneous
    1013 hPa pressure collapse and a bone-dry atmosphere, which is why a single
    dropped packet used to drive PoP to 100%. Holding the previous value keeps
    every derived delta at zero, i.e. "no new information", which is the honest
    interpretation of a missing sample.
    """
    if value is not None:
        return value
    return _last_good(buffer, key)


def _reject_spike(value, key, buffer, log, max_jump, label):
    """Discard a one-minute jump too large to be weather.

    Pressure moves a few hPa per half hour and humidity tens of points at a
    gust front, so a step far larger than that inside a single sample is an
    instrument fault. Holding the previous value leaves the trend features flat
    instead of presenting the model with a step change it reads as a storm.
    """
    if value is None:
        return value
    last = buffer.last()
    # Compare against the last accepted reading, not the raw one: after a spike
    # is rejected the next (good) sample would otherwise look like a second
    # spike in the opposite direction and be rejected too, producing a false
    # jump in the trend features.
    prev = _to_float(last.get(key), None) if isinstance(last, dict) else None
    if prev is None or abs(value - prev) <= max_jump:
        return value
    log.warning(
        f"{label} jump {prev:g} -> {value:g} in one sample exceeds {max_jump:g}; "
        f"holding previous value (suspect sensor)"
    )
    return prev


def _median3(buffer, key, current):
    """Median of the two previous *raw* readings plus the current one.

    A single bad sample is the most common real-world sensor failure, and a jump
    threshold cannot catch it: 55% -> 100% -> 55% humidity is only a 45 point
    step, well inside the range a real gust front can produce, yet on its own it
    makes the trend features look like a saturated airmass and drove PoP to 97%.
    The median removes an isolated outlier of any size without a per-sensor
    constant, and lags a genuine ramp by only one sample.

    The baseline is read from the raw readings recorded alongside each row.
    Reading the filtered value back instead would make the filter recursive and
    it would latch onto the first filtered sample, freezing every later reading.
    """
    if current is None:
        return None
    prev = []
    for row in buffer.last_n(2):
        v = _to_float(row.get(f"_raw_{key}"), None)
        if v is None:
            v = _to_float(row.get(key), None)
        if v is not None:
            prev.append(v)
    if not prev:
        return current
    vals = prev + [current]
    vals.sort()
    return vals[len(vals) // 2]


def _is_stale(buffer, last_ok) -> bool:
    """True when the current row follows a polling outage.

    During an outage no rows are stored, so the first row after it sits a long
    time after its predecessor. Its trend features would compare fresh air with
    pre-outage air, which the model cannot tell apart from a sharp weather
    change, so the forecast is held until a full window of fresh rows exists.

    Note that a sensor that keeps answering with null values does not trip this:
    those rows are added continuously and are handled by the forward fill.
    """
    rows = buffer.get_all()
    if len(rows) < 2:
        return False
    prev_ts = _parse_ts(rows[-2].get("ts"))
    cur_ts = _parse_ts(rows[-1].get("ts"))
    if prev_ts is None or cur_ts is None:
        return False
    return (cur_ts - prev_ts).total_seconds() > STALE_AFTER_MIN * 60


def run():
    cfg = load_config()
    log = Logger(cfg.log_level)
    log.info("Greece Sky and Weather Nowcast engine started")
    log.info(
        f"Station: {cfg.station_name} | Ecowitt: {cfg.ecowitt_url} | "
        f"poll={cfg.poll_interval_sec}s | train={cfg.training_interval_minutes}m"
    )

    buffer = RingBuffer(max_minutes=900)
    engine = ModelEngine(logger=log)
    rain_state = RainDeltaState()

    publisher = None
    if cfg.mqtt_enabled:
        publisher = MqttPublisher(
            cfg.mqtt_host,
            cfg.mqtt_port,
            cfg.mqtt_username,
            cfg.mqtt_password,
            cfg.mqtt_topic_prefix,
            cfg.ha_discovery_prefix,
            cfg.station_name,
            log,
        )
        if publisher.connect():
            publisher.publish_discovery()
        else:
            publisher = None

    last_training = time.time()
    backoff_s = 0
    ticks = 0
    last_ok = None
    last_daily_save_day = None
    last_forecast = None

    last_debug_emit = 0.0
    debug_features_payload = {}

    while True:
        t0 = time.time()

        if backoff_s > 0:
            log.warning(f"Backoff active: sleeping {backoff_s}s (Ecowitt unreachable)")
            time.sleep(backoff_s)

        raw = poll(cfg.ecowitt_url, timeout_s=3)
        if raw is None:
            backoff_s = min(900, 60 if backoff_s == 0 else backoff_s * 2)
            continue

        backoff_s = 0
        last_ok = datetime.utcnow().isoformat()

        parsed = parse_ecowitt(raw)

        # rain_1m
        cum, src = choose_cumulative(parsed)
        rain_state.source = src
        rain_1m = compute_rain_1m(cum, rain_state)

        # Build row
        row = dict(parsed)
        row["ts"] = last_ok
        row["rain_source"] = src
        row["rain_1m_mm"] = round(rain_1m, 4)

        # base features
        temp_c = _to_float(parsed.get("t_out_c"))
        rh_pct = _to_float(parsed.get("rh_out_pct"))
        p_rel = _to_float(parsed.get("p_rel_hpa"))
        p_abs = _to_float(parsed.get("p_abs_hpa"))
        wspd = _to_float(parsed.get("wind_avg_ms"), 0.0) or 0.0
        wgst = _to_float(parsed.get("wind_gust_ms"), 0.0) or 0.0

        solar_wm2 = _to_float(parsed.get("solar_wm2"), None)
        lux_klux = _to_float(parsed.get("lux_klux"), None)

        # Keep the unfiltered readings so the outlier filters can compare against
        # what the sensor actually reported rather than their own output.
        raw_temp, raw_rh = temp_c, rh_pct
        raw_p_rel, raw_p_abs = p_rel, p_abs

        # Hold the last known reading when a sensor drops out, reject
        # single-sample jumps that are physically impossible, and take the
        # median of three so an isolated outlier cannot reach the trend
        # features, before any of these values are used.
        temp_c = _median3(
            buffer, "temperature",
            _reject_spike(_forward_fill(temp_c, "temperature", buffer),
                          "temperature", buffer, log, 25.0, "Outdoor temperature"))
        rh_pct = _median3(
            buffer, "humidity",
            _reject_spike(_forward_fill(rh_pct, "humidity", buffer),
                          "humidity", buffer, log, 70.0, "Outdoor humidity"))
        if rh_pct is not None:
            rh_pct = max(0.0, min(100.0, rh_pct))
        p_rel = _median3(
            buffer, "pressure",
            _reject_spike(_forward_fill(p_rel, "pressure", buffer),
                          "pressure", buffer, log, DP_FAULT_HPA, "Relative pressure"))
        p_abs = _median3(
            buffer, "absolutepressure",
            _reject_spike(_forward_fill(p_abs, "absolutepressure", buffer),
                          "absolutepressure", buffer, log, DP_FAULT_HPA,
                          "Absolute pressure"))

        if solar_wm2 is None and lux_klux is None:
            solar_val = _last_good(buffer, "solarradiation")
            solar_val = 0.0 if solar_val is None else solar_val
        else:
            solar_val = float(solar_wm2) if solar_wm2 is not None else float(lux_klux)
        row["solarradiation"] = solar_val

        row["temperature"] = temp_c
        row["humidity"] = rh_pct
        row["pressure"] = p_rel
        row["absolutepressure"] = p_abs
        row["_raw_temperature"] = raw_temp if raw_temp is not None else temp_c
        row["_raw_humidity"] = raw_rh if raw_rh is not None else rh_pct
        row["_raw_pressure"] = raw_p_rel if raw_p_rel is not None else p_rel
        row["_raw_absolutepressure"] = (
            raw_p_abs if raw_p_abs is not None else p_abs)
        row["windspeed"] = wspd
        row["windgust"] = wgst

        row["rainrate"] = _to_float(parsed.get("rain_rate_mmh"), 0.0) or 0.0
        row["hourlyrain"] = _to_float(parsed.get("rain_hourly_mm"), 0.0) or 0.0

        wd = _to_float(parsed.get("wind_dir_deg"))
        if wd is None:
            row["winddir_sin"] = 0.0
            row["winddir_cos"] = 0.0
        else:
            r = math.radians(float(wd) % 360.0)
            row["winddir_sin"] = math.sin(r)
            row["winddir_cos"] = math.cos(r)

        vpd = _to_float(parsed.get("vpd_kpa"))
        if vpd is None and temp_c is not None and rh_pct is not None:
            try:
                vpd = _compute_vpd_kpa(float(temp_c), float(rh_pct))
            except Exception:
                vpd = 0.0
        row["vpd"] = float(vpd) if vpd is not None else 0.0

        # rich features
        prev10 = buffer.get_minutes_ago(10, now_ts=last_ok)
        prev30 = buffer.get_minutes_ago(30, now_ts=last_ok)
        stale = _is_stale(buffer, last_ok)

        def d(key, prev):
            cur = _to_float(row.get(key), None)
            prv = _to_float(prev.get(key), None) if isinstance(prev, dict) else None
            if cur is None or prv is None:
                return 0.0
            out = cur - prv
            return float(out) if math.isfinite(out) else 0.0

        row["dP_10m"] = d("pressure", prev10)
        row["dP_30m"] = d("pressure", prev30)
        row["dT_10m"] = d("temperature", prev10)
        row["dT_30m"] = d("temperature", prev30)
        row["dRH_10m"] = d("humidity", prev10)
        row["dRH_30m"] = d("humidity", prev30)

        row["dVPD_10m"] = d("vpd", prev10)
        row["dWind_10m"] = d("windspeed", prev10)
        row["dGust_10m"] = d("windgust", prev10)
        row["dSolar_10m"] = d("solarradiation", prev10)

        row["winddir_sin_delta_10m"] = d("winddir_sin", prev10)
        row["winddir_cos_delta_10m"] = d("winddir_cos", prev10)

        if temp_c is not None and rh_pct is not None:
            try:
                td = _compute_dewpoint_c(float(temp_c), float(rh_pct))
            except Exception:
                td = float(temp_c)
        else:
            td = 0.0

        row["dewpoint_c"] = float(td) if math.isfinite(float(td)) else 0.0
        row["spread_td"] = (float(temp_c) - float(row["dewpoint_c"])) if temp_c is not None else 0.0

        last10 = buffer.last_n(9) + [row]
        last30 = buffer.last_n(29) + [row]

        row["temp_mean_30m"] = _mean([_to_float(r.get("temperature"), None) for r in last30])
        row["rh_mean_30m"] = _mean([_to_float(r.get("humidity"), None) for r in last30])
        row["p_mean_30m"] = _mean([_to_float(r.get("pressure"), None) for r in last30])

        row["wind_mean_10m"] = _mean([_to_float(r.get("windspeed"), None) for r in last10])
        row["gust_max_10m"] = _max([_to_float(r.get("windgust"), None) for r in last10])
        row["solar_mean_10m"] = _mean([_to_float(r.get("solarradiation"), None) for r in last10])
        row["rainrate_max_10m"] = _max([_to_float(r.get("rainrate"), None) for r in last10])

        now_utc = datetime.utcnow()
        hour = now_utc.hour + now_utc.minute / 60.0
        row["hour_sin"] = math.sin(2.0 * math.pi * (hour / 24.0))
        row["hour_cos"] = math.cos(2.0 * math.pi * (hour / 24.0))

        doy = now_utc.timetuple().tm_yday
        row["doy_sin"] = math.sin(2.0 * math.pi * (doy / 365.25))
        row["doy_cos"] = math.cos(2.0 * math.pi * (doy / 365.25))

        row["is_day"] = _is_day(row)

        buffer.add(row)

        # === ENHANCED FEATURES INTEGRATION ===
        # The ring buffer already contains this minute's row, so the enhanced
        # features can be computed from it directly and stored back in place.
        buffer_recent = buffer.get_all()
        buffer_30m = buffer.last_n(30)
        enhanced = compute_derived_features(row, buffer_recent, buffer_30m, now_utc)
        for fname, fval in enhanced.items():
            row[fname] = fval
        # === END ENHANCED FEATURES ===

        forecasts = engine.infer(buffer)

        # After an outage the features compare fresh air with pre-outage air,
        # which the model reads as a sharp weather change. Hold the previous
        # forecast until a full lookback window of fresh rows exists again,
        # rather than publishing a reading the data cannot support.
        if stale:
            if last_forecast is None:
                last_forecast = forecasts
            forecasts = dict(last_forecast)
            forecasts["stale"] = 1
            log.warning(
                "No fresh sensor data for over "
                f"{STALE_AFTER_MIN}m; holding previous forecast"
            )
        else:
            last_forecast = forecasts
            forecasts["stale"] = 0

        # manual save button
        if publisher and publisher.consume_save_requested():
            try:
                engine.save_state()
                log.info("Manual save requested: station model state saved")
            except Exception as e:
                log.warning(f"Manual save failed: {e}")

        # Debug once/hour
        if (time.time() - last_debug_emit) >= 3600:
            debug_features_payload = {
                "ts": last_ok,
                "dP_10m": round(_to_float(row.get("dP_10m"), 0.0) or 0.0, 3),
                "dP_30m": round(_to_float(row.get("dP_30m"), 0.0) or 0.0, 3),
                "dT_10m": round(_to_float(row.get("dT_10m"), 0.0) or 0.0, 3),
                "dRH_10m": round(_to_float(row.get("dRH_10m"), 0.0) or 0.0, 3),
                "spread_td": round(_to_float(row.get("spread_td"), 0.0) or 0.0, 3),
                "p_mean_30m": round(_to_float(row.get("p_mean_30m"), 0.0) or 0.0, 3),
                "solar_mean_10m": round(_to_float(row.get("solar_mean_10m"), 0.0) or 0.0, 3),
                "is_day": float(_to_float(row.get("is_day"), 0.0) or 0.0),
                "dP_30m_scaled": round(enhanced.get('dP_30m_scaled', 0), 3),
                "dRH_30m_scaled": round(enhanced.get('dRH_30m_scaled', 0), 3),
                "dSolar_10m_scaled": round(enhanced.get('dSolar_10m_scaled', 0), 3),
                "spread_td_enh": round(enhanced.get('spread_td', 0), 3),
                "pre_rain_index": round(enhanced.get('pre_rain_index', 0), 3),
                "instability_index": round(enhanced.get('instability_index', 0), 3),
                "pop_30m": forecasts.get("pop_30m"),
                "pop_60m": forecasts.get("pop_60m"),
                "pop_120m": forecasts.get("pop_120m"),
                "p50_60m": forecasts.get("p50_60m"),
            }
            log.info(f"Debug features: {debug_features_payload}")
            last_debug_emit = time.time()

        # Train hourly
        last_training_result = "skipped"
        if time.time() - last_training >= cfg.training_interval_minutes * 60:
            if cfg.learning_enabled:
                info = engine.train_from_buffer(
                    buffer.get_all(),
                    horizons_min=(30, 60, 120, 360),
                    threshold_mm=0.1,
                )
                last_training_result = f"updated={info.get('updated', 0)}"
                log.info(f"Training: {info}")
            else:
                log.info("Training skipped (learning_enabled=false)")
                last_training_result = "disabled"
            last_training = time.time()

        # daily save
        if cfg.persist_model and now_utc.hour == int(cfg.model_save_hour):
            if last_daily_save_day != now_utc.date():
                try:
                    engine.save_state()
                    last_daily_save_day = now_utc.date()
                    log.info("Saved station model state (daily)")
                except Exception as e:
                    log.warning(f"Daily save failed: {e}")

        if publisher:
            publisher.publish_state(forecasts)
            publisher.publish_status(
                {
                    "state": "running",
                    "station": cfg.station_name,
                    "ticks": ticks,
                    "last_ok_poll": last_ok,
                    "rain_source": src,
                    "base_model_version": engine.base_version,
                    "trained_samples_total": engine.trained_samples,
                    "last_training_result": last_training_result,
                    "last_training_info": engine.last_training_info,
                    "debug_features": debug_features_payload,
                }
            )

        log.info(
            f"tick={ticks} rain_1m={rain_1m:.3f}mm src={src} pop60={forecasts.get('pop_60m')}%"
        )
        ticks += 1

        elapsed = time.time() - t0
        time.sleep(max(0.0, cfg.poll_interval_sec - elapsed))