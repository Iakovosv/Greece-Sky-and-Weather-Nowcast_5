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


def _to_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _compute_vpd_kpa(temp_c: float, rh_pct: float) -> float:
    es = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = es * (rh_pct / 100.0)
    return max(0.0, es - ea)


def _compute_dewpoint_c(temp_c: float, rh_pct: float) -> float:
    a = 17.27
    b = 237.7
    rh = max(1e-6, min(100.0, rh_pct))
    gamma = (a * temp_c) / (b + temp_c) + math.log(rh / 100.0)
    td = (b * gamma) / (a - gamma)
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

        solar_val = 0.0
        if solar_wm2 is not None:
            solar_val = float(solar_wm2)
        elif lux_klux is not None:
            solar_val = float(lux_klux)
        row["solarradiation"] = solar_val

        row["temperature"] = temp_c
        row["humidity"] = rh_pct
        row["pressure"] = p_rel
        row["absolutepressure"] = p_abs
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
        prev10 = buffer.get_minutes_ago(10)
        prev30 = buffer.get_minutes_ago(30)

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