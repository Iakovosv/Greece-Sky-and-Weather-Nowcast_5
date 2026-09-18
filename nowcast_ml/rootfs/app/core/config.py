import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict

# Home Assistant writes the add-on options here. The path can be overridden so
# the app also runs outside the add-on container (e.g. for tests), and the
# environment variables below let any single option be set without editing the
# add-on configuration.
OPTIONS_PATH = Path(os.environ.get("NOWCAST_OPTIONS_PATH", "/data/options.json"))
ENV_PREFIX = "NOWCAST_"


@dataclass
class Config:
    station_name: str = "Station"
    ecowitt_url: str = "http://127.0.0.1/get_livedata_info"
    poll_interval_sec: int = 60
    training_interval_minutes: int = 60
    learning_enabled: bool = True
    persist_model: bool = True
    model_save_hour: int = 3
    log_level: str = "info"
    mqtt_enabled: bool = True
    mqtt_host: str = "core-mosquitto"
    mqtt_port: int = 1883
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_topic_prefix: str = "greece_sky_and_weather_nowcast"
    ha_discovery_prefix: str = "homeassistant"


_BOOL_KEYS = {f.name for f in fields(Config) if f.type is bool}
_INT_KEYS = {
    f.name for f in fields(Config)
    if f.name in ("poll_interval_sec", "training_interval_minutes", "model_save_hour",
                  "mqtt_port")
}


def _coerce(key: str, value: Any) -> Any:
    """Coerce a string from the environment into the field's type."""
    if isinstance(value, str) and key in _BOOL_KEYS:
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(value, str) and key in _INT_KEYS:
        try:
            return int(value)
        except ValueError:
            return value
    return value


def load_config() -> Config:
    data: Dict[str, Any] = {}
    if OPTIONS_PATH.exists():
        try:
            data = json.loads(OPTIONS_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    valid = {f.name for f in fields(Config)}
    data = {k: v for k, v in data.items() if k in valid}

    for key in valid:
        env = os.environ.get(f"{ENV_PREFIX}{key.upper()}")
        if env is not None:
            data[key] = _coerce(key, env)

    return Config(**data)
