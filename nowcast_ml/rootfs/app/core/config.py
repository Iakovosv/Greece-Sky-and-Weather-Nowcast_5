import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

OPTIONS_PATH = Path("/data/options.json")

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

def load_config() -> Config:
    data: Dict[str, Any] = {}
    if OPTIONS_PATH.exists():
        try:
            data = json.loads(OPTIONS_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    return Config(**{k: v for k, v in data.items() if hasattr(Config, k)})
