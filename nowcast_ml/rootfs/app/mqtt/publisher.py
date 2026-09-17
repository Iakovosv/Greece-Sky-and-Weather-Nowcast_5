from __future__ import annotations
import json
from typing import Any, Dict, Optional
import paho.mqtt.client as mqtt


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in s)


class MqttPublisher:
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        topic_prefix: str,
        discovery_prefix: str,
        station_name: str,
        logger,
    ):
        self.host = host
        self.port = port
        self.username = username or None
        self.password = password or None
        self.topic_prefix = topic_prefix.strip("/")
        self.discovery_prefix = discovery_prefix.strip("/")
        self.station_name = station_name
        self.log = logger

        self.client: Optional[mqtt.Client] = None
        self.node_id = _safe(f"greece_sky_and_weather_nowcast_{station_name}")
        self.base_state = f"{self.topic_prefix}/{_safe(station_name)}"

        # command topic for HA button/manual actions
        self.command_topic = f"{self.base_state}/command"

        self._discovery_sent = False
        self._save_requested = False

    # --- command flag API (loop.py will poll this safely) ---
    def consume_save_requested(self) -> bool:
        """Return True once per requested save (edge-trigger)."""
        if self._save_requested:
            self._save_requested = False
            return True
        return False

    # --- MQTT plumbing ---
    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8", errors="ignore").strip()
            if not payload:
                return
            # allow either plain text or json
            if payload.lower() in ("save", "save_now", "persist"):
                self._save_requested = True
                return
            data = json.loads(payload)
            if isinstance(data, dict) and str(data.get("action", "")).lower() in ("save", "save_now", "persist"):
                self._save_requested = True
        except Exception:
            # ignore malformed messages
            return

    def connect(self) -> bool:
        try:
            self.client = mqtt.Client(client_id=self.node_id, clean_session=True)
            if self.username:
                self.client.username_pw_set(self.username, self.password)

            self.client.on_message = self._on_message

            self.client.connect(self.host, self.port, keepalive=30)

            # subscribe to command topic
            self.client.subscribe(self.command_topic, qos=0)

            self.client.loop_start()
            self.log.info(f"MQTT connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            self.client = None
            self.log.warning(f"MQTT connect failed: {e}")
            return False

    def _pub(self, topic: str, payload: str, retain: bool = True) -> None:
        if not self.client:
            return
        self.client.publish(topic, payload, qos=0, retain=retain)

    # --- HA discovery ---
    def publish_discovery(self) -> None:
        if self._discovery_sent or not self.client:
            return

        device = {
            "identifiers": [self.node_id],
            "name": f"Greece Sky and Weather Nowcast ({self.station_name})",
            "manufacturer": "Greece Sky and Weather",
            "model": "Nowcast Add-on",
        }

        def add_sensor(object_id: str, name: str, unit: str, icon: str):
            cfg = {
                "name": name,
                "unique_id": f"{self.node_id}_{object_id}",
                "state_topic": f"{self.base_state}/state",
                "value_template": "{{ value_json.%s }}" % object_id,
                "unit_of_measurement": unit,
                "icon": icon,
                "device": device,
                "availability_topic": f"{self.base_state}/availability",
            }
            disc = f"{self.discovery_prefix}/sensor/{self.node_id}/{object_id}/config"
            self._pub(disc, json.dumps(cfg), retain=True)

        # probability
        add_sensor("pop_30m", "Nowcast PoP 30m", "%", "mdi:weather-rainy")
        add_sensor("pop_60m", "Nowcast PoP 60m", "%", "mdi:weather-rainy")
        add_sensor("pop_120m", "Nowcast PoP 120m", "%", "mdi:weather-rainy")
        add_sensor("pop_360m", "Nowcast PoP 360m", "%", "mdi:weather-rainy")

        # rain quantiles
        for h in (30, 60, 120, 360):
            add_sensor(f"p10_{h}m", f"Nowcast Rain P10 {h}m", "mm", "mdi:water")
            add_sensor(f"p50_{h}m", f"Nowcast Rain P50 {h}m", "mm", "mdi:water")
            add_sensor(f"p90_{h}m", f"Nowcast Rain P90 {h}m", "mm", "mdi:water")

        # status sensor with attributes
        status_object = "nowcast_status"
        status_cfg = {
            "name": "Nowcast Status",
            "unique_id": f"{self.node_id}_{status_object}",
            "state_topic": f"{self.base_state}/status",
            "value_template": "{{ value_json.state }}",
            "json_attributes_topic": f"{self.base_state}/status",
            "icon": "mdi:chart-timeline-variant",
            "device": device,
            "availability_topic": f"{self.base_state}/availability",
        }
        self._pub(
            f"{self.discovery_prefix}/sensor/{self.node_id}/{status_object}/config",
            json.dumps(status_cfg),
            retain=True,
        )

        # --- NEW: HA Button for manual save ---
        btn_object = "save_model"
        btn_cfg = {
            "name": "Nowcast Save Model",
            "unique_id": f"{self.node_id}_{btn_object}",
            "command_topic": self.command_topic,
            "payload_press": '{"action":"save"}',
            "icon": "mdi:content-save",
            "device": device,
            "availability_topic": f"{self.base_state}/availability",
        }
        self._pub(
            f"{self.discovery_prefix}/button/{self.node_id}/{btn_object}/config",
            json.dumps(btn_cfg),
            retain=True,
        )

        self._pub(f"{self.base_state}/availability", "online", retain=True)
        self._discovery_sent = True
        self.log.info("MQTT discovery published")

    def publish_state(self, payload: Dict[str, Any]) -> None:
        if not self.client:
            return
        self._pub(f"{self.base_state}/state", json.dumps(payload), retain=False)

    def publish_status(self, status: Dict[str, Any]) -> None:
        if not self.client:
            return
        self._pub(f"{self.base_state}/status", json.dumps(status), retain=False)
