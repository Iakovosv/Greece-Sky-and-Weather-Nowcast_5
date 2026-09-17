import json
from pathlib import Path
from typing import Any, Dict

STATE_PATH = Path("/data/station_model.json")

def load_state() -> Dict[str, Any] | None:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None

def atomic_save(obj: Dict[str, Any]) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)
