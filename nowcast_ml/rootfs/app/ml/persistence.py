import json
import os
from pathlib import Path
from typing import Any, Dict

STATE_PATH = Path(os.environ.get("NOWCAST_STATE_PATH", "/data/station_model.json"))

def load_state() -> Dict[str, Any] | None:
    if not STATE_PATH.exists():
        return None
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    # A truncated or hand-edited file can parse to a non-dict; the engine
    # expects a mapping and would fail on the first .get().
    return data if isinstance(data, dict) else None

def atomic_save(obj: Dict[str, Any]) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)
