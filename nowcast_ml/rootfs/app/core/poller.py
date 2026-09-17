import requests
from typing import Optional, Dict, Any
def poll(url: str, timeout_s: int=3) -> Optional[Dict[str, Any]]:
    try:
        r=requests.get(url, timeout=timeout_s)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None
