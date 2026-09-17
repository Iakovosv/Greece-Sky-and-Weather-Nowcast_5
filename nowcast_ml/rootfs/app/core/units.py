import re
from typing import Optional
_NUM_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+")
def parse_float(val) -> Optional[float]:
    if val is None: return None
    if isinstance(val,(int,float)): return float(val)
    m=_NUM_RE.search(str(val))
    if not m: return None
    try: return float(m.group(0))
    except Exception: return None
