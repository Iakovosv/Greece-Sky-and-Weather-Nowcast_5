from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class RainDeltaState:
    prev_cum: Optional[float] = None
    source: str = "none"  # event|daily|none

def choose_cumulative(parsed: dict) -> Tuple[Optional[float], str]:
    # Prefer event rain if available
    ev = parsed.get("rain_event_mm")
    dy = parsed.get("rain_daily_mm")
    if ev is not None:
        return float(ev), "event"
    if dy is not None:
        return float(dy), "daily"
    return None, "none"

def compute_rain_1m(cum_now: Optional[float], state: RainDeltaState) -> float:
    if cum_now is None:
        return 0.0
    if state.prev_cum is None:
        state.prev_cum = cum_now
        return 0.0
    # reset / rollover handling
    if cum_now + 1e-6 < state.prev_cum:
        # assume reset
        state.prev_cum = cum_now
        return 0.0
    delta = cum_now - state.prev_cum
    state.prev_cum = cum_now
    if delta < 0:
        return 0.0
    # guard absurd spikes (sensor glitch)
    if delta > 50.0:
        return 0.0
    return float(delta)
