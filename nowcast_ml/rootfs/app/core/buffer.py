from collections import deque
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


class RingBuffer:
    def __init__(self, max_minutes: int = 240):
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=max_minutes)

    def add(self, row: Dict[str, Any]) -> None:
        self.buffer.append(row)

    def last(self) -> Optional[Dict[str, Any]]:
        return self.buffer[-1] if self.buffer else None

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self.buffer)

    def last_n(self, n: int) -> List[Dict[str, Any]]:
        """
        Return last n rows (most recent). If n<=0 -> [].
        Keeps order oldest->newest within that slice.
        """
        if n <= 0 or not self.buffer:
            return []
        if n >= len(self.buffer):
            return list(self.buffer)
        return list(self.buffer)[-n:]

    def get_minutes_ago(
        self, minutes: int, now_ts: Any = None, tolerance_min: float = 3.0
    ) -> Optional[Dict[str, Any]]:
        """
        Return the row `minutes` minutes before `now_ts`.

        One row is appended per successful poll and the add-on polls once a
        minute, so a row offset normally equals the same number of minutes. Two
        situations break that equivalence:

        * a poll outage, where rows are missing entirely;
        * a poll interval other than 60s.

        When the stored timestamps show a minute-scale cadence the window is
        verified against them and a row that does not actually cover the
        requested span is rejected, so a "10m" delta is never silently a 40m
        change measured across a gap. When the timestamps do not show that
        cadence (a synthetic or accelerated clock) the row offset is trusted,
        which is the same assumption the poll interval documents.
        """
        if minutes <= 0 or not self.buffer:
            return None

        need = minutes + 1
        if len(self.buffer) < need:
            return None
        candidate = list(self.buffer)[-need]

        if now_ts is None or len(self.buffer) < 2:
            return candidate

        rows = list(self.buffer)
        prev_ts = _parse_ts(rows[-2].get("ts"))
        cur_ts = _parse_ts(rows[-1].get("ts"))
        cand_ts = _parse_ts(candidate.get("ts"))
        if prev_ts is None or cur_ts is None or cand_ts is None:
            return candidate

        cadence_min = (cur_ts - prev_ts).total_seconds() / 60.0
        if not (0.5 <= cadence_min <= 2.0):
            # Timestamps are not on a minute cadence; trust the row offset.
            return candidate

        span_min = (cur_ts - cand_ts).total_seconds() / 60.0
        if abs(span_min - minutes) > max(tolerance_min, minutes * 0.5):
            return None
        return candidate
