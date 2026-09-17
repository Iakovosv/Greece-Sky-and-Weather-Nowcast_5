from collections import deque
from typing import Any, Deque, Dict, List, Optional


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

    def get_minutes_ago(self, minutes: int) -> Optional[Dict[str, Any]]:
        """
        Returns the row from `minutes` minutes ago, assuming 1 row/minute (poll_interval_sec=60).

        Correct semantics:
          minutes=1  -> previous row (t-1)
          minutes=10 -> row at t-10

        If not enough history -> None
        """
        if minutes <= 0 or not self.buffer:
            return None

        # need at least minutes+1 rows so that "1 minute ago" exists
        need = minutes + 1
        if len(self.buffer) < need:
            return None

        return list(self.buffer)[-need]
