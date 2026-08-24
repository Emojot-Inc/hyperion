"""Thread-safe in-memory usage event sink."""

from __future__ import annotations

from collections import deque
from threading import Lock

from hyperion.events import UsageEvent


class MemorySink:
    """Bounded FIFO sink for tests, local inspection and lightweight integrations."""

    def __init__(self, max_events: int = 1_000) -> None:
        if not isinstance(max_events, int) or isinstance(max_events, bool):
            raise TypeError("max_events must be int")
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        self._events: deque[UsageEvent] = deque(maxlen=max_events)
        self._lock = Lock()

    def emit(self, event: UsageEvent) -> None:
        """Retain an event according to this sink's bounded FIFO policy."""

        with self._lock:
            self._events.append(event)

    def snapshot(self) -> tuple[UsageEvent, ...]:
        """Return an immutable snapshot without exposing the backing storage."""

        with self._lock:
            return tuple(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


__all__ = ["MemorySink"]
