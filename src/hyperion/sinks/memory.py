"""Thread-safe in-memory usage event sink."""

from __future__ import annotations

from collections import deque
from threading import Lock

from hyperion.events import UsageEvent


class MemorySink:
    """Bounded FIFO sink for tests, local inspection and lightweight integrations.

    Duplicate events are retained by default. When ``deduplicate_by_event_id`` is true,
    events with an ``event_id`` already retained by the sink are ignored.
    """

    def __init__(self, max_events: int = 1_000, *, deduplicate_by_event_id: bool = False) -> None:
        if not isinstance(max_events, int) or isinstance(max_events, bool):
            raise TypeError("max_events must be int")
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        self._events: deque[UsageEvent] = deque(maxlen=max_events)
        self._event_ids: set[str] = set()
        self._deduplicate_by_event_id = deduplicate_by_event_id
        self._lock = Lock()

    def emit(self, event: UsageEvent) -> None:
        """Retain an event according to this sink's bounded FIFO policy."""

        with self._lock:
            if self._deduplicate_by_event_id and event.event_id in self._event_ids:
                return
            if self._deduplicate_by_event_id and len(self._events) == self._events.maxlen:
                evicted = self._events[0]
                self._event_ids.discard(evicted.event_id)
            self._events.append(event)
            if self._deduplicate_by_event_id:
                self._event_ids.add(event.event_id)

    def snapshot(self) -> tuple[UsageEvent, ...]:
        """Return an immutable snapshot without exposing the backing storage."""

        with self._lock:
            return tuple(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


__all__ = ["MemorySink"]
