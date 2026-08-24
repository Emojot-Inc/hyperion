"""Standard-library protocols for Hyperion core boundaries."""

from __future__ import annotations

from typing import Protocol

from hyperion.events import UsageEvent


class UsageEventSink(Protocol):
    """Receives provider-neutral usage events."""

    def emit(self, event: UsageEvent) -> None:
        """Store, forward or otherwise observe a usage event."""


__all__ = ["UsageEventSink"]
