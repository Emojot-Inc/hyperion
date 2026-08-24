"""Monitoring lifecycle and failure-isolated event dispatch."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import Lock
from typing import Any

from hyperion.events import UsageEvent
from hyperion.protocols import UsageEventSink


@dataclass(frozen=True, slots=True)
class SinkErrorReport:
    """Sanitized sink failure report passed to ``on_sink_error`` callbacks.

    The report intentionally contains primitive fields only. It excludes the raw sink,
    event, exception object, exception message, and traceback so callback boundaries do
    not leak prompt-adjacent identifiers, sink configuration, or provider error text.
    ``event_id`` is a stable hash-derived value for correlating repeated sink failures
    without exposing the caller-provided ``UsageEvent.event_id`` text.
    """

    sink_type: str
    sink_name: str
    event_id: str
    exception_type: str

    def to_log_extra(self) -> dict[str, str]:
        """Return structured logging metadata with the same sanitized fields."""

        return {
            "sink_type": self.sink_type,
            "sink_name": self.sink_name,
            "event_id": self.event_id,
            "exception_type": self.exception_type,
        }


SinkErrorCallback = Callable[[SinkErrorReport], None]


@dataclass(frozen=True, slots=True)
class UsageMonitorStats:
    """Immutable counters for one monitor.

    ``delivered`` counts successful sink deliveries, so one event sent to three sinks
    contributes three.  ``sink_failures`` likewise counts failed sink attempts.
    """

    emitted: int = 0
    delivered: int = 0
    sink_failures: int = 0
    truncated_details: int = 0


class UsageMonitor:
    """Dispatches usage events to a fixed snapshot of configured sinks.

    The monitor does not deduplicate. Calling ``emit`` twice with the same event attempts
    delivery twice to each sink, giving sinks control over their own duplicate policy.
    """

    def __init__(
        self,
        sinks: Iterable[UsageEventSink] = (),
        *,
        on_sink_error: SinkErrorCallback | None = None,
        logger: Any | None = None,
        diagnostic_logging: bool = True,
        enabled: bool = True,
    ) -> None:
        self._sinks = tuple(sinks)
        self._on_sink_error = on_sink_error
        self._logger = logging.getLogger(__name__) if logger is None else logger
        if not isinstance(diagnostic_logging, bool):
            raise TypeError("diagnostic_logging must be bool")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        self._diagnostic_logging = diagnostic_logging
        self._enabled = enabled
        self._stats_lock = Lock()
        self._emitted = 0
        self._delivered = 0
        self._sink_failures = 0
        self._truncated_details = 0

    @property
    def sinks(self) -> tuple[UsageEventSink, ...]:
        """The immutable sink snapshot this monitor dispatches to."""

        return self._sinks

    def emit(self, event: UsageEvent) -> None:
        """Emit an event to all configured sinks, isolating sink failures."""

        if not self._enabled:
            return
        with self._stats_lock:
            self._emitted += 1
            if event.details_truncated:
                self._truncated_details += 1
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception as exc:  # noqa: BLE001 - sink failures must not escape
                with self._stats_lock:
                    self._sink_failures += 1
                self._report_sink_error(sink, event, exc)
            else:
                with self._stats_lock:
                    self._delivered += 1

    def stats(self) -> UsageMonitorStats:
        """Return a thread-safe immutable snapshot of monitor counters."""

        with self._stats_lock:
            return UsageMonitorStats(
                emitted=self._emitted,
                delivered=self._delivered,
                sink_failures=self._sink_failures,
                truncated_details=self._truncated_details,
            )

    def _report_sink_error(
        self,
        sink: UsageEventSink,
        event: UsageEvent,
        exc: BaseException,
    ) -> None:
        report = _sink_error_report(sink, event, exc)
        try:
            if self._on_sink_error is not None:
                self._on_sink_error(report)
                return
            if self._diagnostic_logging:
                self._logger.warning(
                    "Hyperion sink failed while emitting usage event",
                    extra=report.to_log_extra(),
                )
        except Exception:  # noqa: BLE001 - reporting must not break model calls
            return


def _sink_error_report(
    sink: UsageEventSink,
    event: UsageEvent,
    exc: BaseException,
) -> SinkErrorReport:
    sink_type = type(sink).__qualname__
    return SinkErrorReport(
        sink_type=sink_type,
        sink_name=sink_type,
        event_id=_sanitized_report_event_id(event.event_id),
        exception_type=type(exc).__name__,
    )


def _sanitized_report_event_id(event_id: str) -> str:
    digest = hashlib.sha256(event_id.encode("utf-8")).hexdigest()
    return "evt_" + digest[:32]


__all__ = ["SinkErrorCallback", "SinkErrorReport", "UsageMonitor", "UsageMonitorStats"]
