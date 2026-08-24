"""Public usage event contract.

This module is the provider-neutral boundary for LLM usage telemetry. It intentionally
uses only standard-library types so framework adapters can translate their native
payloads before events reach sinks.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import InitVar, dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar


class EventKind(str, Enum):
    """Lifecycle shape of a usage event, independent of any framework callback type."""

    COMPLETION = "completion"
    RETRY = "retry"
    STREAM_CHUNK = "stream_chunk"
    STREAM_FINAL = "stream_final"


class EventStatus(str, Enum):
    """Provider-neutral call outcome.

    Legacy token usage records used ``"error"`` for failures. Public Hyperion events use
    ``"failure"`` and map it back to ``"error"`` only when serializing through
    ``to_legacy_dict``.
    """

    SUCCESS = "success"
    FAILURE = "failure"


def _unix_seconds_now() -> float:
    """Return the event timestamp unit used by legacy token usage records."""
    return datetime.now(timezone.utc).timestamp()


_CONTENT_UNSET = object()


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """Immutable public event for one provider-neutral LLM usage observation.

    ``timestamp`` is Unix seconds and ``duration_ms`` is milliseconds. The
    ``to_legacy_dict`` method preserves the twelve field names and units from
    ``reference.token_usage.TokenUsageRecord`` for migration-compatible sinks.

    ``event_id`` is a stable hash of ``duplicate_key`` unless supplied explicitly. The
    default ``duplicate_key`` is a hash of canonical structured identity fields for sinks
    that may receive the same event more than once. Its hash input includes lifecycle,
    request, model, timestamp, retry/stream position, error, and accounting fields so
    distinct observations do not collapse merely because they share request metadata, but
    the public key does not expose raw identifiers or error text.

    Raw prompt/completion content is outside the core event contract. Passing ``prompt``
    or ``completion`` content raises ``ValueError``; adapters must count, redact, or store
    content outside this provider-neutral telemetry type.

    Retry events require ``retry_of_event_id`` and ``attempt > 1``. Non-retry events may
    use higher attempt numbers for sink-defined identity but cannot link to a predecessor.
    Stream chunk/final events require ``stream_id`` and ``stream_sequence``; only
    ``STREAM_FINAL`` may set ``stream_final=True``.
    """

    LEGACY_FIELDS: ClassVar[tuple[str, ...]] = (
        "call_site",
        "model_name",
        "model_provider",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "duration_ms",
        "status",
        "error",
        "request_id",
        "session_id",
        "timestamp",
    )

    call_site: str = ""
    model_name: str = ""
    model_provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    status: EventStatus | str = EventStatus.SUCCESS
    error: str | None = None
    request_id: str = ""
    session_id: str = ""
    timestamp: float = field(default_factory=_unix_seconds_now)
    kind: EventKind | str = EventKind.COMPLETION
    event_id: str = ""
    duplicate_key: str = ""
    attempt: int = 1
    retry_of_event_id: str | None = None
    stream_id: str = ""
    stream_sequence: int | None = None
    stream_final: bool = False
    prompt: InitVar[str | None | object] = _CONTENT_UNSET
    completion: InitVar[str | None | object] = _CONTENT_UNSET

    def __post_init__(self, prompt: str | None | object, completion: str | None | object) -> None:
        if prompt is not _CONTENT_UNSET and prompt is not None:
            raise ValueError("raw content fields are not accepted by UsageEvent")
        if completion is not _CONTENT_UNSET and completion is not None:
            raise ValueError("raw content fields are not accepted by UsageEvent")
        _validate_types(self)
        object.__setattr__(self, "duration_ms", float(self.duration_ms))
        object.__setattr__(self, "timestamp", float(self.timestamp))
        status = _coerce_status(self.status)
        kind = _coerce_kind(self.kind)
        _validate_event(self, status, kind)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "kind", kind)
        if not self.duplicate_key:
            object.__setattr__(self, "duplicate_key", _default_duplicate_key(self))
        if not self.event_id:
            object.__setattr__(self, "event_id", _default_event_id(self.duplicate_key))

    def to_legacy_dict(self) -> dict[str, Any]:
        """Serialize as the legacy twelve-field token usage record."""
        data = self.to_dict()
        if data["status"] == EventStatus.FAILURE.value:
            data["status"] = "error"
        return {field_name: data[field_name] for field_name in self.LEGACY_FIELDS}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the public event without prompt or completion content."""
        data = {item.name: getattr(self, item.name) for item in fields(self)}
        return {
            key: value.value if isinstance(value, Enum) else value
            for key, value in data.items()
        }


def _coerce_status(value: EventStatus | str) -> EventStatus:
    if value == "error":
        return EventStatus.FAILURE
    return EventStatus(value)


def _coerce_kind(value: EventKind | str) -> EventKind:
    return EventKind(value)


def _validate_event(event: UsageEvent, status: EventStatus, kind: EventKind) -> None:
    if event.attempt < 1:
        raise ValueError("attempt must be at least 1")
    if not math.isfinite(event.duration_ms):
        raise ValueError("duration_ms must be finite")
    if not math.isfinite(event.timestamp):
        raise ValueError("timestamp must be finite")
    if event.duration_ms < 0:
        raise ValueError("duration_ms must be zero or greater")
    if event.timestamp < 0:
        raise ValueError("timestamp must be zero or greater")
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        if getattr(event, name) < 0:
            raise ValueError(f"{name} must be zero or greater")
    if event.stream_sequence is not None and event.stream_sequence < 0:
        raise ValueError("stream_sequence must be zero or greater")
    if kind is EventKind.RETRY:
        if event.retry_of_event_id is None:
            raise ValueError("retry events require retry_of_event_id")
        if event.attempt == 1:
            raise ValueError("retry events require attempt greater than 1")
    elif event.retry_of_event_id is not None:
        raise ValueError("non-retry events cannot set retry_of_event_id")
    if kind in {EventKind.STREAM_CHUNK, EventKind.STREAM_FINAL}:
        if not event.stream_id:
            raise ValueError("stream events require stream_id")
        if event.stream_sequence is None:
            raise ValueError("stream events require stream_sequence")
    elif event.stream_id or event.stream_sequence is not None or event.stream_final:
        raise ValueError("non-stream events cannot set stream fields")
    if kind is EventKind.STREAM_CHUNK and event.stream_final:
        raise ValueError("stream_chunk events must set stream_final=False")
    if kind is EventKind.STREAM_FINAL and not event.stream_final:
        raise ValueError("stream_final events must set stream_final=True")
    if status is EventStatus.SUCCESS and event.error is not None:
        raise ValueError("success events cannot set error")
    if status is EventStatus.FAILURE and not event.error:
        raise ValueError("failure events require error")


def _validate_types(event: UsageEvent) -> None:
    for name in (
        "call_site",
        "model_name",
        "model_provider",
        "request_id",
        "session_id",
        "event_id",
        "duplicate_key",
        "stream_id",
    ):
        if not isinstance(getattr(event, name), str):
            raise TypeError(f"{name} must be str")
    for name in ("input_tokens", "output_tokens", "total_tokens", "attempt"):
        if not _is_int(getattr(event, name)):
            raise TypeError(f"{name} must be int")
    for name in ("duration_ms", "timestamp"):
        if not _is_number(getattr(event, name)):
            raise TypeError(f"{name} must be int or float")
    if event.error is not None and not isinstance(event.error, str):
        raise TypeError("error must be str or None")
    if event.retry_of_event_id is not None and not isinstance(event.retry_of_event_id, str):
        raise TypeError("retry_of_event_id must be str or None")
    if event.stream_sequence is not None and not _is_int(event.stream_sequence):
        raise TypeError("stream_sequence must be int or None")
    if not isinstance(event.stream_final, bool):
        raise TypeError("stream_final must be bool")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _default_duplicate_key(event: UsageEvent) -> str:
    identity = {
        "version": "hyperion.usage_event.v2",
        "kind": event.kind.value if isinstance(event.kind, EventKind) else str(event.kind),
        "status": event.status.value if isinstance(event.status, EventStatus) else str(event.status),
        "request_id": event.request_id,
        "session_id": event.session_id,
        "call_site": event.call_site,
        "model_provider": event.model_provider,
        "model_name": event.model_name,
        "input_tokens": event.input_tokens,
        "output_tokens": event.output_tokens,
        "total_tokens": event.total_tokens,
        "duration_ms": event.duration_ms,
        "error": event.error,
        "timestamp": event.timestamp,
        "attempt": event.attempt,
        "retry_of_event_id": event.retry_of_event_id,
        "stream_id": event.stream_id,
        "stream_sequence": event.stream_sequence,
        "stream_final": event.stream_final,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return "dup_" + digest


def _default_event_id(duplicate_key: str) -> str:
    return "evt_" + hashlib.sha256(duplicate_key.encode("utf-8")).hexdigest()[:32]


__all__ = ["EventKind", "EventStatus", "UsageEvent"]
