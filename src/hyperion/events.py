"""The provider-neutral, standard-library-only Hyperion usage event contract."""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar


class EventStatus(str, Enum):
    """Provider-neutral outcome of one model-call observation."""

    SUCCESS = "success"
    FAILURE = "failure"


def _unix_seconds_now() -> float:
    return datetime.now(timezone.utc).timestamp()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _freeze_json(value: object, path: str = "details") -> object:
    """Validate and defensively freeze a JSON-compatible value."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if _is_int(value):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be str")
            frozen[key] = _freeze_json(item, f"{path}[{key!r}]")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise TypeError(f"{path} contains a non-JSON-compatible value: {type(value).__name__}")


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _truncate_details(value: Mapping[str, object], limit: int) -> tuple[Mapping[str, object], bool]:
    """Return deterministic details that fit ``limit`` bytes.

    Details are dropped as a unit when they do not fit. This avoids silently creating
    misleading partial provider metadata and is deterministic for nested structures,
    key order, and Unicode content. The separate boolean makes the loss explicit to
    consumers and to future monitor statistics.
    """

    # An empty mapping carries no provider metadata to lose, even when the
    # configured bound is zero. Non-empty metadata at that bound is explicit loss.
    if not value:
        return value, False
    if len(_json_bytes(value)) <= limit:
        return value, False
    marker: dict[str, object] = {"_hyperion_truncated": True}
    if len(_json_bytes(marker)) <= limit:
        return MappingProxyType(marker), True
    return MappingProxyType({}), True


def _bound_error(error: str | None) -> str | None:
    if error is None:
        return None
    limit = UsageEvent.MAX_ERROR_LENGTH
    if len(error.encode("utf-8")) <= limit:
        return error
    marker = "...[truncated]"
    available = limit - len(marker.encode("utf-8"))
    return error.encode("utf-8")[:available].decode("utf-8", errors="ignore") + marker


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """One immutable, final observation of a provider model operation.

    ``start_time`` and ``end_time`` are Unix seconds. ``to_legacy_dict`` maps the
    latter to the legacy ``timestamp`` field. Every attempt gets a fresh ``event_id``;
    attempts belonging to one logical operation explicitly reuse ``operation_id``.
    ``details`` accepts only JSON-compatible values and is frozen recursively.
    """

    SCHEMA_VERSION: ClassVar[str] = "1"
    DEFAULT_DETAILS_MAX_BYTES: ClassVar[int] = 16_384
    MAX_ERROR_LENGTH: ClassVar[int] = 4_096
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

    schema_version: str = SCHEMA_VERSION
    event_id: str = field(default_factory=lambda: _new_id("evt"))
    operation_id: str = field(default_factory=lambda: _new_id("op"))
    operation_type: str = "completion"
    provider: str = ""
    model: str = ""
    call_site: str = ""
    request_id: str = ""
    session_id: str = ""
    start_time: float = field(default_factory=_unix_seconds_now)
    end_time: float = field(default_factory=_unix_seconds_now)
    duration_ms: float = 0.0
    attempt: int = 1
    status: EventStatus | str = EventStatus.SUCCESS
    error: str | None = None
    stream_state: str = "not_streaming"
    usage_available: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    details: Mapping[str, object] = field(default_factory=dict)
    details_max_bytes: int = field(default=DEFAULT_DETAILS_MAX_BYTES, repr=False, compare=False)
    details_truncated: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._validate_types()
        if self.details_max_bytes < 0:
            raise ValueError("details_max_bytes must be zero or greater")
        if not self.event_id:
            object.__setattr__(self, "event_id", _new_id("evt"))
        if not self.operation_id:
            object.__setattr__(self, "operation_id", _new_id("op"))
        status = EventStatus.FAILURE if self.status == "error" else EventStatus(self.status)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "error", _bound_error(self.error))
        frozen_details = _freeze_json(self.details)
        if not isinstance(frozen_details, Mapping):
            raise TypeError("details must be a mapping")
        bounded_details, truncated = _truncate_details(frozen_details, self.details_max_bytes)
        object.__setattr__(self, "details", bounded_details)
        object.__setattr__(self, "details_truncated", truncated)
        self._validate_invariants()

    def _validate_types(self) -> None:
        for name in (
            "schema_version",
            "event_id",
            "operation_id",
            "operation_type",
            "provider",
            "model",
            "call_site",
            "request_id",
            "session_id",
            "stream_state",
        ):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be str")
        if not self.operation_type:
            raise ValueError("operation_type must not be empty")
        if not self.stream_state:
            raise ValueError("stream_state must not be empty")
        for name in ("start_time", "end_time", "duration_ms"):
            if not _is_number(getattr(self, name)):
                raise TypeError(f"{name} must be int or float")
        if not _is_int(self.attempt):
            raise TypeError("attempt must be int")
        if not isinstance(self.usage_available, bool):
            raise TypeError("usage_available must be bool")
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is not None and not _is_int(value):
                raise TypeError(f"{name} must be int or None")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be str or None")
        if not _is_int(self.details_max_bytes):
            raise TypeError("details_max_bytes must be int")
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping")

    def _validate_invariants(self) -> None:
        if self.attempt < 1:
            raise ValueError("attempt must be at least 1")
        for name in ("start_time", "end_time", "duration_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and zero or greater")
            object.__setattr__(self, name, value)
        if self.end_time < self.start_time:
            raise ValueError("end_time must be greater than or equal to start_time")
        if self.status is EventStatus.SUCCESS and self.error is not None:
            raise ValueError("success events cannot set error")
        if self.status is EventStatus.FAILURE and not self.error:
            raise ValueError("failure events require error")
        if not self.usage_available and any(
            count is not None for count in (self.input_tokens, self.output_tokens, self.total_tokens)
        ):
            raise ValueError("unavailable usage requires all token counts to be None")
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be zero or greater")
        if len(_json_bytes(self.details)) > self.details_max_bytes and self.details:
            raise AssertionError("bounded details exceed details_max_bytes")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the versioned normalized schema as JSON-compatible values."""

        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "provider": self.provider,
            "model": self.model,
            "call_site": self.call_site,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attempt": self.attempt,
            "status": self.status.value,
            "stream_state": self.stream_state,
            "usage_available": self.usage_available,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "error": self.error,
            "details": _json_ready(self.details),
            "details_truncated": self.details_truncated,
        }

    def to_legacy_dict(self) -> dict[str, Any]:
        """Serialize exactly the legacy twelve fields.

        The old ``timestamp`` field is mapped to ``end_time`` (Unix seconds), because
        the compatibility record represents the completed observation.
        """

        return {
            "call_site": self.call_site,
            "model_name": self.model,
            "model_provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "status": "error" if self.status is EventStatus.FAILURE else "success",
            "error": self.error,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "timestamp": self.end_time,
        }


__all__ = ["EventStatus", "UsageEvent"]
