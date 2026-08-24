"""Small, dependency-free helpers shared by optional Hyperion adapters."""

# Adapter callbacks are telemetry boundaries: every provider/framework failure is
# intentionally swallowed so application calls remain unaffected.
# ruff: noqa: BLE001, B009, S110

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from datetime import datetime, timezone

from hyperion.config import HyperionConfig
from hyperion.context import RequestContext, current_context
from hyperion.events import UsageEvent
from hyperion.monitor import UsageMonitor

_CONTENT_KEY_PARTS = (
    "prompt",
    "completion",
    "message",
    "content",
    "tool_call",
    "tool_argument",
    "tool_args",
    "input_text",
    "output_text",
)


def infer_provider(model: object, resolver: object = None) -> tuple[str, str]:
    """Return a normalized model and provider using the documented heuristics."""

    model_name = model.strip() if isinstance(model, str) else ""
    if callable(resolver):
        try:
            resolved = resolver(model_name)
        except Exception:  # provider hooks are telemetry-only
            resolved = None
        if isinstance(resolved, str) and resolved.strip():
            return model_name, resolved.strip().lower()
    if "/" in model_name:
        return model_name, model_name.split("/", 1)[0].strip().lower()
    lower = model_name.lower()
    if "gpt" in lower or "o1" in lower or "o3" in lower:
        return model_name, "openai"
    if "claude" in lower:
        return model_name, "anthropic"
    if "gemini" in lower or "models/" in lower:
        return model_name, "google"
    return model_name, "unknown"


def context_metadata(context: RequestContext | None = None) -> dict[str, object]:
    current = current_context() if context is None else context
    result = dict(current.metadata)
    for key in ("request_id", "session_id", "call_site"):
        value = getattr(current, key)
        # A context's dedicated identity fields fill gaps in metadata, including
        # metadata entries that are present but blank.  This keeps a provider's
        # empty metadata from masking request-scoped identity.
        if _meaningful_identity(value) and not _meaningful_identity(result.get(key)):
            result[key] = value
    return result


def merged_metadata(
    nested: object = None,
    explicit: object = None,
    *,
    context: RequestContext | None = None,
    fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Merge context, nested provider metadata and explicit metadata in that order."""

    result = context_metadata(context)
    if isinstance(nested, Mapping):
        _merge_metadata(result, nested)
    if isinstance(explicit, Mapping):
        # LiteLLM's top-level metadata is more specific than litellm_params,
        # but only meaningful values should override an existing value.
        _merge_metadata(result, explicit)
    if fields:
        # Handler constructor values are explicit integration metadata.
        _merge_metadata(result, fields)
    return result


def _meaningful_identity(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _merge_metadata(target: dict[str, object], values: Mapping[str, object]) -> None:
    for key, value in values.items():
        if not isinstance(key, str):
            continue
        if key in {"request_id", "session_id", "call_site"} and not _meaningful_identity(value):
            continue
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        target[key] = value


def text_field(metadata: Mapping[str, object], key: str, default: str = "") -> str:
    value = metadata.get(key, default)
    return value if isinstance(value, str) and value.strip() else default


def operation_and_attempt(
    metadata: Mapping[str, object],
    *,
    fallback_operation: str | None = None,
    attempt: object = None,
) -> tuple[str | None, int]:
    operation = metadata.get("operation_id")
    if not isinstance(operation, str) or not operation:
        operation = (
            fallback_operation
            if isinstance(fallback_operation, str) and fallback_operation
            else None
        )
    candidate = metadata.get("attempt", attempt)
    if isinstance(candidate, bool):
        candidate = None
    if isinstance(candidate, int) and candidate >= 1:
        return operation, candidate
    return operation, 1


def extract_usage(payload: object) -> tuple[bool, int | None, int | None, int | None]:
    """Extract complete, non-negative integer usage; malformed usage is unavailable."""

    if isinstance(payload, Mapping):
        values = payload
    else:
        values = {}
        if payload is not None:
            try:
                values = {
                    "prompt_tokens": getattr(payload, "prompt_tokens"),
                    "completion_tokens": getattr(payload, "completion_tokens"),
                    "total_tokens": getattr(payload, "total_tokens"),
                }
            except Exception:
                return False, None, None, None
    try:
        input_tokens = values["prompt_tokens"]
        output_tokens = values["completion_tokens"]
        total_tokens = values["total_tokens"]
    except (KeyError, TypeError):
        return False, None, None, None
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (input_tokens, output_tokens, total_tokens)
    ):
        return False, None, None, None
    return True, input_tokens, output_tokens, total_tokens


def response_usage(response: object) -> tuple[bool, int | None, int | None, int | None]:
    if isinstance(response, Mapping):
        return extract_usage(response.get("usage"))
    try:
        return extract_usage(getattr(response, "usage"))
    except Exception:
        return False, None, None, None


_UNSAFE_JSON = object()


def _safe_json(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _UNSAFE_JSON
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or is_content_key(key):
                continue
            cleaned = _safe_json(item)
            if cleaned is not _UNSAFE_JSON:
                result[key] = cleaned
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            cleaned = _safe_json(item)
            if cleaned is not _UNSAFE_JSON:
                result.append(cleaned)
        return result
    return _UNSAFE_JSON


def is_content_key(key: str) -> bool:
    lower = key.lower().replace("-", "_")
    return any(part in lower for part in _CONTENT_KEY_PARTS) or lower in {
        "messages",
        "contents",
        "arguments",
        "tool_arguments",
    }


_NORMALIZED_DETAIL_KEYS = {
    "schema_version",
    "event_id",
    "operation_id",
    "operation_type",
    "attempt",
    "model",
    "model_name",
    "provider",
    "model_provider",
    "call_site",
    "request_id",
    "session_id",
    "start_time",
    "end_time",
    "timestamp",
    "duration_ms",
    "timing",
    "status",
    "stream",
    "streaming",
    "stream_state",
    "stream_id",
    "stream_sequence",
    "stream_final",
    "usage",
    "usage_available",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "error",
    "details",
    "details_max_bytes",
    "details_truncated",
}


def namespaced_details(namespace: str, metadata: Mapping[str, object]) -> dict[str, object]:
    cleaned = _safe_json(_strip_normalized(metadata))
    if not isinstance(cleaned, Mapping):
        return {}
    return {namespace: dict(cleaned)} if cleaned else {}


def _strip_normalized(value: object) -> object:
    """Remove normalized event fields and private prompt-bearing content recursively."""

    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            normalized = key.lower().replace("-", "_")
            if normalized in _NORMALIZED_DETAIL_KEYS or is_content_key(key):
                continue
            result[key] = _strip_normalized(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_strip_normalized(item) for item in value]
    return value


def callback_times(start: object, end: object) -> tuple[float, float, float]:
    """Convert callback datetimes to safe Unix timestamps and duration."""

    def convert(value: object) -> float | None:
        if isinstance(value, datetime):
            try:
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                result = value.timestamp()
                return result if math.isfinite(result) and result >= 0 else None
            except Exception:
                return None
        return None

    first, last = convert(start), convert(end)
    if first is None or last is None or last < first:
        now = time.time()
        if not math.isfinite(now) or now < 0:
            now = 0.0
        return now, now, 0.0
    return first, last, (last - first) * 1000.0


def wall_clock_times(start: object) -> tuple[float, float, float]:
    try:
        first = float(start)
        now = time.time()
        if math.isfinite(first) and first >= 0 and math.isfinite(now) and now >= first:
            return first, now, (now - first) * 1000.0
    except Exception:
        pass
    now = time.time()
    if not math.isfinite(now) or now < 0:
        now = 0.0
    return now, now, 0.0


def emit_event(
    monitor: UsageMonitor,
    config: HyperionConfig,
    *,
    model: object,
    metadata: Mapping[str, object],
    start_time: float,
    end_time: float,
    duration_ms: float,
    status: str,
    error: str | None,
    stream_state: str,
    operation_id: str | None,
    attempt: int,
    usage: tuple[bool, int | None, int | None, int | None],
    details_namespace: str,
) -> None:
    """Construct and dispatch one event; malformed telemetry never escapes."""

    try:
        model_name, provider = infer_provider(model, config.provider_resolver)
        available, input_tokens, output_tokens, total_tokens = usage
        if status == "failure":
            if error is None or not isinstance(error, str) or not error:
                error = "unknown error"
        else:
            error = None
        event = UsageEvent(
            model=model_name,
            provider=provider,
            call_site=text_field(metadata, "call_site", config.default_call_site or "application"),
            request_id=text_field(metadata, "request_id"),
            session_id=text_field(metadata, "session_id"),
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            status=status,
            error=error,
            stream_state=stream_state,
            operation_id=operation_id or "",
            attempt=attempt if isinstance(attempt, int) and attempt >= 1 else 1,
            usage_available=available,
            input_tokens=input_tokens if available else None,
            output_tokens=output_tokens if available else None,
            total_tokens=total_tokens if available else None,
            details=namespaced_details(details_namespace, metadata),
            details_max_bytes=config.details_max_bytes,
        )
        monitor.emit(event)
    except Exception:
        return


def safe_error(value: object) -> str:
    try:
        if isinstance(value, BaseException):
            text = str(value)
        elif isinstance(value, str):
            text = value
        else:
            text = "provider error"
    except Exception:
        text = "provider error"
    return text or "unknown error"


__all__ = [
    "callback_times",
    "context_metadata",
    "emit_event",
    "extract_usage",
    "infer_provider",
    "merged_metadata",
    "namespaced_details",
    "operation_and_attempt",
    "response_usage",
    "safe_error",
    "wall_clock_times",
]
