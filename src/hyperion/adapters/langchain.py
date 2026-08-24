"""Optional LangChain callback integration.

LangChain has no safe public global interception hook.  ``register`` therefore only
configures a default handler factory; applications attach the returned handler to the
callback configuration they own.
"""

# Callback boundaries are deliberately fail-open; no provider callback failure may
# replace an application result or exception.
# ruff: noqa: BLE001, B009

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from hyperion.config import HyperionConfig
from hyperion.monitor import UsageMonitor

from ._common import (
    _merge_metadata,
    emit_event,
    extract_usage,
    merged_metadata,
    operation_and_attempt,
    safe_error,
)

_TRACKER_CAPACITY = 1024


def _config_or_default(config: HyperionConfig | None) -> HyperionConfig:
    if config is not None:
        return config
    return HyperionConfig(enabled=True, integrations=(), sinks=())


@dataclass
class _RunState:
    start: float
    metadata: dict[str, object]
    operation_id: str | None
    attempt: int
    streaming: bool
    parent_run_id: str | None


class LangChainUsageCallback(BaseCallbackHandler):
    """BaseCallbackHandler-compatible handler emitting one event per LLM run."""

    def __init__(
        self,
        monitor: UsageMonitor | str | None = None,
        config: HyperionConfig | None = None,
        *,
        call_site: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        operation_id: str | None = None,
        metadata: dict[str, object] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        # Preserve the reference constructor's first positional ``call_site`` form.
        if isinstance(monitor, str) and call_site is None:
            call_site, monitor = monitor, None
        elif isinstance(monitor, HyperionConfig) and config is None:
            config, monitor = monitor, None
        self.config = _config_or_default(config)
        self.monitor = (
            monitor
            if monitor is not None
            else UsageMonitor(
                self.config.sinks,
                enabled=self.config.enabled,
                diagnostic_logging=bool(self.config.diagnostic_logging),
                logger=self.config.diagnostic_logger,
            )
        )
        fields: dict[str, object] = {}
        for key, value in {
            "call_site": call_site,
            "request_id": request_id,
            "session_id": session_id,
            "operation_id": operation_id,
        }.items():
            if value is not None:
                fields[key] = value
        if metadata:
            _merge_metadata(fields, metadata)
        self._fields = fields
        # Active callback state is lifecycle state. Historical retry bookkeeping
        # is bounded separately.
        self._runs: OrderedDict[str, _RunState] = OrderedDict()
        self._operation_attempts: OrderedDict[str, int] = OrderedDict()
        self._failed_parent_operations: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.RLock()

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str] | None,
        *,
        run_id: object = None,
        parent_run_id: object = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            start = time.time()
            if not math.isfinite(start) or start < 0:
                start = 0.0
            # Framework callback metadata is explicit and therefore wins over defaults;
            # meaningful values only may replace captured context identity.
            merged = merged_metadata(self._fields, metadata)
            _merge_metadata(
                merged,
                {
                    key: value
                    for key, value in kwargs.items()
                    if key in {"operation_id", "attempt", "streaming", "model_name"}
                },
            )
            operation, explicit_attempt = operation_and_attempt(
                merged,
                fallback_operation=_run_key(run_id) if run_id is not None else None,
                attempt=kwargs.get("attempt"),
            )
            parent_key = _parent_key(parent_run_id)
            has_explicit_operation = _has_operation_id(merged)
            retry_signaled = _retry_signaled(merged, kwargs, tags, explicit_attempt)
            with self._lock:
                handed_off = (
                    self._failed_parent_operations.pop(parent_key, None) if parent_key else None
                )
                if handed_off and not has_explicit_operation and retry_signaled:
                    operation = handed_off
                if operation:
                    prior = self._operation_attempts.get(operation, 0)
                    attempt = explicit_attempt if "attempt" in merged else prior + 1
                    self._operation_attempts[operation] = max(prior, attempt)
                    self._operation_attempts.move_to_end(operation)
                    _trim_tracker(self._operation_attempts)
                else:
                    attempt = explicit_attempt
            state = _RunState(
                start=start,
                metadata=merged,
                operation_id=operation,
                attempt=attempt,
                streaming=bool(kwargs.get("streaming", kwargs.get("stream", False))),
                parent_run_id=parent_key,
            )
            with self._lock:
                self._runs[_run_key(run_id)] = state
                self._runs.move_to_end(_run_key(run_id))
        except Exception:
            return

    def on_llm_new_token(self, token: object, **kwargs: Any) -> None:
        # Deliberately no token/chunk events: accounting is finalized by on_llm_end.
        return

    def on_llm_end(self, response: object, *, run_id: object = None, **kwargs: Any) -> None:
        state = self._pop_state(run_id)
        try:
            now = time.time()
            if not math.isfinite(now) or now < 0:
                now = state.start if state else 0.0
            if state is None:
                state = _RunState(
                    now,
                    self._metadata(kwargs.get("metadata")),
                    None,
                    1,
                    bool(kwargs.get("streaming", False)),
                    None,
                )
            end = max(now, state.start)
            llm_output = _llm_output(response)
            model = _value(llm_output, "model_name", "")
            usage = extract_usage(_value(llm_output, "token_usage", None))
            metadata = dict(state.metadata)
            _merge_metadata(metadata, kwargs.get("metadata") or {})
            emit_event(
                self.monitor,
                self.config,
                model=model,
                metadata=metadata,
                start_time=state.start,
                end_time=end,
                duration_ms=(end - state.start) * 1000.0,
                status="success",
                error=None,
                stream_state="final" if state.streaming else "not_streaming",
                operation_id=state.operation_id,
                attempt=state.attempt,
                usage=usage,
                details_namespace="langchain",
            )
            if state.operation_id:
                with self._lock:
                    # A successful callback is authoritative; a later retry
                    # must start a fresh logical operation unless it provides
                    # an explicit attempt in metadata.
                    self._operation_attempts.pop(state.operation_id, None)
                    _clear_failed_parent_operation(
                        self._failed_parent_operations,
                        state.parent_run_id,
                        state.operation_id,
                    )
        except Exception:
            return

    def on_llm_error(self, error: BaseException, *, run_id: object = None, **kwargs: Any) -> None:
        state = self._pop_state(run_id)
        try:
            now = time.time()
            if not math.isfinite(now) or now < 0:
                now = state.start if state else 0.0
            if state is None:
                state = _RunState(
                    now,
                    self._metadata(kwargs.get("metadata")),
                    None,
                    1,
                    bool(kwargs.get("streaming", False)),
                    None,
                )
            end = max(now, state.start)
            metadata = dict(state.metadata)
            _merge_metadata(metadata, kwargs.get("metadata") or {})
            model = kwargs.get("model_name", "")
            emit_event(
                self.monitor,
                self.config,
                model=model,
                metadata=metadata,
                start_time=state.start,
                end_time=end,
                duration_ms=(end - state.start) * 1000.0,
                status="failure",
                error=safe_error(error),
                stream_state="cancelled" if _is_cancelled(error) else "failed",
                operation_id=state.operation_id,
                attempt=state.attempt,
                usage=(False, None, None, None),
                details_namespace="langchain",
            )
            if state.operation_id:
                with self._lock:
                    _remember_attempt(self._operation_attempts, state.operation_id, state.attempt)
                    if state.parent_run_id:
                        self._failed_parent_operations[state.parent_run_id] = state.operation_id
                        self._failed_parent_operations.move_to_end(state.parent_run_id)
                        _trim_tracker(self._failed_parent_operations)
        except Exception:
            return

    def _metadata(self, explicit: object = None) -> dict[str, object]:
        return merged_metadata(self._fields, explicit)

    def _pop_state(self, run_id: object) -> _RunState | None:
        try:
            with self._lock:
                return self._runs.pop(_run_key(run_id), None)
        except Exception:
            return None


def _run_key(run_id: object) -> str:
    if isinstance(run_id, str):
        return run_id
    if run_id is None:
        return "__default__"
    try:
        return str(run_id)
    except Exception:
        return "__unknown__"


def _parent_key(parent_run_id: object) -> str | None:
    if parent_run_id is None:
        return None
    return _run_key(parent_run_id)


def _has_operation_id(metadata: dict[str, object]) -> bool:
    operation = metadata.get("operation_id")
    return isinstance(operation, str) and bool(operation)


def _clear_failed_parent_operation(
    tracker: OrderedDict[str, str], parent_run_id: str | None, operation_id: str
) -> None:
    if parent_run_id and tracker.get(parent_run_id) == operation_id:
        tracker.pop(parent_run_id, None)


def _remember_attempt(tracker: OrderedDict[str, int], operation_id: str, attempt: int) -> None:
    prior = tracker.get(operation_id, 0)
    tracker[operation_id] = max(prior, attempt)
    tracker.move_to_end(operation_id)
    _trim_tracker(tracker)


def _retry_signaled(
    metadata: dict[str, object],
    callback_kwargs: dict[str, Any],
    tags: list[str] | None,
    explicit_attempt: int,
) -> bool:
    """Return true only for explicit, meaningful retry callback data."""

    if explicit_attempt > 1:
        return True
    for source in (metadata, callback_kwargs):
        attempted_retries = source.get("attempted_retries")
        if (
            isinstance(attempted_retries, int)
            and not isinstance(attempted_retries, bool)
            and attempted_retries > 0
        ):
            return True
        if source.get("retry") is True or source.get("retrying") is True:
            return True
    return any(isinstance(tag, str) and tag.strip().lower() == "retry" for tag in tags or ())


def _trim_tracker(tracker: OrderedDict[str, object]) -> None:
    while len(tracker) > _TRACKER_CAPACITY:
        tracker.popitem(last=False)


def _value(value: object, key: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(key, default)
    try:
        return getattr(value, key)
    except Exception:
        return default


def _llm_output(response: object) -> object:
    if isinstance(response, dict):
        return response.get("llm_output") or {}
    try:
        return getattr(response, "llm_output") or {}
    except Exception:
        return {}


def _is_cancelled(error: BaseException) -> bool:
    return (
        isinstance(error, (asyncio.CancelledError,)) or error.__class__.__name__ == "CancelledError"
    )


_FACTORY_LOCK = threading.RLock()
_DEFAULT_FACTORY: Any = None


class _Registration:
    def __init__(self, previous: Any, installed: Any) -> None:
        self.previous = previous
        self.installed = installed
        self._closed = False

    def __call__(self) -> None:
        self.uninstall()

    def uninstall(self) -> None:
        global _DEFAULT_FACTORY
        with _FACTORY_LOCK:
            if self._closed:
                return
            self._closed = True
            if _DEFAULT_FACTORY is self.installed:
                _DEFAULT_FACTORY = self.previous


def register(monitor: UsageMonitor, config: HyperionConfig) -> _Registration:
    """Configure (but do not globally patch) the default LangChain handler factory."""

    global _DEFAULT_FACTORY
    with _FACTORY_LOCK:
        if _DEFAULT_FACTORY is not None:
            return _Registration(_DEFAULT_FACTORY, _DEFAULT_FACTORY)
        previous = _DEFAULT_FACTORY
        installed = lambda: LangChainUsageCallback(monitor=monitor, config=config)
        _DEFAULT_FACTORY = installed
        return _Registration(previous, installed)


def default_handler() -> LangChainUsageCallback | None:
    """Create the registered default handler, if registration is active."""

    with _FACTORY_LOCK:
        factory = _DEFAULT_FACTORY
    if factory is None:
        return None
    try:
        return factory()
    except Exception:
        return None


LangChainTokenMonitor = LangChainUsageCallback
HyperionLangChainCallback = LangChainUsageCallback
LangChainCallbackHandler = LangChainUsageCallback


def _extract_provider(model: object) -> str:
    """Compatibility helper exposing the reference provider heuristic."""

    from ._common import infer_provider

    return infer_provider(model)[1]


__all__ = [
    "HyperionLangChainCallback",
    "LangChainCallbackHandler",
    "LangChainTokenMonitor",
    "LangChainUsageCallback",
    "default_handler",
    "register",
]

register_langchain_callback = register
