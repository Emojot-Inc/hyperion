"""Optional LiteLLM callback integration.

Importing this module is the explicit opt-in boundary for LiteLLM.  The callback is
deliberately limited to final success/failure hooks; chunks are not accounting events.
"""

# Callback boundaries are deliberately fail-open; no provider callback failure may
# replace an application result or exception.
# ruff: noqa: BLE001, S110

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger

from hyperion.config import HyperionConfig
from hyperion.monitor import UsageMonitor

from ._common import (
    callback_times,
    emit_event,
    merged_metadata,
    operation_and_attempt,
    response_usage,
    safe_error,
)


def _config_or_default(config: HyperionConfig | None) -> HyperionConfig:
    if config is not None:
        return config
    return HyperionConfig(enabled=True, integrations=(), sinks=())


class LiteLLMUsageCallback(CustomLogger):
    """CustomLogger-compatible, failure-isolated LiteLLM usage callback."""

    def __init__(
        self,
        monitor: UsageMonitor | None = None,
        config: HyperionConfig | None = None,
        **_: Any,
    ) -> None:
        if isinstance(monitor, HyperionConfig) and config is None:
            config, monitor = monitor, None
        try:
            super().__init__()
        except Exception:
            # A provider's callback base class is not part of Hyperion's contract.
            pass
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

    def log_success_event(
        self,
        kwargs: dict[str, Any] | None,
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        self._record(kwargs, response_obj, start_time, end_time, status="success")

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any] | None,
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    def log_failure_event(
        self,
        kwargs: dict[str, Any] | None,
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        self._record(kwargs, response_obj, start_time, end_time, status="failure")

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any] | None,
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        self.log_failure_event(kwargs, response_obj, start_time, end_time)

    def _record(
        self,
        kwargs: dict[str, Any] | None,
        response_obj: object,
        start_time: object,
        end_time: object,
        *,
        status: str,
    ) -> None:
        try:
            values = kwargs if isinstance(kwargs, Mapping) else {}
            litellm_params = values.get("litellm_params")
            nested = litellm_params.get("metadata") if isinstance(litellm_params, Mapping) else None
            explicit = values.get("metadata")
            metadata = merged_metadata(nested, explicit)

            # LiteLLM's stable call identifier may be exposed directly or under
            # litellm_params.  A request_id is an ordinary request identity and
            # must never be promoted to retry correlation.
            if not isinstance(metadata.get("operation_id"), str) or not metadata.get("operation_id"):
                call_id = metadata.get("litellm_call_id")
                if not isinstance(call_id, str) or not call_id:
                    call_id = values.get("litellm_call_id")
                if (not isinstance(call_id, str) or not call_id) and isinstance(litellm_params, Mapping):
                    call_id = litellm_params.get("litellm_call_id")
                if isinstance(call_id, str) and call_id:
                    metadata["litellm_call_id"] = call_id
                    metadata["operation_id"] = call_id
            # Retain compatibility with provider callback IDs used by older
            # LiteLLM versions, while keeping request IDs out of this path.
            if not metadata.get("operation_id"):
                for key in ("callback_id", "call_id", "completion_id", "id"):
                    identifier = values.get(key)
                    if isinstance(identifier, str) and identifier:
                        metadata["operation_id"] = identifier
                        break

            explicit_attempt = metadata.get("attempt", values.get("attempt"))
            has_explicit_attempt = (
                isinstance(explicit_attempt, int)
                and not isinstance(explicit_attempt, bool)
                and explicit_attempt >= 1
            )
            if has_explicit_attempt:
                metadata["attempt"] = explicit_attempt
            else:
                attempted_retries = metadata.get(
                    "attempted_retries", values.get("attempted_retries")
                )
                if isinstance(litellm_params, Mapping) and attempted_retries is None:
                    attempted_retries = litellm_params.get("attempted_retries")
                if (
                    isinstance(attempted_retries, int)
                    and not isinstance(attempted_retries, bool)
                    and attempted_retries >= 0
                ):
                    metadata["attempt"] = attempted_retries + 1
            operation_id, attempt = operation_and_attempt(
                metadata,
                attempt=values.get("attempt", values.get("retry_count")),
            )
            start, end, duration = callback_times(start_time, end_time)
            usage = response_usage(response_obj)
            error_value = None
            error = None
            if status == "failure":
                error_value = values.get("exception") or values.get("original_exception")
                error = safe_error(error_value)
            stream = bool(values.get("stream", values.get("streaming", False)))
            if status == "success":
                stream_state = "final" if stream else "not_streaming"
            elif _is_cancelled(error_value):
                stream_state = "cancelled"
            else:
                stream_state = "failed"
            emit_event(
                self.monitor,
                self.config,
                model=values.get("model", ""),
                metadata=metadata,
                start_time=start,
                end_time=end,
                duration_ms=duration,
                status=status,
                error=error,
                stream_state=stream_state,
                operation_id=operation_id,
                attempt=attempt,
                usage=usage,
                details_namespace="litellm",
            )
        except Exception:
            return


def _is_cancelled(value: object) -> bool:
    return isinstance(value, BaseException) and value.__class__.__name__ in {
        "CancelledError",
        "KeyboardInterrupt",
    }


class _Registration:
    def __init__(self, callback: LiteLLMUsageCallback | None) -> None:
        self.callback = callback
        self._closed = False

    def __call__(self) -> None:
        self.uninstall()

    def uninstall(self) -> None:
        if self._closed:
            return
        self._closed = True
        callback = self.callback
        if callback is None:
            return
        try:
            callbacks = getattr(litellm, "callbacks", None)
            if isinstance(callbacks, list):
                for index, item in enumerate(callbacks):
                    if item is callback:
                        del callbacks[index]
                        break
        except Exception:
            return


def register(monitor: UsageMonitor, config: HyperionConfig) -> _Registration:
    """Install one callback, scanning LiteLLM's global callbacks idempotently."""

    try:
        callbacks = getattr(litellm, "callbacks", None)
        if not isinstance(callbacks, list):
            return _Registration(None)
        for callback in callbacks:
            if isinstance(callback, LiteLLMUsageCallback):
                return _Registration(None)
        callback = LiteLLMUsageCallback(monitor=monitor, config=config)
        callbacks.append(callback)
        return _Registration(callback)
    except Exception:
        return _Registration(None)


# Names used by the extracted implementation and by early Hyperion adopters.
LiteLLMTokenMonitor = LiteLLMUsageCallback
HyperionLiteLLMCallback = LiteLLMUsageCallback
LiteLLMCallback = LiteLLMUsageCallback


def _extract_provider(model: object) -> str:
    """Compatibility helper exposing the reference provider heuristic."""

    from ._common import infer_provider

    return infer_provider(model)[1]


__all__ = [
    "HyperionLiteLLMCallback",
    "LiteLLMCallback",
    "LiteLLMTokenMonitor",
    "LiteLLMUsageCallback",
    "register",
]

register_litellm_callback = register
