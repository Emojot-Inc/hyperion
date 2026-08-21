"""Reference snapshot of the initial LLM token-usage monitor.

This module preserves the original callback and record behavior as a characterization
baseline. It is intentionally isolated from Hyperion's future public API and is not
included in the built package.
"""

from __future__ import annotations

import contextvars
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import litellm
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from reference.config import settings as config

logger = logging.getLogger("token_monitor")
logger.setLevel(logging.DEBUG)


@dataclass
class TokenUsageRecord:
    """A single structured record of token usage for one LLM call."""

    call_site: str = ""
    model_name: str = ""
    model_provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    status: str = "success"
    error: Optional[str] = None
    request_id: str = ""
    session_id: str = ""
    timestamp: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _is_enabled() -> bool:
    return getattr(config, "ENABLE_TOKEN_MONITORING", False)


def _extract_provider(model_name: str) -> str:
    """Best-effort extraction of the provider from a model string."""
    if "/" in model_name:
        return model_name.split("/", 1)[0]

    lower = model_name.lower()
    if "gpt" in lower or "o1" in lower or "o3" in lower:
        return "openai"
    if "claude" in lower:
        return "anthropic"
    if "gemini" in lower or "models/" in lower:
        return "google"
    return "unknown"


def log_token_usage(record: TokenUsageRecord) -> None:
    """Emit one usage record without allowing monitoring failures to escape."""
    if not _is_enabled():
        return
    try:
        logger.info("llm_token_usage | %s", record.to_dict())
    except Exception as exc:  # pragma: no cover - defensive safety net
        logger.warning("token_monitor: failed to log usage record: %s", exc)


def persist_token_usage(record: TokenUsageRecord, mongo_client=None, company_id: str = "") -> None:
    """Optionally persist a usage record to a tenant-scoped Mongo collection."""
    if not _is_enabled() or not mongo_client or not company_id:
        return
    try:
        logger.debug("Persisting token usage record for company_id=%s", company_id)
        db = mongo_client[f"tenant_{company_id}"]
        db["TokenUsageLogs"].insert_one(record.to_dict())
    except Exception as exc:
        logger.warning("token_monitor: failed to persist usage record: %s", exc)


_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("_tm_request_id", default="")
_company_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("_tm_company_id", default="")
_session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("_tm_session_id", default="")


def set_request_context(
    request_id: str = "",
    company_id: str = "",
    session_id: str = "",
) -> None:
    """Set request metadata used by the LiteLLM callback."""
    _request_id_var.set(request_id)
    _company_id_var.set(company_id)
    _session_id_var.set(session_id)


def clear_request_context() -> None:
    """Clear request metadata after an LLM execution."""
    _request_id_var.set("")
    _company_id_var.set("")
    _session_id_var.set("")


def _get_request_context() -> Dict[str, str]:
    return {
        "request_id": _request_id_var.get(""),
        "company_id": _company_id_var.get(""),
        "session_id": _session_id_var.get(""),
    }


class LiteLLMTokenMonitor(litellm.integrations.custom_logger.CustomLogger):
    """LiteLLM callback that records completion success and failure events."""

    def __init__(self, mongo_client=None):
        self._mongo_client = mongo_client

    def log_success_event(self, kwargs: dict, response_obj, start_time, end_time):
        if not _is_enabled():
            return
        try:
            self._record(kwargs, response_obj, start_time, end_time, status="success")
        except Exception as exc:
            logger.warning("token_monitor(litellm): error in log_success_event: %s", exc)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs: dict, response_obj, start_time, end_time):
        if not _is_enabled():
            return
        try:
            self._record(kwargs, response_obj, start_time, end_time, status="error")
        except Exception as exc:
            logger.warning("token_monitor(litellm): error in log_failure_event: %s", exc)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.log_failure_event(kwargs, response_obj, start_time, end_time)

    def _record(self, kwargs, response_obj, start_time, end_time, status: str):
        model = kwargs.get("model", "") or ""
        usage = {}

        if response_obj and hasattr(response_obj, "usage") and response_obj.usage:
            usage_obj = response_obj.usage
            if hasattr(usage_obj, "prompt_tokens"):
                usage = {
                    "input_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
                }
            elif isinstance(usage_obj, dict):
                usage = {
                    "input_tokens": usage_obj.get("prompt_tokens", 0) or 0,
                    "output_tokens": usage_obj.get("completion_tokens", 0) or 0,
                    "total_tokens": usage_obj.get("total_tokens", 0) or 0,
                }

        duration_ms = 0.0
        if start_time and end_time:
            try:
                duration_ms = (end_time - start_time).total_seconds() * 1000
            except Exception:
                pass

        litellm_params = kwargs.get("litellm_params", {}) or {}
        metadata = litellm_params.get("metadata", {}) or {}
        proxy_metadata = kwargs.get("metadata", {}) or {}
        merged_meta = {**metadata, **proxy_metadata}

        request_context = _get_request_context()
        for key in ("request_id", "company_id", "session_id"):
            if not merged_meta.get(key):
                merged_meta[key] = request_context.get(key, "")

        error_str = None
        if status == "error":
            exc = kwargs.get("exception") or kwargs.get("original_exception")
            error_str = str(exc) if exc else "unknown error"

        record = TokenUsageRecord(
            call_site=merged_meta.get("call_site", "adk_agent"),
            model_name=model,
            model_provider=_extract_provider(model),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            duration_ms=duration_ms,
            status=status,
            error=error_str,
            request_id=merged_meta.get("request_id", ""),
            session_id=merged_meta.get("session_id", ""),
        )

        log_token_usage(record)
        persist_token_usage(record, self._mongo_client, company_id=merged_meta.get("company_id", ""))


class LangChainTokenMonitor(BaseCallbackHandler):
    """LangChain callback handler for ``llm.invoke()`` usage records."""

    def __init__(
        self,
        call_site: str = "",
        request_id: str = "",
        company_id: str = "",
        session_id: str = "",
        mongo_client=None,
    ):
        super().__init__()
        self.call_site = call_site
        self.request_id = request_id
        self.company_id = company_id
        self.session_id = session_id
        self._mongo_client = mongo_client
        self._start_time: Optional[float] = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        if not _is_enabled():
            return
        self._start_time = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs):
        if not _is_enabled():
            return
        try:
            duration_ms = 0.0
            if self._start_time:
                duration_ms = (time.time() - self._start_time) * 1000

            llm_output = response.llm_output or {}
            token_usage = llm_output.get("token_usage", {})
            model_name = llm_output.get("model_name", "") or ""

            record = TokenUsageRecord(
                call_site=self.call_site,
                model_name=model_name,
                model_provider=_extract_provider(model_name),
                input_tokens=token_usage.get("prompt_tokens", 0) or 0,
                output_tokens=token_usage.get("completion_tokens", 0) or 0,
                total_tokens=token_usage.get("total_tokens", 0) or 0,
                duration_ms=duration_ms,
                status="success",
                request_id=self.request_id,
                session_id=self.session_id,
            )

            log_token_usage(record)
            persist_token_usage(record, self._mongo_client, company_id=self.company_id)
        except Exception as exc:
            logger.warning("token_monitor(langchain): error in on_llm_end: %s", exc)

    def on_llm_error(self, error: BaseException, **kwargs):
        if not _is_enabled():
            return
        try:
            duration_ms = 0.0
            if self._start_time:
                duration_ms = (time.time() - self._start_time) * 1000

            record = TokenUsageRecord(
                call_site=self.call_site,
                model_name="",
                model_provider="",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                duration_ms=duration_ms,
                status="error",
                error=str(error),
                request_id=self.request_id,
                session_id=self.session_id,
            )

            log_token_usage(record)
            persist_token_usage(record, self._mongo_client, company_id=self.company_id)
        except Exception as exc:
            logger.warning("token_monitor(langchain): error in on_llm_error: %s", exc)


def register_litellm_callback(mongo_client=None) -> None:
    """Register the LiteLLM token-monitor callback once."""
    if not _is_enabled():
        logger.info("Token monitoring is DISABLED. Skipping LiteLLM callback registration.")
        return

    for callback in litellm.callbacks:
        if isinstance(callback, LiteLLMTokenMonitor):
            logger.debug("LiteLLM token-monitor callback already registered.")
            return

    litellm.callbacks.append(LiteLLMTokenMonitor(mongo_client=mongo_client))
    logger.info("Token monitoring is ENABLED. LiteLLM callback registered.")
