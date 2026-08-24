"""Deterministic adapter contract tests; no live providers are used."""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta, timezone

import pytest

litellm = pytest.importorskip("litellm")
pytest.importorskip("langchain_core")

from langchain_core.outputs import LLMResult

from hyperion import HyperionConfig, UsageMonitor, context_scope
from hyperion.adapters._common import infer_provider
from hyperion.adapters.langchain import LangChainUsageCallback, default_handler
from hyperion.adapters.langchain import register as register_langchain
from hyperion.adapters.litellm import LiteLLMUsageCallback
from hyperion.adapters.litellm import register as register_litellm
from hyperion.sinks.memory import MemorySink


class Usage:
    prompt_tokens = 4
    completion_tokens = 6
    total_tokens = 10


class Response:
    usage = Usage()


def setup_monitor(**kwargs):
    sink = MemorySink()
    config = HyperionConfig(enabled=True, integrations=(), sinks=(sink,), **kwargs)
    return sink, config, UsageMonitor(config.sinks, enabled=True)


def test_litellm_success_object_usage_context_precedence_and_privacy():
    sink, config, monitor = setup_monitor(default_call_site="application")
    callback = LiteLLMUsageCallback(monitor, config)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with context_scope(
        request_id="context-request",
        session_id="context-session",
        metadata={"company_id": "tenant", "call_site": "context-site"},
    ):
        callback.log_success_event(
            {
                "model": "openai/gpt-5-mini",
                "litellm_params": {"metadata": {"request_id": "nested-request", "nested": "ok"}},
                "metadata": {"request_id": "explicit-request", "messages": ["secret"]},
            },
            Response(),
            start,
            start + timedelta(milliseconds=125),
        )
    event = sink.snapshot()[0]
    assert (event.provider, event.input_tokens, event.output_tokens, event.total_tokens) == (
        "openai",
        4,
        6,
        10,
    )
    assert event.request_id == "explicit-request"
    assert event.session_id == "context-session"
    assert event.call_site == "context-site"
    assert event.details["litellm"]["company_id"] == "tenant"
    assert "messages" not in event.details["litellm"]
    assert "request_id" not in event.details["litellm"]
    assert "session_id" not in event.details["litellm"]
    assert "call_site" not in event.details["litellm"]


def test_litellm_blank_metadata_preserves_context_and_configured_call_site():
    sink, config, monitor = setup_monitor(default_call_site="configured")
    callback = LiteLLMUsageCallback(monitor, config)
    with context_scope(request_id="ctx-request", session_id="ctx-session"):
        callback.log_success_event(
            {
                "model": "gpt-5-mini",
                "metadata": {"request_id": "", "session_id": None, "call_site": ""},
            },
            Response(),
            None,
            None,
        )
    event = sink.snapshot()[0]
    assert event.request_id == "ctx-request"
    assert event.session_id == "ctx-session"
    assert event.call_site == "configured"


@pytest.mark.parametrize("error_key", ["exception", "original_exception"])
def test_litellm_missing_usage_and_failure_are_nullable_and_bounded(error_key):
    sink, config, monitor = setup_monitor(details_max_bytes=10)
    callback = LiteLLMUsageCallback(monitor, config)
    callback.log_failure_event(
        {
            "model": "custom-model",
            error_key: RuntimeError("x" * 5_000),
            "metadata": {"key": "value"},
        },
        None,
        None,
        None,
    )
    event = sink.snapshot()[0]
    assert event.status.value == "failure"
    assert len(event.error or "") == event.MAX_ERROR_LENGTH
    assert event.error and event.error.endswith("...[truncated]")
    assert not event.usage_available
    assert event.input_tokens is event.output_tokens is event.total_tokens is None
    assert event.details_truncated


def test_litellm_retry_metadata_and_inconsistent_provider_usage_are_preserved():
    sink, config, monitor = setup_monitor()
    callback = LiteLLMUsageCallback(monitor, config)
    response = {"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 99}}
    for retries in (0, 1):
        callback.log_success_event(
            {
                "model": "gpt-5-mini",
                "litellm_params": {
                    "litellm_call_id": "stable-call",
                    "metadata": {"attempted_retries": retries},
                },
            },
            response,
            None,
            None,
        )
    events = sink.snapshot()
    assert [event.operation_id for event in events] == ["stable-call", "stable-call"]
    assert [event.attempt for event in events] == [1, 2]
    assert [(event.input_tokens, event.output_tokens, event.total_tokens) for event in events] == [
        (2, 3, 99),
        (2, 3, 99),
    ]
    assert events[0].event_id != events[1].event_id


def test_litellm_top_level_call_id_and_retry_metadata_require_no_tracker():
    sink, config, monitor = setup_monitor()
    callback = LiteLLMUsageCallback(monitor, config)
    response = {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    callback.log_success_event(
        {
            "model": "gpt-5-mini",
            "litellm_call_id": "top-level-call",
            "attempt": 1,
        },
        response,
        None,
        None,
    )
    callback.log_success_event(
        {
            "model": "gpt-5-mini",
            "litellm_params": {
                "litellm_call_id": "top-level-call",
                "attempted_retries": 1,
            },
        },
        response,
        None,
        None,
    )

    events = sink.snapshot()
    assert [event.operation_id for event in events] == ["top-level-call", "top-level-call"]
    assert [event.attempt for event in events] == [1, 2]
    assert not hasattr(callback, "_attempts")


def test_litellm_stream_failure_and_cancellation_are_terminal_states():
    sink, config, monitor = setup_monitor()
    callback = LiteLLMUsageCallback(monitor, config)
    callback.log_failure_event(
        {"model": "gpt", "stream": True, "exception": RuntimeError("failed")},
        None,
        None,
        None,
    )
    callback.log_failure_event(
        {"model": "gpt", "stream": True, "exception": asyncio.CancelledError()},
        None,
        None,
        None,
    )
    assert [event.stream_state for event in sink.snapshot()] == ["failed", "cancelled"]


def test_litellm_async_callbacks_and_retry_operation():
    sink, config, monitor = setup_monitor()
    callback = LiteLLMUsageCallback(monitor, config)
    for attempt in (1, 2):
        asyncio.run(
            callback.async_log_success_event(
                {"model": "claude-3", "metadata": {"operation_id": "op", "attempt": attempt}},
                {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
                None,
                None,
            )
        )
    events = sink.snapshot()
    assert [event.operation_id for event in events] == ["op", "op"]
    assert [event.attempt for event in events] == [1, 2]
    assert all(event.usage_available for event in events)
    assert events[0].event_id != events[1].event_id


def test_litellm_async_failure_cancellation_is_terminal_and_fail_open():
    sink, config, monitor = setup_monitor()
    callback = LiteLLMUsageCallback(monitor, config)

    asyncio.run(
        callback.async_log_failure_event(
            {
                "model": "gpt-5-mini",
                "litellm_call_id": "cancelled-call",
                "exception": asyncio.CancelledError(),
            },
            None,
            None,
            None,
        )
    )

    events = sink.snapshot()
    assert len(events) == 1
    assert events[0].status.value == "failure"
    assert events[0].stream_state == "cancelled"
    assert events[0].usage_available is False
    assert events[0].input_tokens is None


def test_litellm_provider_override_and_registration_teardown():
    sink, config, monitor = setup_monitor(provider_resolver=lambda model: "vendor")
    callback = LiteLLMUsageCallback(monitor, config)
    callback.log_success_event({"model": "gemini-2.0"}, Response(), None, None)
    assert sink.snapshot()[0].provider == "vendor"
    previous = list(litellm.callbacks)
    litellm.callbacks[:] = []
    try:
        first = register_litellm(monitor, config)
        second = register_litellm(monitor, config)
        assert len(litellm.callbacks) == 1
        second.uninstall()
        assert len(litellm.callbacks) == 1
        first.uninstall()
        assert litellm.callbacks == []
    finally:
        litellm.callbacks[:] = previous


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("openai/gpt-5-mini", "openai"),
        ("gpt-5-mini", "openai"),
        ("o1-mini", "openai"),
        ("o3-mini", "openai"),
        ("claude-3-7-sonnet", "anthropic"),
        ("gemini-2.0-flash", "google"),
        ("custom-model", "unknown"),
    ],
)
def test_provider_inference_uses_prefix_and_documented_model_families(model, provider):
    assert infer_provider(model) == (model, provider)


def test_langchain_provider_override_is_used():
    sink, config, monitor = setup_monitor(provider_resolver=lambda model: "vendor")
    callback = LangChainUsageCallback(monitor, config)
    callback.on_llm_start({}, [], run_id="override")
    callback.on_llm_end(
        LLMResult(generations=[], llm_output={"model_name": "gemini-2.0", "token_usage": {}}),
        run_id="override",
    )

    assert sink.snapshot()[0].provider == "vendor"


def test_litellm_successful_stream_emits_one_final_event_without_chunks():
    sink, config, monitor = setup_monitor()
    callback = LiteLLMUsageCallback(monitor, config)
    callback.log_success_event(
        {"model": "gpt-5-mini", "stream": True},
        {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}},
        None,
        None,
    )

    events = sink.snapshot()
    assert len(events) == 1
    assert events[0].stream_state == "final"
    assert not hasattr(events[0], "stream_sequence")
    assert not hasattr(events[0], "stream_id")


def test_litellm_malformed_success_usage_is_unavailable():
    sink, config, monitor = setup_monitor()
    callback = LiteLLMUsageCallback(monitor, config)
    callback.log_success_event(
        {"model": "gpt-5-mini"},
        {"usage": {"prompt_tokens": "bad", "completion_tokens": 2, "total_tokens": 3}},
        None,
        None,
    )

    event = sink.snapshot()[0]
    assert not event.usage_available
    assert event.input_tokens is None
    assert event.output_tokens is None
    assert event.total_tokens is None


def test_langchain_success_tracks_run_duration_and_stream_final(monkeypatch):
    sink, config, monitor = setup_monitor()
    callback = LangChainUsageCallback(monitor, config, call_site="routing")
    clock = iter((10.0, 11.5))
    monkeypatch.setattr("hyperion.adapters.langchain.time.time", lambda: next(clock))
    callback.on_llm_start({}, [], run_id="run", streaming=True)
    callback.on_llm_new_token("secret", run_id="run")
    callback.on_llm_end(
        LLMResult(
            generations=[],
            llm_output={
                "model_name": "gpt-5-mini",
                "token_usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            },
        ),
        run_id="run",
    )
    event = sink.snapshot()[0]
    assert event.duration_ms == 1500.0
    assert event.stream_state == "final"
    assert event.call_site == "routing"


def test_langchain_error_and_malformed_payload_are_isolated(monkeypatch):
    sink, config, monitor = setup_monitor()
    callback = LangChainUsageCallback(monitor, config)
    clock = iter((10.0, 11.0, 12.0, 13.0))
    monkeypatch.setattr("hyperion.adapters.langchain.time.time", lambda: next(clock))
    callback.on_llm_start({}, [], run_id="error")
    callback.on_llm_error(RuntimeError("failed"), run_id="error")
    assert "error" not in callback._runs
    callback.on_llm_start({}, [], run_id="bad")
    callback.on_llm_end(
        {"llm_output": {"model_name": "bad", "token_usage": {"prompt_tokens": "no"}}}, run_id="bad"
    )
    events = sink.snapshot()
    assert events[0].status.value == "failure"
    assert events[0].stream_state == "failed"
    assert events[0].usage_available is False
    assert events[1].usage_available is False
    assert events[1].input_tokens is None


def test_langchain_retry_attempts_share_operation_but_have_distinct_events():
    sink, config, monitor = setup_monitor()
    callback = LangChainUsageCallback(monitor, config)
    callback.on_llm_start({}, [], run_id="failed-attempt", metadata={"operation_id": "logical-run"})
    callback.on_llm_error(RuntimeError("temporary"), run_id="failed-attempt")
    callback.on_llm_start(
        {}, [], run_id="successful-attempt", metadata={"operation_id": "logical-run"}
    )
    callback.on_llm_end(
        LLMResult(
            generations=[],
            llm_output={
                "model_name": "gpt-5-mini",
                "token_usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
        ),
        run_id="successful-attempt",
    )

    events = sink.snapshot()
    assert [event.operation_id for event in events] == ["logical-run", "logical-run"]
    assert [event.attempt for event in events] == [1, 2]
    assert [event.usage_available for event in events] == [False, True]
    assert events[0].event_id != events[1].event_id


def test_langchain_parent_retry_with_signal_reuses_failed_operation():
    sink, config, monitor = setup_monitor()
    callback = LangChainUsageCallback(monitor, config)
    callback.on_llm_start({}, [], run_id="failed-attempt", parent_run_id="parent")
    callback.on_llm_error(RuntimeError("temporary"), run_id="failed-attempt")
    callback.on_llm_start(
        {}, [], run_id="successful-attempt", parent_run_id="parent", metadata={"retrying": True}
    )
    callback.on_llm_end(
        LLMResult(
            generations=[],
            llm_output={
                "model_name": "gpt-5-mini",
                "token_usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
        ),
        run_id="successful-attempt",
    )

    events = sink.snapshot()
    assert [event.operation_id for event in events] == ["failed-attempt", "failed-attempt"]
    assert [event.attempt for event in events] == [1, 2]
    assert events[0].event_id != events[1].event_id


def test_langchain_failed_child_then_unrelated_metadata_free_sibling_is_distinct():
    sink, config, monitor = setup_monitor()
    callback = LangChainUsageCallback(monitor, config)
    callback.on_llm_start({}, [], run_id="failed-child", parent_run_id="parent")
    callback.on_llm_error(RuntimeError("temporary"), run_id="failed-child")
    callback.on_llm_start({}, [], run_id="unrelated-sibling", parent_run_id="parent")
    callback.on_llm_end(
        LLMResult(generations=[], llm_output={"model_name": "gpt-5-mini"}),
        run_id="unrelated-sibling",
    )

    events = sink.snapshot()
    assert [event.operation_id for event in events] == ["failed-child", "unrelated-sibling"]
    assert [event.attempt for event in events] == [1, 1]


def test_langchain_explicit_operation_sibling_clears_stale_parent_handoff():
    sink, config, monitor = setup_monitor()
    callback = LangChainUsageCallback(monitor, config)
    callback.on_llm_start({}, [], run_id="failed-child", parent_run_id="parent")
    callback.on_llm_error(RuntimeError("temporary"), run_id="failed-child")
    callback.on_llm_start(
        {},
        [],
        run_id="explicit-sibling",
        parent_run_id="parent",
        metadata={"operation_id": "explicit-operation"},
    )
    callback.on_llm_end(
        LLMResult(generations=[], llm_output={"model_name": "gpt-5-mini"}),
        run_id="explicit-sibling",
    )
    callback.on_llm_start({}, [], run_id="later-sibling", parent_run_id="parent")
    callback.on_llm_end(
        LLMResult(generations=[], llm_output={"model_name": "gpt-5-mini"}),
        run_id="later-sibling",
    )

    events = sink.snapshot()
    assert [event.operation_id for event in events] == [
        "failed-child",
        "explicit-operation",
        "later-sibling",
    ]
    assert [event.attempt for event in events] == [1, 1, 1]


def test_langchain_successful_siblings_with_parent_remain_separate_operations():
    sink, config, monitor = setup_monitor()
    callback = LangChainUsageCallback(monitor, config)
    for run_id in ("first-sibling", "second-sibling"):
        callback.on_llm_start({}, [], run_id=run_id, parent_run_id="parent")
        callback.on_llm_end(
            LLMResult(generations=[], llm_output={"model_name": "gpt-5-mini"}), run_id=run_id
        )

    events = sink.snapshot()
    assert len(events) == 2
    assert events[0].operation_id != events[1].operation_id
    assert [event.attempt for event in events] == [1, 1]


def test_langchain_manager_run_ids_do_not_conflate_ordinary_calls():
    from langchain_core.callbacks.manager import CallbackManager

    sink, config, monitor = setup_monitor()
    callback = LangChainUsageCallback(monitor, config)
    manager = CallbackManager.configure([callback])
    runs = manager.on_llm_start({}, ["first", "second"])
    for run in runs:
        run.on_llm_end(LLMResult(generations=[], llm_output={"model_name": "gpt-5-mini"}))

    events = sink.snapshot()
    assert len(events) == 2
    assert events[0].operation_id != events[1].operation_id
    assert [event.attempt for event in events] == [1, 1]


def test_langchain_metadata_blanks_at_start_and_end_preserve_context_identity():
    sink, config, monitor = setup_monitor()
    callback = LangChainUsageCallback(monitor, config)
    with context_scope(request_id="context-request", session_id="context-session"):
        callback.on_llm_start(
            {}, [], run_id="metadata-merge", metadata={"request_id": "", "session_id": None}
        )
        callback.on_llm_end(
            LLMResult(generations=[], llm_output={"model_name": "gpt-5-mini"}),
            run_id="metadata-merge",
            metadata={"request_id": "", "session_id": None},
        )

    event = sink.snapshot()[0]
    assert event.request_id == "context-request"
    assert event.session_id == "context-session"


def test_langchain_details_and_failure_error_are_bounded():
    sink, config, monitor = setup_monitor(details_max_bytes=10)
    callback = LangChainUsageCallback(monitor, config)
    callback.on_llm_start({}, [], run_id="bounded", metadata={"provider_metadata": "x" * 100})
    callback.on_llm_error(RuntimeError("provider-error-" + "x" * 10_000), run_id="bounded")

    event = sink.snapshot()[0]
    assert event.details_truncated
    assert len(event.error or "") <= event.MAX_ERROR_LENGTH
    assert event.error and event.error.endswith("...[truncated]")


def test_langchain_stream_cancellation_and_recursive_detail_privacy():
    sink, config, monitor = setup_monitor()
    callback = LangChainUsageCallback(monitor, config, metadata={"company_id": "tenant"})
    callback.on_llm_start(
        {},
        ["secret prompt"],
        run_id="cancel",
        streaming=True,
        metadata={
            "request_id": "req",
            "nested": {"content": "secret", "safe": False, "operation_id": "private"},
        },
    )
    callback.on_llm_error(asyncio.CancelledError(), run_id="cancel")
    assert "cancel" not in callback._runs
    event = sink.snapshot()[0]
    assert event.stream_state == "cancelled"
    assert event.details["langchain"]["company_id"] == "tenant"
    assert "request_id" not in event.details["langchain"]
    assert "content" not in event.details["langchain"]["nested"]
    assert "operation_id" not in event.details["langchain"]["nested"]
    assert event.details["langchain"]["nested"]["safe"] is False


def test_langchain_active_run_state_survives_tracker_capacity(monkeypatch):
    sink, config, monitor = setup_monitor()
    callback = LangChainUsageCallback(monitor, config)
    monkeypatch.setattr("hyperion.adapters.langchain.time.time", lambda: 10.0)
    callback.on_llm_start(
        {},
        [],
        run_id="oldest",
        metadata={"operation_id": "oldest-operation", "request_id": "oldest-request"},
    )
    for index in range(1_025):
        callback.on_llm_start(
            {}, [], run_id=f"run-{index}", metadata={"operation_id": f"operation-{index}"}
        )
    assert len(callback._runs) == 1_026

    monkeypatch.setattr("hyperion.adapters.langchain.time.time", lambda: 11.5)
    callback.on_llm_end(
        LLMResult(generations=[], llm_output={"model_name": "gpt-5-mini"}), run_id="oldest"
    )
    event = sink.snapshot()[0]
    assert event.operation_id == "oldest-operation"
    assert event.request_id == "oldest-request"
    assert event.start_time == 10.0
    assert event.duration_ms == 1500.0
    assert "oldest" not in callback._runs


def test_langchain_retry_bookkeeping_stays_bounded_and_success_retires_attempt_state():
    _sink, config, monitor = setup_monitor()
    callback = LangChainUsageCallback(monitor, config)
    for index in range(2_000):
        callback.on_llm_start(
            {}, [], run_id=f"run-{index}", metadata={"operation_id": f"operation-{index}"}
        )
    assert len(callback._runs) == 2_000
    assert len(callback._operation_attempts) <= 1024

    callback.on_llm_start({}, [], run_id="successful", metadata={"operation_id": "done"})
    callback.on_llm_end(
        LLMResult(generations=[], llm_output={"model_name": "gpt-5-mini"}),
        run_id="successful",
    )
    assert "done" not in callback._operation_attempts


def test_langchain_failure_restores_evicted_attempt_state_before_retry():
    sink, config, monitor = setup_monitor()
    callback = LangChainUsageCallback(monitor, config)
    callback.on_llm_start(
        {},
        [],
        run_id="oldest",
        parent_run_id="parent",
        metadata={"operation_id": "oldest-operation"},
    )
    for index in range(1_024):
        callback.on_llm_start(
            {}, [], run_id=f"run-{index}", metadata={"operation_id": f"operation-{index}"}
        )
    assert "oldest-operation" not in callback._operation_attempts

    callback.on_llm_error(RuntimeError("temporary"), run_id="oldest")
    assert callback._operation_attempts["oldest-operation"] == 1
    callback.on_llm_start({}, [], run_id="retry", parent_run_id="parent", tags=["retry"])
    callback.on_llm_end(
        LLMResult(generations=[], llm_output={"model_name": "gpt-5-mini"}), run_id="retry"
    )

    events = sink.snapshot()
    assert [event.operation_id for event in events] == ["oldest-operation", "oldest-operation"]
    assert [event.attempt for event in events] == [1, 2]


def test_langchain_registration_default_handler_and_signature_are_idempotent():
    _sink, config, monitor = setup_monitor()
    assert "company_id" not in inspect.signature(LangChainUsageCallback).parameters
    first = register_langchain(monitor, config)
    second = register_langchain(monitor, config)
    try:
        assert isinstance(default_handler(), LangChainUsageCallback)
        second.uninstall()
        assert isinstance(default_handler(), LangChainUsageCallback)
    finally:
        first.uninstall()
        second.uninstall()
    assert default_handler() is None


def test_langchain_async_manager_path_preserves_context():
    from langchain_core.callbacks.manager import AsyncCallbackManager

    sink, config, monitor = setup_monitor()
    callback = LangChainUsageCallback(monitor, config)
    manager = AsyncCallbackManager.configure([callback])

    async def execute():
        with context_scope(request_id="async-request"):
            run = await manager.on_llm_start({}, ["hidden"], run_id="async-run")
        await run[0].on_llm_end(
            LLMResult(
                generations=[],
                llm_output={
                    "model_name": "models/gemini",
                    "token_usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            ),
        )

    asyncio.run(execute())
    assert sink.snapshot()[0].request_id == "async-request"


def test_langchain_async_manager_error_cancellation_emits_one_nullable_failure():
    from langchain_core.callbacks.manager import AsyncCallbackManager

    sink, config, monitor = setup_monitor()
    callback = LangChainUsageCallback(monitor, config)
    manager = AsyncCallbackManager.configure(
        [callback], local_metadata={"operation_id": "async-op"}
    )

    async def execute():
        runs = await manager.on_llm_start({}, ["hidden"], run_id="async-cancel")
        await runs[0].on_llm_error(asyncio.CancelledError())

    asyncio.run(execute())
    events = sink.snapshot()
    assert len(events) == 1
    assert events[0].status.value == "failure"
    assert events[0].stream_state == "cancelled"
    assert events[0].usage_available is False
    assert events[0].input_tokens is None
