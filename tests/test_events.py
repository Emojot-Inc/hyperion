import math
from dataclasses import FrozenInstanceError
from time import time

import pytest

from hyperion import EventKind, EventStatus, UsageEvent


def test_usage_event_serializes_legacy_token_usage_record_fields():
    event = UsageEvent(
        call_site="agent_router",
        model_name="openai/gpt-4.1-mini",
        model_provider="openai",
        input_tokens=11,
        output_tokens=17,
        total_tokens=28,
        duration_ms=123.5,
        request_id="req-1",
        session_id="sess-1",
        timestamp=1_725_000_000.25,
    )

    assert event.to_legacy_dict() == {
        "call_site": "agent_router",
        "model_name": "openai/gpt-4.1-mini",
        "model_provider": "openai",
        "input_tokens": 11,
        "output_tokens": 17,
        "total_tokens": 28,
        "duration_ms": 123.5,
        "status": "success",
        "error": None,
        "request_id": "req-1",
        "session_id": "sess-1",
        "timestamp": 1_725_000_000.25,
    }


def test_usage_event_is_immutable_and_rejects_raw_content_fields():
    event = UsageEvent(request_id="req-privacy")

    with pytest.raises(FrozenInstanceError):
        event.request_id = "changed"

    serialized = event.to_dict()
    assert "prompt" not in serialized
    assert "completion" not in serialized
    assert "content" not in serialized

    with pytest.raises(ValueError, match="raw content"):
        UsageEvent(prompt="secret prompt")
    with pytest.raises(ValueError, match="raw content"):
        UsageEvent(completion="secret answer")


def test_usage_event_has_stable_identity_and_duplicate_key():
    first = UsageEvent(request_id="req-1", model_name="gpt-4.1", timestamp=10.0)
    duplicate = UsageEvent(request_id="req-1", model_name="gpt-4.1", timestamp=10.0)
    different_attempt = UsageEvent(request_id="req-1", model_name="gpt-4.1", timestamp=10.0, attempt=2)

    assert first.event_id == duplicate.event_id
    assert first.duplicate_key == duplicate.duplicate_key
    assert different_attempt.event_id != first.event_id
    assert different_attempt.duplicate_key != first.duplicate_key


def test_default_event_identity_is_canonical_for_delimiter_placement():
    first = UsageEvent(
        request_id="req|sess",
        session_id="site",
        call_site="provider",
        model_provider="model",
        model_name="10",
        input_tokens=20,
        output_tokens=30,
        total_tokens=40,
        duration_ms=50.0,
        timestamp=60.0,
    )
    delimiter_shifted = UsageEvent(
        request_id="req",
        session_id="sess|site",
        call_site="provider",
        model_provider="model",
        model_name="10",
        input_tokens=20,
        output_tokens=30,
        total_tokens=40,
        duration_ms=50.0,
        timestamp=60.0,
    )

    assert delimiter_shifted.event_id != first.event_id
    assert delimiter_shifted.duplicate_key != first.duplicate_key


def test_default_duplicate_key_does_not_expose_raw_identifiers_or_error_text():
    event = UsageEvent(
        status=EventStatus.FAILURE,
        error="provider timeout for user-secret@example.com",
        request_id="req-secret",
        session_id="session-secret",
        call_site="checkout-agent",
        model_provider="openai",
        model_name="gpt-secret",
        timestamp=10.0,
    )

    duplicate_key = event.duplicate_key
    assert "req-secret" not in duplicate_key
    assert "session-secret" not in duplicate_key
    assert "checkout-agent" not in duplicate_key
    assert "gpt-secret" not in duplicate_key
    assert "provider timeout" not in duplicate_key
    assert "user-secret@example.com" not in duplicate_key


def test_default_event_identity_includes_accounting_fields_without_breaking_reconstruction():
    base_kwargs = {
        "kind": EventKind.COMPLETION,
        "request_id": "req-1",
        "session_id": "sess-1",
        "call_site": "agent",
        "model_provider": "openai",
        "model_name": "gpt-4.1",
        "timestamp": 10.0,
        "attempt": 1,
    }
    first = UsageEvent(**base_kwargs, input_tokens=10, output_tokens=20, total_tokens=30, duration_ms=5.0)
    reconstructed = UsageEvent(
        **base_kwargs,
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        duration_ms=5.0,
    )
    different_input = UsageEvent(
        **base_kwargs,
        input_tokens=11,
        output_tokens=20,
        total_tokens=31,
        duration_ms=5.0,
    )
    different_duration = UsageEvent(
        **base_kwargs,
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        duration_ms=6.0,
    )
    failure = UsageEvent(
        **base_kwargs,
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        duration_ms=5.0,
        status=EventStatus.FAILURE,
        error="timeout",
    )
    different_error = UsageEvent(
        **base_kwargs,
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        duration_ms=5.0,
        status=EventStatus.FAILURE,
        error="rate limit",
    )
    explicit = UsageEvent(**base_kwargs, input_tokens=99, event_id="evt-explicit")

    assert reconstructed.event_id == first.event_id
    assert reconstructed.duplicate_key == first.duplicate_key
    assert different_input.event_id != first.event_id
    assert different_duration.event_id != first.event_id
    assert failure.event_id != first.event_id
    assert different_error.event_id != failure.event_id
    assert explicit.event_id == "evt-explicit"


def test_usage_event_models_failure_retry_and_streaming_without_framework_types():
    failure = UsageEvent(status=EventStatus.FAILURE, error="provider timeout")
    retry = UsageEvent(kind=EventKind.RETRY, attempt=2, retry_of_event_id=failure.event_id)
    chunk = UsageEvent(
        kind=EventKind.STREAM_CHUNK,
        stream_id="stream-1",
        stream_sequence=3,
        stream_final=False,
    )
    final = UsageEvent(
        kind=EventKind.STREAM_FINAL,
        stream_id="stream-1",
        stream_sequence=4,
        stream_final=True,
    )

    assert failure.to_legacy_dict()["status"] == "error"
    assert retry.to_dict()["kind"] == "retry"
    assert retry.to_dict()["attempt"] == 2
    assert retry.to_dict()["retry_of_event_id"] == failure.event_id
    assert chunk.to_dict()["stream_sequence"] == 3
    assert chunk.to_dict()["stream_final"] is False
    assert final.to_dict()["stream_final"] is True


def test_usage_event_rejects_success_with_error_before_serialization():
    with pytest.raises(ValueError, match="success events cannot set error"):
        UsageEvent(status=EventStatus.SUCCESS, error="provider timeout")
    with pytest.raises(ValueError, match="success events cannot set error"):
        UsageEvent(status="success", error="")

    failure = UsageEvent(status=EventStatus.FAILURE, error="provider timeout")
    assert failure.error == "provider timeout"
    assert failure.to_legacy_dict()["error"] == "provider timeout"


def test_usage_event_enforces_retry_and_streaming_invariants():
    with pytest.raises(ValueError, match="retry events require retry_of_event_id"):
        UsageEvent(kind=EventKind.RETRY, attempt=2)
    with pytest.raises(ValueError, match="retry events require attempt greater than 1"):
        UsageEvent(kind=EventKind.RETRY, attempt=1, retry_of_event_id="evt-original")
    with pytest.raises(ValueError, match="non-retry events cannot set retry_of_event_id"):
        UsageEvent(kind=EventKind.COMPLETION, retry_of_event_id="evt-original")

    with pytest.raises(ValueError, match="stream events require stream_sequence"):
        UsageEvent(kind=EventKind.STREAM_CHUNK, stream_id="stream-1")
    with pytest.raises(ValueError, match="stream_chunk events must set stream_final=False"):
        UsageEvent(
            kind=EventKind.STREAM_CHUNK,
            stream_id="stream-1",
            stream_sequence=1,
            stream_final=True,
        )
    with pytest.raises(ValueError, match="stream_final events must set stream_final=True"):
        UsageEvent(kind=EventKind.STREAM_FINAL, stream_id="stream-1", stream_sequence=1)
    with pytest.raises(ValueError, match="non-stream events cannot set stream fields"):
        UsageEvent(kind=EventKind.COMPLETION, stream_id="stream-1")


def test_usage_event_has_sensible_defaults_and_validates_public_field_types():
    before = time()
    event = UsageEvent()
    after = time()

    assert event.call_site == ""
    assert event.input_tokens == 0
    assert event.output_tokens == 0
    assert event.total_tokens == 0
    assert event.duration_ms == 0.0
    assert before <= event.timestamp <= after

    with pytest.raises(TypeError, match="input_tokens"):
        UsageEvent(input_tokens=1.5)
    with pytest.raises(TypeError, match="duration_ms"):
        UsageEvent(duration_ms="123")
    with pytest.raises(ValueError, match="attempt"):
        UsageEvent(attempt=0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field_name", ["duration_ms", "timestamp"])
def test_usage_event_rejects_non_finite_numeric_timing_values(field_name, value):
    with pytest.raises(ValueError, match=field_name):
        UsageEvent(**{field_name: value})
