import json
import math
from dataclasses import FrozenInstanceError, fields

import pytest

import hyperion
from hyperion.events import EventStatus, UsageEvent


def test_event_is_immutable_and_has_unique_observation_ids():
    event = UsageEvent(start_time=10, end_time=11)
    other = UsageEvent(start_time=10, end_time=11)

    assert event.event_id != other.event_id
    assert event.event_id.startswith("evt_")
    with pytest.raises(FrozenInstanceError):
        event.model = "changed"


def test_attempts_share_explicit_operation_id_but_have_distinct_event_ids():
    first = UsageEvent(operation_id="operation-1", attempt=1)
    retry = UsageEvent(operation_id=first.operation_id, attempt=2)

    assert retry.operation_id == first.operation_id
    assert retry.event_id != first.event_id
    assert UsageEvent().operation_id != first.operation_id


def test_operation_type_is_extensible_and_completion_is_default():
    assert UsageEvent().operation_type == "completion"
    assert UsageEvent(operation_type="embedding-v2").operation_type == "embedding-v2"


def test_normalized_serialization_is_versioned_and_excludes_legacy_only_fields():
    event = UsageEvent(
        provider="openai",
        model="gpt-test",
        operation_type="custom",
        start_time=100.0,
        end_time=101.5,
        duration_ms=1500,
        stream_state="final",
        usage_available=True,
        input_tokens=0,
        output_tokens=3,
        total_tokens=3,
        details={"nested": {"ok": True}, "items": [1, "two"]},
    )

    data = event.to_dict()
    assert data["schema_version"] == event.SCHEMA_VERSION
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-test"
    assert data["start_time"] == 100.0
    assert data["end_time"] == 101.5
    assert data["stream_state"] == "final"
    assert json.loads(json.dumps(data)) == data
    assert "kind" not in data
    assert "content" not in data


def test_timing_status_stream_and_attempt_invariants():
    with pytest.raises(ValueError, match="attempt"):
        UsageEvent(attempt=0)
    with pytest.raises(ValueError, match="end_time"):
        UsageEvent(start_time=2, end_time=1)
    with pytest.raises(ValueError, match="duration_ms"):
        UsageEvent(duration_ms=-1)
    with pytest.raises(ValueError, match="start_time"):
        UsageEvent(start_time=math.nan)
    with pytest.raises(ValueError, match="success events"):
        UsageEvent(error="not allowed")
    with pytest.raises(ValueError, match="failure events"):
        UsageEvent(status=EventStatus.FAILURE)
    with pytest.raises(ValueError, match="stream_state"):
        UsageEvent(stream_state="")


def test_stream_state_is_one_final_observation_and_no_chunk_surface_exists():
    event = UsageEvent(stream_state="cancelled", status="failure", error="cancelled")

    assert event.stream_state == "cancelled"
    assert not hasattr(event, "stream_sequence")
    assert not hasattr(event, "stream_id")
    assert not hasattr(event, "stream_final")
    assert not hasattr(event, "kind")


def test_unknown_usage_is_null_and_available_usage_preserves_genuine_zero():
    unknown = UsageEvent()
    assert unknown.usage_available is False
    assert unknown.input_tokens is None
    assert unknown.output_tokens is None
    assert unknown.total_tokens is None

    zero = UsageEvent(usage_available=True, input_tokens=0, output_tokens=0, total_tokens=0)
    assert zero.to_dict()["input_tokens"] == 0
    assert zero.to_dict()["output_tokens"] == 0
    assert zero.to_dict()["total_tokens"] == 0

    with pytest.raises(ValueError, match="unavailable"):
        UsageEvent(input_tokens=0)
    inconsistent = UsageEvent(
        usage_available=True, input_tokens=1, output_tokens=2, total_tokens=99
    )
    assert (inconsistent.input_tokens, inconsistent.output_tokens, inconsistent.total_tokens) == (
        1,
        2,
        99,
    )
    with pytest.raises(ValueError, match="zero"):
        UsageEvent(usage_available=True, input_tokens=-1)
    with pytest.raises(TypeError, match="input_tokens"):
        UsageEvent(usage_available=True, input_tokens=1.5)


def test_partial_available_usage_does_not_invent_missing_counts():
    event = UsageEvent(usage_available=True, input_tokens=4, output_tokens=None, total_tokens=None)

    assert event.input_tokens == 4
    assert event.output_tokens is None
    assert event.total_tokens is None


def test_error_is_bounded_deterministically():
    error = "secret-provider-error-" + ("x" * 10_000)
    event = UsageEvent(status="failure", error=error)
    again = UsageEvent(status="failure", error=error)

    assert len(event.error or "") <= UsageEvent.MAX_ERROR_LENGTH
    assert event.error == again.error
    assert event.error and event.error.endswith("...[truncated]")
    assert error not in event.error


def test_details_are_recursively_frozen_and_defensively_copied():
    supplied = {"nested": {"values": [1, 2]}}
    event = UsageEvent(details=supplied)
    supplied["nested"]["values"].append(3)

    assert event.details["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        event.details["nested"]["new"] = "nope"
    with pytest.raises(TypeError):
        event.details["nested"]["values"][0] = 9


@pytest.mark.parametrize(
    "details",
    [
        {1: "non-string key"},
        {"bad": {"value"}},
        {"bad": object()},
        {"bad": float("nan")},
        {"bad": float("inf")},
    ],
)
def test_details_reject_non_json_compatible_values(details):
    with pytest.raises(TypeError, match="details"):
        UsageEvent(details=details)


def test_details_size_bound_is_deterministic_and_exposes_truncation_metadata():
    details = {"secret": "value-" + ("x" * 1000), "other": [1, 2, 3]}
    first = UsageEvent(details=details, details_max_bytes=32)
    second = UsageEvent(details={"other": [1, 2, 3], "secret": "value-" + ("x" * 1000)}, details_max_bytes=32)

    assert first.details_truncated is True
    assert first.to_dict()["details_truncated"] is True
    assert first.details == second.details
    assert len(json.dumps(first.to_dict()["details"], separators=(",", ":")).encode()) <= 32

    untouched = UsageEvent(details={"small": "ok"}, details_max_bytes=32)
    assert untouched.details_truncated is False


def test_details_zero_bound_preserves_event_with_explicit_truncation():
    event = UsageEvent(details={"large": "x"}, details_max_bytes=0)

    assert event.details == {}
    assert event.details_truncated is True


def test_empty_details_at_zero_bound_are_not_marked_truncated():
    event = UsageEvent(details={}, details_max_bytes=0)

    assert event.details == {}
    assert event.details_truncated is False


def test_legacy_serializer_has_exact_fields_and_maps_end_time_to_timestamp():
    event = UsageEvent(
        call_site="router",
        model="gpt-test",
        provider="openai",
        request_id="req-1",
        session_id="sess-1",
        start_time=100.0,
        end_time=123.25,
        duration_ms=23250,
        usage_available=False,
        status="failure",
        error="timeout",
    )

    legacy = event.to_legacy_dict()
    assert tuple(legacy) == UsageEvent.LEGACY_FIELDS
    assert set(legacy) == {
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
    }
    assert legacy["model_name"] == "gpt-test"
    assert legacy["model_provider"] == "openai"
    assert legacy["timestamp"] == 123.25
    assert legacy["input_tokens"] is None
    assert legacy["status"] == "error"


def test_forbidden_legacy_identity_and_content_surface_removed():
    names = {item.name for item in fields(UsageEvent)}
    forbidden = {
        "duplicate_key",
        "retry_of_event_id",
        "prompt",
        "completion",
        "content",
        "stream_id",
        "stream_sequence",
        "stream_final",
        "kind",
    }
    assert names.isdisjoint(forbidden)
    assert not hasattr(hyperion, "EventKind")
    with pytest.raises(TypeError):
        UsageEvent(content="secret")
