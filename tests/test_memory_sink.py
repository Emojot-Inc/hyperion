import pytest

from hyperion.events import UsageEvent
from hyperion.sinks.memory import MemorySink


def test_memory_sink_retains_bounded_fifo_snapshot_without_exposing_storage():
    sink = MemorySink(max_events=2)
    first = UsageEvent(request_id="req-1", start_time=1, end_time=1)
    second = UsageEvent(request_id="req-2", start_time=2, end_time=2)
    third = UsageEvent(request_id="req-3", start_time=3, end_time=3)

    sink.emit(first)
    sink.emit(second)
    before_eviction = sink.snapshot()
    sink.emit(third)

    assert before_eviction == (first, second)
    assert sink.snapshot() == (second, third)
    assert isinstance(sink.snapshot(), tuple)


def test_memory_sink_retains_duplicate_observations_and_has_no_dedup_option():
    sink = MemorySink(max_events=10)
    event = UsageEvent(request_id="req-1", start_time=1, end_time=1)

    sink.emit(event)
    sink.emit(event)

    assert sink.snapshot() == (event, event)
    with pytest.raises(TypeError):
        MemorySink(max_events=10, deduplicate_by_event_id=True)


def test_memory_sink_rejects_invalid_bounds():
    with pytest.raises(TypeError):
        MemorySink(max_events=True)
    for max_events in (0, -1):
        with pytest.raises(ValueError, match="max_events"):
            MemorySink(max_events=max_events)
