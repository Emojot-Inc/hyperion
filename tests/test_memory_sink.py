from hyperion.events import UsageEvent
from hyperion.sinks.memory import MemorySink


def test_memory_sink_retains_bounded_fifo_snapshot_without_exposing_storage():
    sink = MemorySink(max_events=2)
    first = UsageEvent(request_id="req-1", timestamp=1.0)
    second = UsageEvent(request_id="req-2", timestamp=2.0)
    third = UsageEvent(request_id="req-3", timestamp=3.0)

    sink.emit(first)
    sink.emit(second)
    snapshot_before_eviction = sink.snapshot()
    sink.emit(third)

    assert snapshot_before_eviction == (first, second)
    assert sink.snapshot() == (second, third)
    assert isinstance(sink.snapshot(), tuple)


def test_memory_sink_rejects_non_positive_bounds():
    for max_events in (0, -1):
        try:
            MemorySink(max_events=max_events)
        except ValueError as exc:
            assert "max_events" in str(exc)
        else:
            raise AssertionError("MemorySink accepted a non-positive max_events")


def test_memory_sink_retains_duplicate_events_by_default():
    sink = MemorySink(max_events=10)
    event = UsageEvent(request_id="req-1", timestamp=1.0)

    sink.emit(event)
    sink.emit(event)

    assert sink.snapshot() == (event, event)


def test_memory_sink_can_deduplicate_by_event_id():
    sink = MemorySink(max_events=10, deduplicate_by_event_id=True)
    first = UsageEvent(request_id="req-1", timestamp=1.0)
    duplicate = UsageEvent(request_id="req-1", timestamp=1.0)
    different = UsageEvent(request_id="req-1", timestamp=1.0, attempt=2)

    sink.emit(first)
    sink.emit(duplicate)
    sink.emit(different)

    assert duplicate.event_id == first.event_id
    assert sink.snapshot() == (first, different)
