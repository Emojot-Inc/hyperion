from hyperion.events import UsageEvent
from hyperion.monitor import SinkErrorReport, UsageMonitor


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[UsageEvent] = []

    def emit(self, event: UsageEvent) -> None:
        self.events.append(event)


class FailingSink:
    def __init__(self, message: str = "sink failed") -> None:
        self.message = message

    def emit(self, event: UsageEvent) -> None:
        raise RuntimeError(self.message)


class RecordingLogger:
    def __init__(self) -> None:
        self.exception_calls: list[tuple[str, dict[str, object]]] = []
        self.warning_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def exception(self, message: str, **kwargs: object) -> None:
        self.exception_calls.append((message, kwargs))

    def warning(self, *args: object, **kwargs: object) -> None:
        self.warning_calls.append((args, kwargs))


class SecretReprFailingSink(FailingSink):
    def __repr__(self) -> str:
        return "SecretReprFailingSink(endpoint='https://example.invalid', token='sk-secret')"


def test_monitor_emits_to_sinks_in_configured_order_and_retains_duplicates():
    first = RecordingSink()
    second = RecordingSink()
    monitor = UsageMonitor([first, second])
    event = UsageEvent(request_id="req-1", start_time=1.0, end_time=1.0)

    monitor.emit(event)
    monitor.emit(event)

    assert first.events == [event, event]
    assert second.events == [event, event]


def test_monitor_uses_fixed_sink_snapshot():
    first = RecordingSink()
    sinks = [first]
    monitor = UsageMonitor(sinks)
    second = RecordingSink()
    sinks.append(second)

    event = UsageEvent(request_id="req-1", start_time=1.0, end_time=1.0)
    monitor.emit(event)

    assert first.events == [event]
    assert second.events == []
    assert monitor.sinks == (first,)


def test_monitor_isolates_sink_failures_and_reports_them():
    failed = FailingSink("database unavailable")
    after_failure = RecordingSink()
    reports: list[SinkErrorReport] = []
    monitor = UsageMonitor([failed, after_failure], on_sink_error=reports.append)
    event = UsageEvent(request_id="req-1", start_time=1.0, end_time=1.0)

    monitor.emit(event)

    assert after_failure.events == [event]
    assert len(reports) == 1
    assert reports[0].sink_type == "FailingSink"
    assert reports[0].sink_name == "FailingSink"
    assert reports[0].exception_type == "RuntimeError"
    assert reports[0].event_id != event.event_id
    assert reports[0].event_id.startswith("evt_")


def test_monitor_sink_error_callback_receives_no_raw_failure_objects_or_messages():
    secret = "sk-secret-callback"
    failed = FailingSink(f"database unavailable: {secret}")
    reports: list[SinkErrorReport] = []
    monitor = UsageMonitor([failed], on_sink_error=reports.append)
    event = UsageEvent(request_id="req-secret", start_time=1.0, end_time=1.0)

    monitor.emit(event)

    assert len(reports) == 1
    report = reports[0]
    assert report.sink_type == "FailingSink"
    assert report.sink_name == "FailingSink"
    assert report.event_id != event.event_id
    assert report.event_id.startswith("evt_")
    assert report.exception_type == "RuntimeError"
    report_text = repr(report)
    assert secret not in report_text
    assert "database unavailable" not in report_text
    assert "req-secret" not in report_text


def test_monitor_sanitizes_explicit_event_id_in_sink_error_reports_and_logs():
    event_id_secret = "evt-user-token-sk-secret-event-id"
    exception_secret = "sk-secret-exception-message"
    failed = FailingSink(f"database unavailable: {exception_secret}")
    reports: list[SinkErrorReport] = []
    monitor = UsageMonitor([failed], on_sink_error=reports.append)
    event = UsageEvent(
        event_id=event_id_secret,
        request_id="req-secret",
        start_time=1.0,
        end_time=1.0,
    )

    monitor.emit(event)

    assert event.event_id == event_id_secret
    assert len(reports) == 1
    report = reports[0]
    report_text = f"{report} {report.to_log_extra()}"
    assert report.event_id != event_id_secret
    assert report.event_id.startswith("evt_")
    assert event_id_secret not in report_text
    assert exception_secret not in report_text
    assert "database unavailable" not in report_text

    logger = RecordingLogger()
    UsageMonitor([failed], logger=logger).emit(event)

    assert len(logger.warning_calls) == 1
    args, kwargs = logger.warning_calls[0]
    log_text = f"{args} {kwargs}"
    assert event_id_secret not in log_text
    assert exception_secret not in log_text
    assert "database unavailable" not in log_text
    assert kwargs["extra"]["event_id"] == report.event_id


def test_monitor_default_sink_error_report_excludes_sink_repr_and_configuration():
    logger = RecordingLogger()
    monitor = UsageMonitor([SecretReprFailingSink()], logger=logger)
    event = UsageEvent(request_id="req-1", start_time=1.0, end_time=1.0)

    monitor.emit(event)

    assert logger.exception_calls == []
    assert len(logger.warning_calls) == 1
    args, kwargs = logger.warning_calls[0]
    message = " ".join(str(arg) for arg in args)
    report_text = f"{message} {kwargs}"
    assert "sk-secret" not in report_text
    assert "endpoint" not in report_text
    assert set(kwargs) == {"extra"}
    assert kwargs["extra"]["exception_type"] == "RuntimeError"
    assert kwargs["extra"]["sink_name"] == "SecretReprFailingSink"
    assert kwargs["extra"]["sink_type"] == "SecretReprFailingSink"
    assert kwargs["extra"]["event_id"] != event.event_id
    assert kwargs["extra"]["event_id"].startswith("evt_")


def test_monitor_default_sink_error_log_excludes_exception_message_and_traceback():
    secret = "sk-secret-logger"
    logger = RecordingLogger()
    monitor = UsageMonitor([FailingSink(f"database unavailable: {secret}")], logger=logger)
    event = UsageEvent(request_id="req-1", start_time=1.0, end_time=1.0)

    monitor.emit(event)

    assert logger.exception_calls == []
    assert len(logger.warning_calls) == 1
    args, kwargs = logger.warning_calls[0]
    report_text = f"{args} {kwargs}"
    assert secret not in report_text
    assert "database unavailable" not in report_text
    assert "traceback" not in report_text.lower()
    assert kwargs.get("exc_info") is None
