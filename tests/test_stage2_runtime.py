import json
import logging
import threading

import pytest

from hyperion import HyperionConfig, UsageEvent, UsageMonitor, install, runtime
from hyperion.sinks import JsonLoggingSink, MemorySink


class RecordingLogger:
    def __init__(self):
        self.records = []
        self.diagnostics = []

    def log(self, level, message):
        self.records.append((level, message))

    def warning(self, message, **kwargs):
        self.diagnostics.append((message, kwargs))


def test_config_resolves_namespace_and_legacy_enabled_seed(monkeypatch):
    monkeypatch.setenv("ENABLE_TOKEN_MONITORING", "yes")
    monkeypatch.setenv("HYPERION_INTEGRATIONS", "litellm, langchain, litellm, langchain")
    monkeypatch.setenv("HYPERION_DETAILS_MAX_BYTES", "42")
    config = HyperionConfig()

    assert config.enabled is True
    assert config.integrations == ("litellm", "langchain")
    assert config.details_max_bytes == 42
    assert config.default_call_site == "application"
    assert len(config.sinks) == 1

    explicit = HyperionConfig(enabled=False, integrations=(), details_max_bytes=7)
    assert explicit.enabled is False
    assert explicit.integrations == ()
    assert explicit.details_max_bytes == 7


@pytest.mark.parametrize("integrations", ["litellm", b"litellm"])
def test_config_rejects_bare_string_integrations(integrations):
    with pytest.raises(TypeError, match=r"integrations.*sequence.*str or bytes"):
        HyperionConfig(integrations=integrations)


def test_config_deduplicates_explicit_integrations_in_first_seen_order():
    config = HyperionConfig(integrations=("langchain", "litellm", "langchain", "litellm"))

    assert config.integrations == ("langchain", "litellm")


@pytest.mark.parametrize("sinks", ["sink", b"sink", (object(),)])
def test_config_rejects_malformed_sinks(sinks):
    with pytest.raises(TypeError, match=r"sinks.*(sequence.*str or bytes|callable emit)"):
        HyperionConfig(sinks=sinks)


def test_config_prefers_hyperion_enabled_and_rejects_invalid_booleans(monkeypatch):
    monkeypatch.setenv("ENABLE_TOKEN_MONITORING", "false")
    monkeypatch.setenv("HYPERION_ENABLED", "true")
    assert HyperionConfig().enabled is True
    monkeypatch.setenv("HYPERION_ENABLED", "invalid")
    with pytest.raises(ValueError, match="HYPERION_ENABLED"):
        HyperionConfig()


def test_config_defaults_enabled_when_namespace_and_legacy_env_are_absent(monkeypatch):
    monkeypatch.delenv("HYPERION_ENABLED", raising=False)
    monkeypatch.delenv("ENABLE_TOKEN_MONITORING", raising=False)

    assert HyperionConfig().enabled is True


def test_json_sink_emits_one_compact_normalized_record():
    logger = RecordingLogger()
    sink = JsonLoggingSink(logger)
    event = UsageEvent(
        event_id="evt-test",
        start_time=1,
        end_time=2,
        details={"secret": "not-a-prompt"},
        details_max_bytes=1,
    )

    sink.emit(event)

    assert len(logger.records) == 1
    level, payload = logger.records[0]
    assert level == logging.INFO
    assert json.loads(payload) == event.to_dict()
    assert "usage_available" in payload
    assert "details_truncated" in payload
    assert "not-a-prompt" not in payload


def test_monitor_stats_count_sink_attempts_and_truncation():
    class FailingSink:
        def emit(self, event):
            raise RuntimeError("secret provider message")

    delivered = MemorySink()
    monitor = UsageMonitor([delivered, FailingSink()], diagnostic_logging=False)
    monitor.emit(UsageEvent(details={"large": "x"}, details_max_bytes=0))

    assert monitor.stats().emitted == 1
    assert monitor.stats().delivered == 1
    assert monitor.stats().sink_failures == 1
    assert monitor.stats().truncated_details == 1
    assert set(monitor.stats().__dataclass_fields__) == {
        "emitted",
        "delivered",
        "sink_failures",
        "truncated_details",
    }


def test_install_only_activates_allowlisted_explicit_integrations_and_is_idempotent(monkeypatch):
    calls = []
    teardowns = []

    class Adapter:
        def register(self, monitor, config):
            calls.append((monitor, config))
            return lambda: teardowns.append("teardown")

    adapter = Adapter()
    imported = []
    original_import = runtime.importlib.import_module

    def fake_import(name):
        imported.append(name)
        if name == "hyperion.adapters.litellm":
            return adapter
        return original_import(name)

    monkeypatch.setattr(runtime.importlib, "import_module", fake_import)
    handle = install(HyperionConfig(enabled=True, integrations=("litellm", "litellm"), sinks=()))
    assert install(HyperionConfig(enabled=True, integrations=("litellm",), sinks=())) is handle
    assert imported == ["hyperion.adapters.litellm"]
    assert len(calls) == 1
    assert handle.config.integrations == ("litellm",)
    handle.shutdown()
    handle.uninstall()
    assert teardowns == ["teardown"]


def test_install_registration_and_teardown_contract(monkeypatch):
    registrations = []
    teardown_calls = []

    class Registration:
        def uninstall(self):
            teardown_calls.append("once")

    class Module:
        def register(self, monitor, config):
            registrations.append((monitor, config))
            return Registration()

    monkeypatch.setitem(runtime.INTEGRATION_MODULES, "fake", "hyperion.adapters.fake")
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: Module())
    handle = install(HyperionConfig(enabled=True, integrations=("fake",), sinks=()))
    assert len(registrations) == 1
    handle.shutdown()
    handle.shutdown()
    handle.uninstall()
    assert teardown_calls == ["once"]


def test_install_serializes_new_registration_until_old_teardown_finishes(monkeypatch):
    teardown_started = threading.Event()
    release_teardown = threading.Event()
    second_registration = threading.Event()
    operations = []

    class Registration:
        def uninstall(self):
            operations.append("teardown-start")
            teardown_started.set()
            assert release_teardown.wait(2)
            operations.append("teardown-end")

    class Module:
        def register(self, monitor, config):
            operations.append("register")
            if teardown_started.is_set():
                second_registration.set()
            return Registration()

    monkeypatch.setitem(runtime.INTEGRATION_MODULES, "blocking", "hyperion.adapters.blocking")
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: Module())
    old = install(HyperionConfig(enabled=True, integrations=("blocking",), sinks=()))
    shutdown_thread = threading.Thread(target=old.shutdown)
    install_thread = None
    new = None
    try:
        shutdown_thread.start()
        assert teardown_started.wait(2)

        installed = {}

        def install_new():
            installed["handle"] = install(
                HyperionConfig(enabled=True, integrations=("blocking",), sinks=())
            )

        install_thread = threading.Thread(target=install_new)
        install_thread.start()
        assert not second_registration.wait(0.05)
        release_teardown.set()
        shutdown_thread.join(2)
        install_thread.join(2)
        assert not shutdown_thread.is_alive()
        assert not install_thread.is_alive()
        new = installed["handle"]
        assert operations == ["register", "teardown-start", "teardown-end", "register"]
    finally:
        release_teardown.set()
        shutdown_thread.join(2)
        if install_thread is not None:
            install_thread.join(2)
        if new is not None:
            new.shutdown()
        elif runtime._ACTIVE_INSTALLATION is old:
            old.shutdown()


def test_disabled_install_has_no_sinks_or_activation():
    handle = install(HyperionConfig(enabled=False, integrations=("litellm",)))
    try:
        handle.monitor.emit(UsageEvent())
        assert handle.monitor.sinks == ()
        assert handle.monitor.stats().emitted == 0
        assert handle.monitor.stats().delivered == 0
    finally:
        handle.uninstall()
