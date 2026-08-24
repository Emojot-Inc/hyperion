from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("litellm")
pytest.importorskip("langchain_core")

from langchain_core.outputs import LLMResult

from reference import token_usage


class UsageObject:
    prompt_tokens = 120
    completion_tokens = 30
    total_tokens = 150


class ResponseObject:
    usage = UsageObject()


class FakeCollection:
    def __init__(self):
        self.documents = []

    def insert_one(self, document):
        self.documents.append(document)


class FakeDatabase:
    def __init__(self):
        self.collections = {"TokenUsageLogs": FakeCollection()}

    def __getitem__(self, name):
        return self.collections[name]


class FakeMongoClient:
    def __init__(self):
        self.databases = {}

    def __getitem__(self, name):
        self.databases.setdefault(name, FakeDatabase())
        return self.databases[name]


def test_record_serialization_preserves_legacy_fields():
    record = token_usage.TokenUsageRecord(
        call_site="routing",
        model_name="gpt-5-mini",
        model_provider="openai",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        duration_ms=12.5,
        request_id="request-1",
        session_id="session-1",
        timestamp=123.0,
    )

    assert record.to_dict() == {
        "call_site": "routing",
        "model_name": "gpt-5-mini",
        "model_provider": "openai",
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "duration_ms": 12.5,
        "status": "success",
        "error": None,
        "request_id": "request-1",
        "session_id": "session-1",
        "timestamp": 123.0,
    }


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("openai/gpt-5-mini", "openai"),
        ("gpt-5-mini", "openai"),
        ("claude-3-7-sonnet", "anthropic"),
        ("gemini-2.0-flash", "google"),
        ("custom-model", "unknown"),
    ],
)
def test_provider_extraction_matches_reference_heuristics(model, provider):
    assert token_usage._extract_provider(model) == provider


def test_litellm_success_extracts_usage_metadata_and_persists(monkeypatch):
    records = []
    monkeypatch.setattr(token_usage, "log_token_usage", records.append)
    client = FakeMongoClient()
    start = datetime(2026, 8, 21, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(milliseconds=250)

    token_usage.set_request_context(
        request_id="context-request",
        company_id="tenant-1",
        session_id="context-session",
    )
    try:
        callback = token_usage.LiteLLMTokenMonitor(mongo_client=client)
        callback.log_success_event(
            {
                "model": "openai/gpt-5-mini",
                "litellm_params": {
                    "metadata": {
                        "call_site": "context-call-site",
                        "request_id": "metadata-request",
                    }
                },
                "metadata": {"call_site": "proxy-call-site"},
            },
            ResponseObject(),
            start,
            end,
        )
    finally:
        token_usage.clear_request_context()

    assert len(records) == 1
    record = records[0]
    assert record.model_provider == "openai"
    assert record.input_tokens == 120
    assert record.output_tokens == 30
    assert record.total_tokens == 150
    assert record.duration_ms == 250.0
    assert record.call_site == "proxy-call-site"
    assert record.request_id == "metadata-request"
    assert record.session_id == "context-session"

    persisted = client.databases["tenant_tenant-1"].collections["TokenUsageLogs"].documents
    assert persisted == [record.to_dict()]


def test_litellm_failure_records_error_and_zero_usage(monkeypatch):
    records = []
    monkeypatch.setattr(token_usage, "log_token_usage", records.append)
    callback = token_usage.LiteLLMTokenMonitor()

    callback.log_failure_event(
        {"model": "custom-model", "exception": ValueError("provider failure")},
        None,
        None,
        None,
    )

    assert len(records) == 1
    assert records[0].status == "error"
    assert records[0].error == "provider failure"
    assert records[0].total_tokens == 0


def test_langchain_success_uses_llm_output_token_usage(monkeypatch):
    records = []
    monkeypatch.setattr(token_usage, "log_token_usage", records.append)
    clock = iter([10.0, 11.5])
    monkeypatch.setattr(token_usage.time, "time", lambda: next(clock))

    callback = token_usage.LangChainTokenMonitor(
        call_site="routing",
        request_id="request-2",
        session_id="session-2",
    )
    callback.on_llm_start({}, ["prompt"])
    callback.on_llm_end(
        LLMResult(
            generations=[],
            llm_output={
                "model_name": "gpt-5-mini",
                "token_usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28,
                },
            },
        )
    )

    assert len(records) == 1
    assert records[0].call_site == "routing"
    assert records[0].model_provider == "openai"
    assert records[0].input_tokens == 20
    assert records[0].output_tokens == 8
    assert records[0].total_tokens == 28
    assert records[0].duration_ms == 1500.0


def test_langchain_error_records_error_and_request_context(monkeypatch):
    records = []
    monkeypatch.setattr(token_usage, "log_token_usage", records.append)
    callback = token_usage.LangChainTokenMonitor(
        call_site="routing",
        request_id="request-3",
        session_id="session-3",
    )

    callback.on_llm_error(RuntimeError("llm failed"))

    assert len(records) == 1
    assert records[0].status == "error"
    assert records[0].error == "llm failed"
    assert records[0].request_id == "request-3"
    assert records[0].session_id == "session-3"


def test_litellm_registration_is_idempotent(monkeypatch):
    callbacks = []
    monkeypatch.setattr(token_usage.litellm, "callbacks", callbacks)

    token_usage.register_litellm_callback()
    token_usage.register_litellm_callback()

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], token_usage.LiteLLMTokenMonitor)
