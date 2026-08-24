"""Deterministic LangChain callback example (no provider or network call).

Run with ``python examples/langchain_fake.py`` after installing
``hyperion[langchain]``.  The fake run uses LangChain's public callback lifecycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from hyperion import HyperionConfig, context_scope, install
from hyperion.sinks.memory import MemorySink


@dataclass
class FakeLlmOutput:
    model_name: str = "anthropic/fake-model"
    token_usage: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.token_usage is None:
            self.token_usage = {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}


@dataclass
class FakeLlmResult:
    llm_output: FakeLlmOutput = field(default_factory=FakeLlmOutput)


def run_fake() -> dict[str, object]:
    """Emit one event through the registered LangChain handler lifecycle."""

    try:
        from hyperion.adapters.langchain import default_handler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This example needs the optional LangChain integration. "
            "Install it with: python -m pip install 'hyperion[langchain]'"
        ) from exc

    sink = MemorySink()
    installation = install(
        HyperionConfig(enabled=True, integrations=("langchain",), sinks=(sink,))
    )
    try:
        handler = default_handler()
        if handler is None:
            raise RuntimeError("Hyperion could not register its LangChain default_handler")
        with context_scope(
            request_id="req-fake-langchain",
            session_id="session-fake",
            call_site="support-answer",
            metadata={"tenant_id": "tenant-demo"},
        ):
            handler.on_llm_start(
                {"id": ["fake", "model"]},
                ["prompt omitted from telemetry"],
                run_id="run-fake-langchain",
                metadata={"operation_id": "op-fake-langchain"},
            )
            handler.on_llm_end(
                FakeLlmResult(),
                run_id="run-fake-langchain",
            )
        if len(sink) != 1:
            raise RuntimeError("The fake LangChain callback did not emit one usage event")
        return sink.snapshot()[0].to_dict()
    finally:
        installation.shutdown()


if __name__ == "__main__":
    try:
        print(json.dumps(run_fake(), indent=2, sort_keys=True))
    except RuntimeError as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
