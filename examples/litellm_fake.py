"""Deterministic LiteLLM callback example (no provider or network call).

Run with ``python examples/litellm_fake.py`` after installing
``hyperion[litellm]``.  The fake response exercises LiteLLM's callback seam directly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Keep the fake example offline even with LiteLLM versions that fetch their model
# catalog during import.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

from hyperion import HyperionConfig, context_scope, install
from hyperion.sinks.memory import MemorySink


@dataclass
class FakeUsage:
    prompt_tokens: int = 12
    completion_tokens: int = 5
    total_tokens: int = 17


@dataclass
class FakeResponse:
    usage: FakeUsage = field(default_factory=FakeUsage)


def run_fake() -> dict[str, object]:
    """Emit one event through the real LiteLLM callback without calling a provider."""

    try:
        from hyperion.adapters.litellm import LiteLLMUsageCallback
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This example needs the optional LiteLLM integration. "
            "Install it with: python -m pip install 'hyperion[litellm]'"
        ) from exc

    sink = MemorySink()
    installation = install(
        HyperionConfig(enabled=True, integrations=("litellm",), sinks=(sink,))
    )
    try:
        callback = next(
            (item for item in _litellm_callbacks() if isinstance(item, LiteLLMUsageCallback)),
            None,
        )
        if callback is None:
            raise RuntimeError("Hyperion could not register its LiteLLM callback")
        with context_scope(
            request_id="req-fake-litellm",
            session_id="session-fake",
            call_site="checkout",
            metadata={"tenant_id": "tenant-demo"},
        ):
            callback.log_success_event(
                {
                    "model": "openai/fake-model",
                    "litellm_call_id": "op-fake-litellm",
                },
                FakeResponse(),
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 0, 0, 0, 25_000, tzinfo=timezone.utc),
            )
        if len(sink) != 1:
            raise RuntimeError("The fake LiteLLM callback did not emit one usage event")
        return sink.snapshot()[0].to_dict()
    finally:
        installation.shutdown()


def _litellm_callbacks() -> list[object]:
    try:
        import litellm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This example needs LiteLLM. Install it with: "
            "python -m pip install 'hyperion[litellm]'"
        ) from exc
    callbacks = getattr(litellm, "callbacks", ())
    return callbacks if isinstance(callbacks, list) else []


if __name__ == "__main__":
    try:
        print(json.dumps(run_fake(), indent=2, sort_keys=True))
    except RuntimeError as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
