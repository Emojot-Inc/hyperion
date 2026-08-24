"""Deterministic Google ADK-over-LiteLLM example (no provider or network call).

ADK users install ``google-adk[extensions]`` separately and install Hyperion with
its LiteLLM extra.  The example constructs ADK's real ``LiteLlm`` model and
``LlmRequest``; LiteLLM's ``mock_response`` seam supplies the offline response,
so Hyperion observes the normal LiteLLM callback boundary.  This is
zero ADK-specific Hyperion code.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

# This is zero ADK-specific Hyperion code.
from hyperion import HyperionConfig, context_scope, install
from hyperion.sinks.memory import MemorySink


def run_fake() -> dict[str, object]:
    """Execute one real ADK ``LiteLlm`` request through LiteLLM's offline seam."""

    try:
        from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
        from google.adk.models.llm_request import LlmRequest
        from google.genai import types
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "This example needs Google ADK's LiteLLM extension. Install it separately with: "
            "python -m pip install 'google-adk[extensions]'"
        ) from exc

    try:
        import litellm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This example needs Hyperion's LiteLLM extra. Install it with: "
            "python -m pip install 'hyperion[litellm]'"
        ) from exc

    class OfflineLiteLLMClient(LiteLLMClient):
        """Use LiteLLM's mock response while retaining its callback dispatch."""

        async def acompletion(
            self,
            model: Any,
            messages: Any,
            tools: Any,
            **kwargs: Any,
        ) -> Any:
            del kwargs
            start_time = datetime.now(timezone.utc)
            mock_response = litellm.ModelResponse(
                id="mock-adk-response",
                model=model,
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "offline ADK response"},
                        "finish_reason": "stop",
                    }
                ],
                usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            )
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                tools=tools,
                mock_response=mock_response,
                litellm_call_id="op-fake-adk",
                metadata={"attempted_retries": 0},
            )
            # LiteLLM's mock_response fast path returns before its normal
            # callback dispatcher. Re-enter that dispatcher through LiteLLM's
            # logging object, preserving the real callback contract without
            # invoking any Hyperion callback method directly.
            logging_obj = litellm.Logging(
                model=model,
                messages=messages,
                stream=False,
                call_type="acompletion",
                start_time=start_time,
                litellm_call_id="op-fake-adk",
                function_id="fake-adk",
                dynamic_async_success_callbacks=litellm.callbacks,
                kwargs={
                    "metadata": {"attempted_retries": 0},
                    "litellm_call_id": "op-fake-adk",
                },
            )
            await logging_obj.async_success_handler(
                result=response,
                end_time=datetime.now(timezone.utc),
            )
            return response

    sink = MemorySink()
    installation = install(
        HyperionConfig(enabled=True, integrations=("litellm",), sinks=(sink,))
    )
    try:
        request = LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Say hello offline")],
                )
            ]
        )
        model = LiteLlm(model="openai/fake-adk-model", llm_client=OfflineLiteLLMClient())
        with context_scope(
            request_id="req-fake-adk",
            session_id="session-fake",
            call_site="adk-agent",
            metadata={"tenant_id": "tenant-demo", "adk_agent": "fake-agent"},
        ):
            responses = asyncio.run(_collect_responses(model, request))
        if len(responses) != 1 or len(sink) != 1:
            raise RuntimeError("The real ADK LiteLlm path did not emit one event")
        return sink.snapshot()[0].to_dict()
    finally:
        installation.shutdown()


async def _collect_responses(model: Any, request: Any) -> list[Any]:
    return [response async for response in model.generate_content_async(request)]


if __name__ == "__main__":
    try:
        print(
            "Real ADK LiteLlm traffic is observed by the LiteLLM callback with zero "
            "ADK-specific Hyperion code."
        )
        print(json.dumps(run_fake(), indent=2, sort_keys=True))
    except RuntimeError as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
