# Hyperion

Hyperion is a provider-neutral Python toolkit for recording structured LLM usage
events. It captures model metadata, token usage, latency, outcome and application
context without recording prompts, completions or tool arguments.

The package currently supports Python 3.10 through 3.14. The core has no runtime
dependencies; integrations are explicit optional installations.

## Install

```bash
python -m pip install hyperion
python -m pip install "hyperion[litellm]"
python -m pip install "hyperion[langchain]"
python -m pip install "hyperion[all]"
```

ADK users install Google ADK separately, then install Hyperion's LiteLLM integration:

```bash
python -m pip install "google-adk[extensions]"
python -m pip install "hyperion[litellm]"
```

There is no ADK extra. ADK's `google.adk.models.lite_llm.LiteLlm` traffic is already
observed by Hyperion's LiteLLM callback; no ADK-specific Hyperion adapter is required.

## Bootstrap and context

Install once during application startup. Configuration is explicit: Hyperion never
discovers integrations merely because a package happens to be installed.

```python
from hyperion import HyperionConfig, context_scope, install

installation = install(
    HyperionConfig(
        enabled=True,
        integrations=("litellm",),
        # Omit sinks to use the default JSON logging sink.
    )
)

with context_scope(
    request_id="req-123",
    session_id="session-456",
    call_site="checkout",
    metadata={"tenant_id": "tenant-42", "workflow": "payment-help"},
):
    application_model_call()
```

`context_scope` belongs at an application boundary, not around every model call.
Contexts nest, restore correctly and follow asynchronous tasks. Tenancy belongs in
`metadata` (for example, `tenant_id`); it is not a normalized event field and Hyperion
does not resolve tenants or choose tenant storage.

The returned installation is a lifecycle handle. `shutdown()` and `uninstall()` are
idempotent and remove registered integration hooks. Telemetry is fail-open: malformed
provider payloads and sink failures do not replace an application's result or error.

When neither `HYPERION_ENABLED` nor legacy `ENABLE_TOKEN_MONITORING` is set,
monitoring is enabled by default. Either environment variable, and then an explicit
`HyperionConfig(enabled=...)` value, can disable it.

## Events and sinks

Each `UsageEvent` has a schema version, event and operation identity, provider/model,
status, timing, stream state, context and immutable JSON-compatible details. The
default sink writes one compact JSON record per event through the `hyperion` logger.
Use `MemorySink` or your own object with an `emit(event)` method for local inspection
or application-owned delivery.

Unknown usage is different from zero usage. If a provider does not supply a complete,
valid usage payload, the event has `usage_available=False` and all three token counts
are `None`. A genuine zero-token response has `usage_available=True` and zero counts.
This prevents missing usage from being mistaken for a free call.

The installation monitor exposes process-local diagnostics:

```python
stats = installation.monitor.stats()
print(stats.emitted, stats.delivered, stats.sink_failures, stats.truncated_details)
```

## Integrations

LiteLLM is activated with `integrations=("litellm",)`. It records final success and
failure callbacks, including asynchronous callbacks, retries and authoritative stream
usage. LiteLLM's provider/model metadata is normalized into the common event shape.
Retry correlation uses LiteLLM's stable `litellm_call_id` (top-level or inside
`litellm_params`) plus callback `attempt`/`attempted_retries` metadata. Applications
should pass an explicit `operation_id` when their retry layer does not preserve those
fields; request IDs are not treated as retry correlation.

LangChain has no safe global interception hook. Activate it with
`integrations=("langchain",)` and attach the handler returned by `default_handler()`
to the callback configuration owned by your application:

```python
from hyperion.adapters.langchain import default_handler

handler = default_handler()
chain.invoke({"question": "..."}, config={"callbacks": [handler]})
```

The handler can also be constructed directly as `LangChainUsageCallback` when an
application needs explicit ownership or per-handler metadata.
In pinned `langchain-core==1.4.9`, each `CallbackManager.on_llm_start` call uses a
fresh `run_id` unless the caller supplies one; these internal callback run IDs identify
individual operations. External retry layers must preserve an explicit
`operation_id`, or provide retry/attempt metadata (with a stable parent run) when
retrying. Hyperion does not guess that metadata-free child runs sharing a parent are
retries. Explicit `metadata={"operation_id": "..."}` always takes precedence. The
callback manager's `on_retry` notification does not itself create a second LLM run.

For ADK, use `google.adk.models.lite_llm.LiteLlm` and the LiteLLM integration. ADK
LiteLlm traffic reaches LiteLLM's callback seam, so this path has zero ADK-specific
Hyperion code. Hyperion does not claim coverage for a native ADK model in this cut.

## Fake examples

The three standalone examples are deterministic and make no provider calls, use no
API keys and print one structured event:

```bash
python examples/litellm_fake.py
python examples/langchain_fake.py
python examples/adk_over_litellm_fake.py
```

Each example demonstrates startup installation, an application-boundary context
(including tenancy metadata) and emitted usage. Optional-package failures print an
installation instruction and exit with status 2.

## Development

The pinned characterization baseline and development tools are kept in the non-
published PEP 735 `baseline` and `dev` dependency groups. A requirements mirror is
provided for pip versions that do not yet install dependency groups:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pytest
python -m ruff check .
python -m build
```

## Non-goals for this release

Hyperion does not calculate or price usage, capture content, provide a native ADK
adapter, persist to MongoDB or a durable queue, call OpenAI directly, provide FastAPI
middleware, or deduplicate events. Durable delivery, exporters, content policy and
other provider/framework integrations remain application-owned or future work.
