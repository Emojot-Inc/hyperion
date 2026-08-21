# Hyperion

Hyperion is a provider-neutral Python toolkit for understanding how applications use
large language models.

It is designed to capture token usage, model metadata, latency and request context in a
consistent event format, regardless of whether an application uses LangChain, LiteLLM,
direct provider clients or another framework.

## Why Hyperion?

LLM usage is often scattered across framework callbacks, provider responses and custom
application code. That makes it difficult to answer basic operational questions:

- Which models are being called?
- How many input and output tokens are being used?
- Which workflows or requests are driving cost?
- Where are latency, retries or failed calls occurring?
- Can usage be recorded without coupling the application to one database or vendor?

Hyperion provides a common monitoring boundary for those concerns. Applications emit
structured usage events, then choose how to store, inspect or aggregate them.

## What you can do with it

The project is being built around a small standard-library core and optional integrations.
Planned capabilities include:

- Record normalized token usage and model metadata.
- Associate model calls with request, session, tenant and workflow context.
- Capture successful, failed, retried and streamed calls.
- Send events to logging, JSONL, in-memory or database sinks.
- Add framework adapters without adding those frameworks to the core package.
- Calculate usage-based costs from configurable model pricing.
- Keep prompts, completions and tool arguments out of telemetry by default.

The intended flow is:

```text
application call → framework adapter → usage event → sink → analysis or reporting
```

## Current status

Hyperion is currently in the scaffolding stage. The package layout, build configuration
and development checks are present, but the public event schema and integration APIs are
not stable yet. Do not depend on this package for production monitoring until the first
functional release is published.

## Getting started

Clone the repository and install the development dependencies in an isolated environment:

```bash
git clone <repository-url>
cd hyperion
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the current checks:

```bash
pytest
ruff check .
```

At this stage, the checks verify the package scaffold and import boundary. Framework
adapters and monitoring workflows will be added in subsequent releases.

## Installation roadmap

Once the first functional release is available, integrations will be installable through
optional dependency groups rather than forcing every user to install every framework:

```bash
python -m pip install "hyperion[langchain]"
python -m pip install "hyperion[litellm]"
python -m pip install "hyperion[mongo]"
```

The core package is intended to remain independent of LangChain, LiteLLM, MongoDB and
specific model-provider SDKs.

## Getting help

For questions, bug reports and feature requests, open an issue in the repository hosting
this project. Include:

- The Hyperion version and Python version.
- Which optional integration is involved.
- The relevant event or callback shape, with secrets and user content removed.
- A small reproducible example and the expected behavior.

Before opening a new issue, search existing issues for related discussions. Public API
changes and new integrations should include documentation and tests.
