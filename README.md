# Hyperion

Provider-neutral LLM usage monitoring and cost instrumentation for Python applications.

Hyperion provides a small standard-library core, with optional adapters for frameworks
such as LangChain and LiteLLM and optional persistence integrations.

## Status

Early scaffolding. The public event schema and adapter contracts are not yet stable.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Planned boundaries

- Core package: events, request context, monitoring, dispatch and sink protocols.
- Optional integrations: LangChain, LiteLLM, embedding providers and MongoDB.
- Application-specific database naming, conversation joins, HTTP routes and evaluation
  reporting remain outside Hyperion.
