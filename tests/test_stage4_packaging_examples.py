"""Focused Stage 4 packaging and adoption-example checks."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).parents[1]
EXAMPLES = (
    ROOT / "examples" / "litellm_fake.py",
    ROOT / "examples" / "langchain_fake.py",
    ROOT / "examples" / "adk_over_litellm_fake.py",
)


def _load(path: Path):
    name = f"hyperion_example_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses with postponed annotations resolve their module through
    # sys.modules while the import is executing.
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_exactly_three_standalone_examples_emit_structured_usage():
    assert len(tuple(ROOT.glob("examples/*.py"))) == 3
    for path in EXAMPLES:
        module = _load(path)
        try:
            event = module.run_fake()
        except RuntimeError as exc:
            pytest.skip(str(exc))
        assert event["usage_available"] is True
        assert event["input_tokens"] > 0
        assert event["output_tokens"] > 0
        assert event["details"]
        assert event["details"][next(iter(event["details"]))]["tenant_id"] == "tenant-demo"


def test_adk_example_states_litellm_callback_boundary():
    source = (ROOT / "examples" / "adk_over_litellm_fake.py").read_text()
    assert "zero ADK-specific Hyperion code" in source
    assert "integrations=(\"litellm\",)" in source
    assert "google-adk" in source
    assert "from google.adk.models.lite_llm import LiteLlm, LiteLLMClient" in source
    assert "from google.adk.models.llm_request import LlmRequest" in source
    assert "types.Content" in source
    assert "generate_content_async" in source
    assert "mock_response" in source
    assert "callback.log_success_event" not in source


def test_adk_example_executes_real_optional_adk_path_when_installed():
    try:
        import google.adk.models.lite_llm  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pytest.skip("google-adk[extensions] is not installed")

    module = _load(ROOT / "examples" / "adk_over_litellm_fake.py")
    try:
        event = module.run_fake()
    except RuntimeError as exc:
        if "google-adk[extensions]" in str(exc) or "hyperion[litellm]" in str(exc):
            pytest.skip(str(exc))
        raise
    assert event["usage_available"] is True
    assert event["input_tokens"] == 7
    assert event["output_tokens"] == 3
    assert event["operation_id"] == "op-fake-adk"


def test_metadata_extras_and_python_support():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)
    project = metadata["project"]
    assert project["requires-python"] == ">=3.10,<3.15"
    assert project["license"] == "Apache-2.0"
    assert set(project["optional-dependencies"]) == {"litellm", "langchain", "all"}
    assert project["optional-dependencies"]["all"] == ["litellm>=0", "langchain-core>=0"]
    assert set(metadata["dependency-groups"]) == {"baseline", "dev"}
    assert project["authors"] == [{"name": "Emojot"}]
    classifiers = set(project["classifiers"])
    assert {f"Programming Language :: Python :: 3.{minor}" for minor in range(10, 15)} <= classifiers


def test_cost_package_is_not_shipped():
    assert not (ROOT / "src" / "hyperion" / "cost").exists()
    assert not any(ROOT.glob("src/hyperion/cost*"))


def test_copyright_notice_names_author():
    assert (ROOT / "NOTICE").read_text() == "Copyright 2026 Emojot\n"


def test_docs_do_not_advertise_unpublished_extras_or_adapters():
    readme = (ROOT / "README.md").read_text()
    plan = (ROOT / "docs" / "hyperion-product-and-integration-plan.md").read_text()
    for text in (readme, plan):
        assert not re.search(r"hyperion\[(?:adk|mongo|openai|fastapi|dev|baseline)\]", text)
    assert "There is no ADK extra" in readme
    assert "native ADK model" in readme
    assert "native-ADK adapter" in plan
