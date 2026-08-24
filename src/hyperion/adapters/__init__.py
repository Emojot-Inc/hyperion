"""Optional framework adapters for Hyperion.

Framework modules are imported lazily so importing :mod:`hyperion` and its core
modules never requires LiteLLM or LangChain to be installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "LiteLLMTokenMonitor": ("hyperion.adapters.litellm", "LiteLLMTokenMonitor"),
    "LiteLLMUsageCallback": ("hyperion.adapters.litellm", "LiteLLMUsageCallback"),
    "LiteLLMCallback": ("hyperion.adapters.litellm", "LiteLLMCallback"),
    "LangChainTokenMonitor": ("hyperion.adapters.langchain", "LangChainTokenMonitor"),
    "LangChainUsageCallback": ("hyperion.adapters.langchain", "LangChainUsageCallback"),
    "LangChainCallbackHandler": ("hyperion.adapters.langchain", "LangChainCallbackHandler"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module = import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
