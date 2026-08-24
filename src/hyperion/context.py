"""Request-scoped monitoring context.

``context_scope`` overlays only the fields provided by the caller; omitted values inherit
from the active context. ``None`` is an explicit clear value, and empty strings are
preserved as explicit values.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType

_UNSET = object()


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Immutable request metadata carried across instrumentation boundaries.

    ``request_id``, ``session_id`` and ``call_site`` must be ``str`` or ``None``.
    ``metadata`` accepts a mapping of string keys and string values, is defensively copied
    at construction, and is exposed as an immutable mapping.
    """

    request_id: str | None = None
    session_id: str | None = None
    call_site: str | None = None
    metadata: Mapping[str, str] | None = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _coerce_context_value(self.request_id, "request_id"))
        object.__setattr__(self, "session_id", _coerce_context_value(self.session_id, "session_id"))
        object.__setattr__(self, "call_site", _coerce_context_value(self.call_site, "call_site"))
        object.__setattr__(self, "metadata", MappingProxyType(dict(_coerce_metadata(self.metadata))))

    def overlay(
        self,
        *,
        request_id: str | None | object = _UNSET,
        session_id: str | None | object = _UNSET,
        call_site: str | None | object = _UNSET,
        metadata: Mapping[str, str] | None | object = _UNSET,
    ) -> RequestContext:
        """Return a context with provided fields replaced and omitted fields inherited."""

        return RequestContext(
            request_id=self.request_id
            if request_id is _UNSET
            else _coerce_context_value(request_id, "request_id"),
            session_id=self.session_id
            if session_id is _UNSET
            else _coerce_context_value(session_id, "session_id"),
            call_site=self.call_site
            if call_site is _UNSET
            else _coerce_context_value(call_site, "call_site"),
            metadata=self.metadata if metadata is _UNSET else _coerce_metadata(metadata),
        )


_CURRENT_CONTEXT: ContextVar[RequestContext | None] = ContextVar(
    "hyperion_request_context",
    default=None,
)


def current_context() -> RequestContext:
    """Return the current immutable request context."""

    return _CURRENT_CONTEXT.get() or _EMPTY_CONTEXT


@contextmanager
def context_scope(
    *,
    request_id: str | None | object = _UNSET,
    session_id: str | None | object = _UNSET,
    call_site: str | None | object = _UNSET,
    metadata: Mapping[str, str] | None | object = _UNSET,
) -> Iterator[RequestContext]:
    """Temporarily overlay request context and restore the exact prior context."""

    next_context = current_context().overlay(
        request_id=request_id,
        session_id=session_id,
        call_site=call_site,
        metadata=metadata,
    )
    token = _CURRENT_CONTEXT.set(next_context)
    try:
        yield next_context
    finally:
        _CURRENT_CONTEXT.reset(token)


def _coerce_context_value(value: str | None | object, name: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"{name} must be str or None")


def _coerce_metadata(value: Mapping[str, str] | None | object) -> Mapping[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping or None")
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("metadata keys must be str")
        if not isinstance(item, str):
            raise TypeError("metadata values must be str")
    return value


_EMPTY_CONTEXT = RequestContext()


__all__ = ["RequestContext", "context_scope", "current_context"]
