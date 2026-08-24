"""Immutable, provider-neutral Hyperion runtime configuration."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from hyperion.protocols import UsageEventSink
from hyperion.sinks.logging import JsonLoggingSink

ProviderResolver: TypeAlias = Callable[..., str | None]


def _env_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true/false, 1/0, yes/no, or on/off")


def _resolve_bool(name: str, explicit: bool | None, default: bool) -> bool:
    if explicit is not None:
        if not isinstance(explicit, bool):
            raise TypeError(f"{name.lower()} must be bool or None")
        return explicit
    value = os.getenv(name)
    if value is None:
        return default
    return _env_bool(name, value)


def _env_integrations() -> tuple[str, ...]:
    raw = os.getenv("HYPERION_INTEGRATIONS")
    if raw is None or not raw.strip():
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _normalize_integrations(value: Sequence[str] | str | bytes) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("integrations must be a sequence of names, not str or bytes")

    normalized: list[str] = []
    seen: set[str] = set()
    for name in value:
        if not isinstance(name, str):
            raise TypeError("integrations must contain non-empty strings")
        name = name.strip()
        if not name:
            raise TypeError("integrations must contain non-empty strings")
        if name not in seen:
            seen.add(name)
            normalized.append(name)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class HyperionConfig:
    """Resolved configuration for a Hyperion installation.

    The ``HYPERION_*`` environment variables provide defaults when a constructor
    field is ``None``.  ``ENABLE_TOKEN_MONITORING`` is accepted only as a legacy
    fallback for ``enabled`` when ``HYPERION_ENABLED`` is absent.  ``sinks=None``
    selects one :class:`JsonLoggingSink`; ``sinks=()`` explicitly selects no sinks.
    Integrations are always explicit and are never inferred from installed packages.
    """

    enabled: bool | None = None
    integrations: Sequence[str] | None = None
    sinks: Sequence[UsageEventSink] | None = None
    details_max_bytes: int | None = None
    diagnostic_logging: bool | None = None
    default_call_site: str | None = None
    provider_resolver: ProviderResolver | None = None
    provider_inference_hook: ProviderResolver | None = None
    diagnostic_logger: Any | None = None

    def __post_init__(self) -> None:
        if self.provider_resolver is not None and self.provider_inference_hook is not None:
            raise ValueError("provider_resolver and provider_inference_hook are mutually exclusive")
        if self.provider_resolver is None:
            object.__setattr__(self, "provider_resolver", self.provider_inference_hook)

        explicit_enabled = self.enabled
        if explicit_enabled is None:
            env_name = "HYPERION_ENABLED"
            env_value = os.getenv(env_name)
            if env_value is None:
                env_value = os.getenv("ENABLE_TOKEN_MONITORING")
                env_name = "ENABLE_TOKEN_MONITORING"
            enabled = True if env_value is None else _env_bool(env_name, env_value)
        else:
            enabled = _resolve_bool("HYPERION_ENABLED", explicit_enabled, True)
        object.__setattr__(self, "enabled", enabled)

        integrations = _env_integrations() if self.integrations is None else self.integrations
        object.__setattr__(self, "integrations", _normalize_integrations(integrations))

        if self.sinks is None:
            logger = self.diagnostic_logger
            object.__setattr__(self, "sinks", (JsonLoggingSink(logger),))
        else:
            if isinstance(self.sinks, (str, bytes)):
                raise TypeError("sinks must be a sequence of sink objects, not str or bytes")
            sinks = tuple(self.sinks)
            if any(not callable(getattr(sink, "emit", None)) for sink in sinks):
                raise TypeError("sinks must contain objects with a callable emit method")
            object.__setattr__(self, "sinks", sinks)

        details_max_bytes = self.details_max_bytes
        if details_max_bytes is None:
            raw_limit = os.getenv("HYPERION_DETAILS_MAX_BYTES")
            details_max_bytes = 16_384 if raw_limit is None else _parse_limit(raw_limit)
        elif not isinstance(details_max_bytes, int) or isinstance(details_max_bytes, bool):
            raise TypeError("details_max_bytes must be int or None")
        if details_max_bytes < 0:
            raise ValueError("details_max_bytes must be zero or greater")
        object.__setattr__(self, "details_max_bytes", details_max_bytes)

        object.__setattr__(
            self,
            "diagnostic_logging",
            _resolve_bool("HYPERION_DIAGNOSTIC_LOGGING", self.diagnostic_logging, True),
        )
        default_call_site = self.default_call_site
        if default_call_site is None:
            default_call_site = os.getenv("HYPERION_DEFAULT_CALL_SITE", "application")
        if not isinstance(default_call_site, str):
            raise TypeError("default_call_site must be str or None")
        object.__setattr__(self, "default_call_site", default_call_site)

    @classmethod
    def from_env(cls) -> HyperionConfig:
        """Build a configuration entirely from environment defaults."""

        return cls()

    @classmethod
    def from_environment(cls) -> HyperionConfig:
        """Alias for :meth:`from_env` used by applications that prefer explicit wording."""

        return cls.from_env()


def _parse_limit(value: str) -> int:
    try:
        parsed = int(value.strip(), 10)
    except ValueError as exc:
        raise ValueError("HYPERION_DETAILS_MAX_BYTES must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError("HYPERION_DETAILS_MAX_BYTES must be a non-negative integer")
    return parsed


__all__ = ["HyperionConfig", "ProviderResolver"]
