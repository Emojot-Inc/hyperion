"""Explicit integration activation and installation lifecycle."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from threading import RLock
from typing import Any

from hyperion.config import HyperionConfig
from hyperion.monitor import UsageMonitor

# This is the complete Stage 3 activation allowlist.  Adding an integration requires
# an intentional source change here; user-provided names are never imported as modules.
INTEGRATION_MODULES: dict[str, str] = {
    "litellm": "hyperion.adapters.litellm",
    "langchain": "hyperion.adapters.langchain",
}


class Installation:
    """Handle returned by :func:`install`.

    Adapter registration objects may expose ``uninstall()`` (preferred),
    ``shutdown()``, or be directly callable as a teardown callback.  Teardown is
    attempted at most once even when either lifecycle method is called repeatedly.
    """

    def __init__(
        self,
        config: HyperionConfig,
        monitor: UsageMonitor,
        registrations: tuple[Any, ...],
    ) -> None:
        self.config = config
        self.monitor = monitor
        self._registrations = registrations
        self._closed = False
        self._lock = RLock()

    def shutdown(self) -> None:
        """Tear down active adapter registrations; safe to call repeatedly."""

        self._teardown()

    def uninstall(self) -> None:
        """Remove active adapter hooks; safe to call repeatedly."""

        self._teardown()

    @property
    def active(self) -> bool:
        with self._lock:
            return not self._closed

    def _teardown(self) -> None:
        global _ACTIVE_INSTALLATION
        with _INSTALL_LOCK:
            # Keep the global installation transition serialized for the entire
            # teardown.  Otherwise a concurrent install can register new hooks
            # while the old registration is still being removed.
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                registrations = self._registrations
                self._registrations = ()
            for registration in reversed(registrations):
                _teardown_registration(registration, self.config)
            if _ACTIVE_INSTALLATION is self:
                _ACTIVE_INSTALLATION = None


_INSTALL_LOCK = RLock()
_ACTIVE_INSTALLATION: Installation | None = None


def install(config: HyperionConfig | None = None) -> Installation:
    """Install explicitly configured integrations and return an idempotent handle.

    Calling this function does no global patching by itself.  Only allowlisted,
    explicitly named adapters are imported.  Adapter modules implement the Stage 3
    contract ``register(monitor, config)`` and return a teardown callback or object.
    Missing optional dependencies and adapter failures are reported diagnostically
    (without exception text) and leave the installation usable.
    """

    global _ACTIVE_INSTALLATION
    with _INSTALL_LOCK:
        # A non-None handle also represents a teardown in progress.  Keep it
        # authoritative until its hooks have been removed and the transition
        # clears the active slot below.
        if _ACTIVE_INSTALLATION is not None:
            return _ACTIVE_INSTALLATION
        resolved = HyperionConfig.from_env() if config is None else _coerce_config(config)

        monitor = UsageMonitor(
            () if not resolved.enabled else resolved.sinks,
            logger=resolved.diagnostic_logger,
            diagnostic_logging=bool(resolved.diagnostic_logging),
            enabled=resolved.enabled,
        )
        registrations: list[Any] = []
        if resolved.enabled:
            for integration in resolved.integrations:
                registration = _activate_integration(integration, monitor, resolved)
                if registration is not None:
                    registrations.append(registration)
        handle = Installation(resolved, monitor, tuple(registrations))
        _ACTIVE_INSTALLATION = handle
        return handle


def _coerce_config(config: HyperionConfig) -> HyperionConfig:
    if not isinstance(config, HyperionConfig):
        raise TypeError("config must be HyperionConfig or None")
    return config


def _activate_integration(
    integration: str,
    monitor: UsageMonitor,
    config: HyperionConfig,
) -> Any | None:
    module_name = INTEGRATION_MODULES.get(integration)
    if module_name is None:
        _diagnose(config, "Hyperion integration is not allowlisted", integration=integration)
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - optional integrations fail open
        _diagnose(
            config,
            "Hyperion integration could not be loaded",
            integration=integration,
            exception_type=type(exc).__name__,
        )
        return None
    register = getattr(module, "register", None)
    if not callable(register):
        _diagnose(config, "Hyperion integration has no register function", integration=integration)
        return None
    try:
        return register(monitor, config)
    except Exception as exc:  # noqa: BLE001 - adapter startup fails open
        _diagnose(
            config,
            "Hyperion integration registration failed",
            integration=integration,
            exception_type=type(exc).__name__,
        )
        return None


def _teardown_registration(registration: Any, config: HyperionConfig) -> None:
    try:
        teardown: Callable[[], Any] | None = None
        uninstall = getattr(registration, "uninstall", None)
        shutdown = getattr(registration, "shutdown", None)
        if callable(uninstall):
            teardown = uninstall
        elif callable(shutdown):
            teardown = shutdown
        elif callable(registration):
            teardown = registration
        if teardown is not None:
            teardown()
    except Exception as exc:  # noqa: BLE001 - teardown must be fail-open
        _diagnose(config, "Hyperion integration teardown failed", exception_type=type(exc).__name__)


def _diagnose(config: HyperionConfig, message: str, **fields: str) -> None:
    if not config.diagnostic_logging:
        return
    logger = config.diagnostic_logger or logging.getLogger("hyperion")
    try:
        logger.warning(message, extra=fields)
    except Exception:  # noqa: BLE001 - diagnostics must never block startup/calls
        return


__all__ = ["INTEGRATION_MODULES", "Installation", "install"]
