"""Structured JSON logging sink for normalized Hyperion events."""

from __future__ import annotations

import json
import logging
from typing import Any

from hyperion.events import UsageEvent


class JsonLoggingSink:
    """Emit exactly one compact JSON log record for each usage event.

    The payload comes exclusively from :meth:`UsageEvent.to_dict`; prompts,
    completions and tool arguments are not part of that normalized contract.
    """

    def __init__(self, logger: Any | None = None, *, level: int = logging.INFO) -> None:
        self._logger = logging.getLogger("hyperion") if logger is None else logger
        if not isinstance(level, int) or isinstance(level, bool):
            raise TypeError("level must be int")
        self._level = level

    @property
    def logger(self) -> Any:
        return self._logger

    def emit(self, event: UsageEvent) -> None:
        payload = json.dumps(
            event.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        log = getattr(self._logger, "log", None)
        if callable(log):
            log(self._level, payload)
        else:
            self._logger.info(payload)


# A short, discoverable name for applications configuring a logging sink directly.
LoggingSink = JsonLoggingSink

__all__ = ["JsonLoggingSink", "LoggingSink"]
