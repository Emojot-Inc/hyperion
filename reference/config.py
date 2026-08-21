"""Minimal configuration seam for the reference monitoring snapshot."""

import os
from types import SimpleNamespace

settings = SimpleNamespace(
    ENABLE_TOKEN_MONITORING=os.getenv("ENABLE_TOKEN_MONITORING", "TRUE").upper() == "TRUE",
)
