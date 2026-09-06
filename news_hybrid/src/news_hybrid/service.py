"""Small adapter used by Python backends and the JSON command-line interface."""

from pathlib import Path
from typing import Any, Mapping

from .contracts import HybridRequest
from .engine import HybridEngine


def decide(payload: Mapping[str, Any], *, state_path: str | Path) -> dict:
    request = HybridRequest.from_dict(payload)
    with HybridEngine(state_path) as engine:
        return engine.process(request)
