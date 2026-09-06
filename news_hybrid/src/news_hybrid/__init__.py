"""Public API for the Alfa news and FX-history signal module."""

from .cbr import CBRClient, RateRecord
from .contracts import HybridRequest, NewsEvent
from .engine import HybridEngine
from .feature_builder import build_features
from .predictor import HybridPredictor, ModelRegistry
from .service import decide

__all__ = [
    "CBRClient",
    "HybridEngine",
    "HybridPredictor",
    "HybridRequest",
    "ModelRegistry",
    "NewsEvent",
    "RateRecord",
    "build_features",
    "decide",
]
