"""Dependency-free inference for the exported L5_NEWS HistGradientBoosting models."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from importlib.resources import files
from pathlib import Path
from typing import Mapping

from .contracts import (
    CORRIDORS,
    MODEL_FEATURES,
    MOSCOW,
    NEWS_FEATURES,
    PRICE_FEATURES,
    digest,
    finite,
    utc,
)


class HybridPredictor:
    def __init__(self, artifact: dict):
        artifact = json.loads(json.dumps(artifact, allow_nan=False))
        checksum = artifact.pop("sha256", None)
        if checksum != digest(artifact):
            raise ValueError("model artifact checksum mismatch")
        if artifact.get("schema_version") != "alfa-news-hybrid.model.v1":
            raise ValueError("unsupported model artifact schema")
        if artifact.get("features") != list(MODEL_FEATURES):
            raise ValueError("model feature order mismatch")
        self.artifact_sha256 = checksum
        self.version = artifact["version"]
        self.valid_from = date.fromisoformat(artifact["valid_from"])
        self.valid_until = date.fromisoformat(artifact["valid_until"])
        if self.valid_until <= self.valid_from:
            raise ValueError("empty model validity interval")
        self._medians = tuple(finite(value, "median") for value in artifact["medians"])
        if len(self._medians) != len(MODEL_FEATURES):
            raise ValueError("wrong median vector length")
        self._thresholds = {
            key: finite(value, "threshold") for key, value in artifact["thresholds"].items()
        }
        if set(self._thresholds) != set(CORRIDORS):
            raise ValueError("corridor thresholds are incomplete")
        if any(not 0 <= value <= 1 for value in self._thresholds.values()):
            raise ValueError("threshold must be between zero and one")
        self._baseline = finite(artifact["baseline"], "baseline")
        self._trees = artifact["trees"]
        if not isinstance(self._trees, list) or not 1 <= len(self._trees) <= 256:
            raise ValueError("invalid tree collection")
        for tree in self._trees:
            self._validate_tree(tree)

    @staticmethod
    def _validate_tree(nodes: list) -> None:
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("invalid tree")
        visited: set[int] = set()

        def visit(index: int, depth: int) -> None:
            if type(index) is not int or not 0 <= index < len(nodes):
                raise ValueError("invalid tree child")
            if index in visited or depth > 16:
                raise ValueError("cyclic or excessively deep tree")
            visited.add(index)
            node = nodes[index]
            if type(node.get("is_leaf")) is not bool:
                raise ValueError("invalid leaf flag")
            finite(node["value"], "tree value")
            if node["is_leaf"]:
                return
            if type(node["feature_idx"]) is not int or not 0 <= node["feature_idx"] < len(
                MODEL_FEATURES
            ):
                raise ValueError("invalid feature index")
            finite(node["num_threshold"], "tree threshold")
            visit(node["left"], depth + 1)
            visit(node["right"], depth + 1)

        visit(0, 0)
        if len(visited) != len(nodes):
            raise ValueError("unreachable tree node")

    @classmethod
    def load(cls, path: str | Path) -> "HybridPredictor":
        source = Path(path)
        content = source.read_bytes()
        if len(content) > 2_000_000:
            raise ValueError("model artifact exceeds size limit")
        return cls(json.loads(content))

    def supports(self, when: datetime) -> bool:
        day = utc(when).astimezone(MOSCOW).date()
        return self.valid_from <= day < self.valid_until

    def threshold(self, currency: str) -> float:
        if currency not in CORRIDORS:
            raise ValueError(f"unsupported currency: {currency}")
        return self._thresholds[currency]

    def score(
        self, currency: str, price: Mapping[str, float | None], news: Mapping[str, float]
    ) -> float:
        if set(price) != set(PRICE_FEATURES) or set(news) != set(NEWS_FEATURES):
            raise ValueError("feature schema mismatch")
        values = [price[name] for name in PRICE_FEATURES]
        values.extend(float(currency == other) for other in CORRIDORS[1:])
        values.extend(news[name] for name in NEWS_FEATURES)
        design = [
            median if value is None else finite(value, name)
            for name, value, median in zip(MODEL_FEATURES, values, self._medians, strict=True)
        ]
        raw = self._baseline
        for nodes in self._trees:
            node = nodes[0]
            while not node["is_leaf"]:
                index = node["feature_idx"]
                node = nodes[
                    node["left"] if design[index] <= node["num_threshold"] else node["right"]
                ]
            raw += node["value"]
        return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, raw))))


class ModelRegistry:
    def __init__(self, paths=None):
        resources = files("news_hybrid").joinpath("artifacts")
        sources = (
            list(paths)
            if paths is not None
            else [
                item
                for item in resources.iterdir()
                if item.name.startswith("l5_news_") and item.name.endswith(".json")
            ]
        )
        self.models = sorted(
            (HybridPredictor.load(item) for item in sources), key=lambda model: model.valid_from
        )
        if not self.models:
            raise ValueError("no packaged hybrid models")
        for previous, current in zip(self.models, self.models[1:]):
            if current.valid_from < previous.valid_until:
                raise ValueError("overlapping model validity intervals")

    def at(self, when: datetime) -> HybridPredictor | None:
        return next((model for model in self.models if model.supports(when)), None)
