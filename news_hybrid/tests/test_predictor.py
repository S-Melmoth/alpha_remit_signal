import json
from importlib.resources import files
from pathlib import Path

import pytest

from news_hybrid.predictor import HybridPredictor, ModelRegistry

CASES = json.loads((Path(__file__).parent / "parity_cases.json").read_text(encoding="utf-8"))


def test_all_walk_forward_models_are_packaged_without_gaps():
    registry = ModelRegistry()
    assert len(registry.models) == 8
    assert registry.models[0].valid_from.isoformat() == "2023-01-01"
    assert registry.models[-1].valid_until.isoformat() == "2027-01-01"
    for previous, current in zip(registry.models, registry.models[1:]):
        assert previous.valid_until == current.valid_from


@pytest.mark.parametrize("case", CASES, ids=lambda value: f"{value['fold']}-{value['currency']}")
def test_dependency_free_export_matches_frozen_sklearn_score(case):
    source = files("news_hybrid").joinpath("artifacts", f"l5_news_{case['fold']}.json")
    model = HybridPredictor.load(source)
    score = model.score(case["currency"], case["price_features"], case["news_features"])
    assert score == pytest.approx(case["expected_score"], abs=1e-12, rel=0)


def test_artifact_checksum_rejects_mutation():
    source = files("news_hybrid").joinpath("artifacts", "l5_news_2026_h2.json")
    artifact = json.loads(source.read_text(encoding="utf-8"))
    artifact["thresholds"]["AMD"] += 0.01
    with pytest.raises(ValueError, match="checksum"):
        HybridPredictor(artifact)
