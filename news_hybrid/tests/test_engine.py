from datetime import UTC, date, datetime, timedelta

import pytest

from news_hybrid import HybridEngine, HybridRequest, NewsEvent
from news_hybrid.contracts import PRICE_FEATURES

START = datetime(2026, 9, 7, 15, tzinfo=UTC)


class FakeModel:
    version = "test-model"

    @staticmethod
    def score(currency, price, news):
        return price[PRICE_FEATURES[0]]

    @staticmethod
    def threshold(currency):
        return 0.5


class FakeRegistry:
    @staticmethod
    def at(when):
        return FakeModel()


def request(number, score, *, at=None, events=True):
    instant = at or START + timedelta(days=number)
    news = (
        ()
        if not events
        else (
            NewsEvent(
                "event",
                START - timedelta(hours=1),
                "SANCTIONS",
                "Проверенная новость",
                ("https://example.test/source",),
                rub_applies=True,
            ),
        )
    )
    return HybridRequest(
        event_id=f"request-{number}",
        decision_at=instant,
        features_available_at=instant,
        currency="AMD",
        rub_per_unit=0.2194,
        effective_date=date(2026, 9, 8),
        price_features={**dict.fromkeys(PRICE_FEATURES, 0.0), PRICE_FEATURES[0]: score},
        news_events=news,
    )


def test_low_then_high_creates_complete_push_and_is_idempotent(tmp_path):
    with HybridEngine(tmp_path / "state.sqlite3", FakeRegistry()) as engine:
        assert engine.process(request(0, 0.1))["reason"] == "low_score_rearmed"
        high = request(1, 0.9)
        result = engine.process(high)
        assert result["status"] == "READY"
        assert result["news_event_keys"] == ["event"]
        assert result["notification"]["title"] == "Проверьте курс для перевода в Армению"
        assert "1 AMD = 0,2194 ₽" in result["notification"]["body"]
        assert engine.process(high) == result


def test_crossing_is_consumed_before_news_gate_and_weekly_cap(tmp_path):
    with HybridEngine(tmp_path / "state.sqlite3", FakeRegistry()) as engine:
        engine.process(request(0, 0.1))
        assert engine.process(request(1, 0.9, events=False))["reason"] == "new_episode_without_news"
        assert engine.process(request(2, 0.9))["reason"] == "no_new_high_episode"


def test_conflicting_id_and_backdated_request_are_rejected(tmp_path):
    with HybridEngine(tmp_path / "state.sqlite3", FakeRegistry()) as engine:
        original = request(0, 0.1)
        engine.process(original)
        with pytest.raises(ValueError, match="different inputs"):
            engine.process(request(0, 0.2))
        with pytest.raises(ValueError, match="chronological"):
            engine.process(request(3, 0.1, at=START - timedelta(days=1), events=False))
