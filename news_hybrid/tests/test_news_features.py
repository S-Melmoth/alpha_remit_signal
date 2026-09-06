from datetime import UTC, datetime, timedelta

import pytest

from news_hybrid import NewsEvent
from news_hybrid.news_features import aggregate_news

NOW = datetime(2026, 9, 7, 15, tzinfo=UTC)


def test_news19_uses_only_relevant_visible_independent_events():
    events = [
        NewsEvent(
            "rub",
            NOW - timedelta(hours=4),
            "CENTRAL_BANK_DECISION",
            "Банк России",
            rub_applies=True,
            rub_action="raise",
        ),
        NewsEvent(
            "amd",
            NOW - timedelta(hours=2),
            "CENTRAL_BANK_DECISION",
            "ЦБ Армении",
            recipient_currencies=("AMD",),
            recipient_actions={"AMD": "hold"},
        ),
        NewsEvent(
            "other",
            NOW - timedelta(hours=1),
            "SANCTIONS",
            "Только Узбекистан",
            recipient_currencies=("UZS",),
        ),
        NewsEvent(
            "old", NOW - timedelta(days=14), "OIL_SUPPLY", "Старая граница", global_applies=True
        ),
    ]
    values, visible = aggregate_news(events, "AMD", NOW)
    assert [event.key for event in visible] == ["rub", "amd"]
    assert values["news_count"] == 2
    assert values["news_cb_decision_count"] == 2
    assert values["news_rub_raise_count"] == 1
    assert values["news_recipient_hold_count"] == 1
    assert values["news_pair_decision_stance_sum"] == -1
    assert values["news_latest_age_hours"] == 2


def test_no_news_vector_and_future_guard():
    values, visible = aggregate_news([], "AMD", NOW)
    assert not visible and values["news_latest_age_hours"] == 336
    assert sum(value for name, value in values.items() if name != "news_latest_age_hours") == 0
    future = NewsEvent(
        "future", NOW + timedelta(seconds=1), "SANCTIONS", "Будущее", rub_applies=True
    )
    with pytest.raises(ValueError, match="future"):
        aggregate_news([future], "AMD", NOW)
