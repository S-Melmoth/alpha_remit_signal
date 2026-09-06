from datetime import UTC, date, datetime, timedelta

import pytest

from news_hybrid import HybridRequest, NewsEvent
from news_hybrid.contracts import PRICE_FEATURES


def valid_request(**changes):
    now = datetime(2026, 9, 7, 15, tzinfo=UTC)
    values = dict(
        event_id="id",
        decision_at=now,
        features_available_at=now,
        currency="AMD",
        rub_per_unit=0.2194,
        effective_date=date(2026, 9, 8),
        price_features=dict.fromkeys(PRICE_FEATURES, 0.0),
        news_events=(),
    )
    values.update(changes)
    return HybridRequest(**values)


def test_request_rejects_unknown_features_and_future_inputs():
    with pytest.raises(ValueError, match="schema"):
        valid_request(price_features={})
    with pytest.raises(ValueError, match="not available"):
        now = datetime(2026, 9, 7, 15, tzinfo=UTC)
        valid_request(features_available_at=now + timedelta(seconds=1))


def test_event_requires_explicit_scope_and_verified_action():
    now = datetime(2026, 9, 7, 15, tzinfo=UTC)
    with pytest.raises(ValueError, match="scope"):
        NewsEvent("key", now, "SANCTIONS", "Заголовок")
    with pytest.raises(ValueError, match="requires"):
        NewsEvent("key", now, "CENTRAL_BANK_DECISION", "Заголовок", rub_action="raise")
