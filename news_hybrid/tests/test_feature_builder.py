from datetime import date, timedelta

from news_hybrid import RateRecord, build_features
from news_hybrid.contracts import PRICE_FEATURES


def history():
    records = []
    currencies = ("AMD", "KGS", "KZT", "TJS", "UZS", "USD", "EUR", "CNY")
    start = date(2026, 1, 1)
    for offset in range(130):
        day = start + timedelta(days=offset)
        for index, currency in enumerate(currencies, start=1):
            nominal = 100.0 if currency in {"AMD", "KZT"} else 1.0
            rate = 0.1 * index + 0.0001 * offset
            records.append(RateRecord(currency, day, nominal, nominal * rate, rate))
    return records


def test_price_builder_returns_exact_model_schema():
    records = history()
    result = build_features(records, "AMD", records[-1].effective_date)
    assert tuple(result) == PRICE_FEATURES
    assert all(value is not None for value in result.values())
