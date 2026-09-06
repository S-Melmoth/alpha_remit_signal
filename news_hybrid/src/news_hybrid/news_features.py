"""Causal 14-day aggregation of normalized news into the frozen news19 schema."""

from __future__ import annotations

from datetime import timedelta
from typing import Sequence

from .contracts import NEWS_FEATURES, NewsEvent, utc

NEWS_WINDOW = timedelta(days=14)
TYPE_FEATURE = {
    "CENTRAL_BANK_DECISION": "news_cb_decision_count",
    "RATE_EXPECTATION": "news_expectation_count",
    "SANCTIONS": "news_sanctions_count",
    "OIL_SUPPLY": "news_oil_count",
    "FX_FLOW_ANNOUNCEMENT": "news_fx_flow_count",
}
ACTION_CODE = {"raise": 1, "hold": 0, "cut": -1}


def aggregate_news(events: Sequence[NewsEvent], currency: str, decision_at) -> tuple[dict, tuple]:
    """Return news19 values and the exact relevant events in ``(T-14d, T]``."""

    decision_at = utc(decision_at)
    selected: dict[str, NewsEvent] = {}
    for event in events:
        if not isinstance(event, NewsEvent):
            raise ValueError("events must contain NewsEvent values")
        if event.available_at > decision_at:
            raise ValueError("future news is not allowed")
        if not decision_at - NEWS_WINDOW < event.available_at <= decision_at:
            continue
        if not event.relevant_to(currency):
            continue
        previous = selected.get(event.key)
        if previous is not None and previous != event:
            raise ValueError(f"conflicting duplicate news event: {event.key}")
        selected[event.key] = event
    visible = tuple(sorted(selected.values(), key=lambda item: (item.available_at, item.key)))
    values = dict.fromkeys(NEWS_FEATURES, 0.0)
    values["news_latest_age_hours"] = 336.0
    signs: set[int] = set()
    for event in visible:
        values["news_count"] += 1.0
        values[TYPE_FEATURE[event.event_type]] += 1.0
        values["news_rub_count"] += float(event.rub_applies)
        recipient_applies = currency in event.recipient_currencies
        values["news_recipient_count"] += float(recipient_applies)
        values["news_global_count"] += float(event.global_applies)
        stance = 0
        legs = (
            ("rub", event.rub_applies, event.rub_action, -1),
            ("recipient", recipient_applies, event.recipient_actions.get(currency), 1),
        )
        for name, applies, action, sign in legs:
            if applies and action is not None:
                values[f"news_{name}_{action}_count"] += 1.0
                stance += sign * ACTION_CODE[action]
            elif applies and event.event_type == "CENTRAL_BANK_DECISION":
                values["news_unknown_decision_action_count"] += 1.0
        values["news_pair_decision_stance_sum"] += float(stance)
        if stance:
            signs.add(1 if stance > 0 else -1)
        age = (decision_at - event.available_at).total_seconds() / 3600
        values["news_latest_age_hours"] = min(values["news_latest_age_hours"], age)
    values["news_pair_conflict_present"] = float(len(signs) == 2)
    return values, visible
