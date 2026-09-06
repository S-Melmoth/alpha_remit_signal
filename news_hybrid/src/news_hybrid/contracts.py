"""Validated public input and output contracts for one hybrid decision."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence

CORRIDORS = ("AMD", "KGS", "KZT", "TJS", "UZS")
PRICE_FEATURES = (
    "corridor_move_1_bps",
    "corridor_move_5_bps",
    "corridor_move_20_bps",
    "corridor_level_percentile_20",
    "corridor_level_percentile_120",
    "corridor_volatility_20_bps",
    "corridor_reversal_score_3_bps",
    "rub_common_move_1_bps",
    "rub_common_move_5_bps",
    "rub_common_move_20_bps",
    "days_since_prior_publication",
    "publication_is_friday",
    "publication_month_sin",
    "publication_month_cos",
)
CORRIDOR_FEATURES = tuple(f"corridor__{currency}" for currency in CORRIDORS[1:])
NEWS_FEATURES = (
    "news_count",
    "news_cb_decision_count",
    "news_expectation_count",
    "news_sanctions_count",
    "news_oil_count",
    "news_fx_flow_count",
    "news_rub_count",
    "news_recipient_count",
    "news_global_count",
    "news_rub_raise_count",
    "news_rub_hold_count",
    "news_rub_cut_count",
    "news_recipient_raise_count",
    "news_recipient_hold_count",
    "news_recipient_cut_count",
    "news_unknown_decision_action_count",
    "news_pair_decision_stance_sum",
    "news_pair_conflict_present",
    "news_latest_age_hours",
)
MODEL_FEATURES = (*PRICE_FEATURES, *CORRIDOR_FEATURES, *NEWS_FEATURES)
EVENT_TYPES = (
    "CENTRAL_BANK_DECISION",
    "RATE_EXPECTATION",
    "SANCTIONS",
    "OIL_SUPPLY",
    "FX_FLOW_ANNOUNCEMENT",
)
ACTIONS = ("raise", "hold", "cut")
MOSCOW = timezone(timedelta(hours=3), name="Europe/Moscow")


def utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)


def timestamp(value: datetime) -> str:
    return utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid ISO timestamp") from exc
    return utc(parsed)


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or value != value.strip() or not value or len(value) > limit:
        raise ValueError(f"invalid {name}")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"control character in {name}")
    return value


def _action(value: str | None, name: str) -> str | None:
    if value is not None and value not in ACTIONS:
        raise ValueError(f"{name} must be one of {ACTIONS} or null")
    return value


@dataclass(frozen=True)
class NewsEvent:
    """One independently identified event, using facts known at ``available_at``."""

    key: str
    available_at: datetime
    event_type: str
    headline: str
    source_urls: tuple[str, ...] = ()
    rub_applies: bool = False
    rub_action: str | None = None
    recipient_currencies: tuple[str, ...] = ()
    recipient_actions: Mapping[str, str | None] = field(default_factory=dict)
    global_applies: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _text(self.key, "event key", 256))
        object.__setattr__(self, "available_at", utc(self.available_at))
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"event_type must be one of {EVENT_TYPES}")
        object.__setattr__(self, "headline", _text(self.headline, "headline", 4000))
        urls = tuple(_text(value, "source URL", 4096) for value in self.source_urls)
        if any(not value.startswith(("https://", "http://")) for value in urls):
            raise ValueError("source URLs must use HTTP(S)")
        object.__setattr__(self, "source_urls", urls)
        if type(self.rub_applies) is not bool or type(self.global_applies) is not bool:
            raise ValueError("applicability flags must be booleans")
        object.__setattr__(self, "rub_action", _action(self.rub_action, "rub_action"))
        recipients = tuple(sorted(set(self.recipient_currencies)))
        if any(value not in CORRIDORS for value in recipients):
            raise ValueError("recipient_currencies contains an unsupported currency")
        actions = dict(self.recipient_actions)
        if set(actions) - set(recipients):
            raise ValueError("recipient_actions must be a subset of recipient_currencies")
        actions = {
            name: _action(value, f"recipient_actions[{name}]") for name, value in actions.items()
        }
        object.__setattr__(self, "recipient_currencies", recipients)
        object.__setattr__(self, "recipient_actions", MappingProxyType(actions))
        if self.rub_action is not None and not self.rub_applies:
            raise ValueError("rub_action requires rub_applies=true")
        if not (self.rub_applies or self.global_applies or recipients):
            raise ValueError("news event has no relevant scope")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NewsEvent":
        return cls(
            key=value["key"],
            available_at=parse_timestamp(value["available_at"]),
            event_type=value["event_type"],
            headline=value["headline"],
            source_urls=tuple(value.get("source_urls", ())),
            rub_applies=value.get("rub_applies", False),
            rub_action=value.get("rub_action"),
            recipient_currencies=tuple(value.get("recipient_currencies", ())),
            recipient_actions=value.get("recipient_actions", {}),
            global_applies=value.get("global_applies", False),
        )

    def relevant_to(self, currency: str) -> bool:
        return self.rub_applies or self.global_applies or currency in self.recipient_currencies

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "available_at": timestamp(self.available_at),
            "event_type": self.event_type,
            "headline": self.headline,
            "source_urls": list(self.source_urls),
            "rub_applies": self.rub_applies,
            "rub_action": self.rub_action,
            "recipient_currencies": list(self.recipient_currencies),
            "recipient_actions": dict(self.recipient_actions),
            "global_applies": self.global_applies,
        }


@dataclass(frozen=True)
class HybridRequest:
    event_id: str
    decision_at: datetime
    features_available_at: datetime
    currency: str
    rub_per_unit: float
    effective_date: date
    price_features: Mapping[str, float | None]
    news_events: Sequence[NewsEvent]

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _text(self.event_id, "event_id", 160))
        object.__setattr__(self, "decision_at", utc(self.decision_at))
        object.__setattr__(self, "features_available_at", utc(self.features_available_at))
        if self.features_available_at > self.decision_at:
            raise ValueError("price features were not available at decision_at")
        if self.currency not in CORRIDORS:
            raise ValueError(f"unsupported currency: {self.currency}")
        rate = finite(self.rub_per_unit, "rub_per_unit")
        if not 0.0001 <= rate <= 50:
            raise ValueError("rub_per_unit is outside supported CBR bounds")
        object.__setattr__(self, "rub_per_unit", rate)
        if type(self.effective_date) is not date:
            raise ValueError("effective_date must be a date")
        if set(self.price_features) != set(PRICE_FEATURES):
            missing = sorted(set(PRICE_FEATURES) - set(self.price_features))
            extra = sorted(set(self.price_features) - set(PRICE_FEATURES))
            raise ValueError(f"price feature schema mismatch: missing={missing}, extra={extra}")
        normalized = {
            name: None
            if self.price_features[name] is None
            else finite(self.price_features[name], name)
            for name in PRICE_FEATURES
        }
        object.__setattr__(self, "price_features", MappingProxyType(normalized))
        events = tuple(self.news_events)
        if any(not isinstance(event, NewsEvent) for event in events):
            raise ValueError("news_events must contain NewsEvent values")
        if any(event.available_at > self.decision_at for event in events):
            raise ValueError("future news is not allowed")
        object.__setattr__(self, "news_events", events)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HybridRequest":
        return cls(
            event_id=value["event_id"],
            decision_at=parse_timestamp(value["decision_at"]),
            features_available_at=parse_timestamp(value["features_available_at"]),
            currency=value["currency"],
            rub_per_unit=value["rub_per_unit"],
            effective_date=date.fromisoformat(value["effective_date"]),
            price_features=value["price_features"],
            news_events=tuple(NewsEvent.from_dict(item) for item in value.get("news_events", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "decision_at": timestamp(self.decision_at),
            "features_available_at": timestamp(self.features_available_at),
            "currency": self.currency,
            "rub_per_unit": self.rub_per_unit,
            "effective_date": self.effective_date.isoformat(),
            "price_features": dict(self.price_features),
            "news_events": [event.to_dict() for event in self.news_events],
        }
