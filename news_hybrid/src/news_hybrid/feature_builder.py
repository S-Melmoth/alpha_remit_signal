"""Build the serving-time B1 base14 features from normalized CBR events.

The formulas mirror the corresponding columns in ``alfa_features_v3.py`` but
this runtime module deliberately does not import the checksum-bound research
code, pandas, or numpy. All histories are event histories: missing calendar
days are never forward-filled into synthetic observations.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from datetime import date, timedelta

from .cbr import CONTROL_CURRENCIES, TARGET_CURRENCIES, RateRecord
from .contracts import PRICE_FEATURES as FEATURES

TARGET_HISTORY_REQUIRED = 120
CONTROL_HISTORY_REQUIRED = 21  # current event plus 20 prior control events


class InsufficientHistory(ValueError):
    """A required series is present at the event but its trailing window is short."""


class IncompleteSources(ValueError):
    """The target or a synchronous USD/EUR/CNY event is absent."""


def _move(log_rates: Sequence[float], window: int) -> float:
    # alfa_features_v3._base_rate_features: 1e4 * log_rate.diff(window)
    return 1e4 * (log_rates[-1] - log_rates[-1 - window])


def _percentile(rates: Sequence[float], window: int) -> float:
    # pandas rolling.rank(method="max", pct=True): share of values <= current.
    trailing = rates[-window:]
    current = trailing[-1]
    return sum(value <= current for value in trailing) / window


def _rolling_population_std(values: Sequence[float], window: int) -> float:
    """Reproduce pandas' sliding Welford/Kahan ``rolling.std(ddof=0)``.

    B1's artifact records pandas 3.0.5, and the research store was serialized
    only after this rolling calculation. Recomputing the final window in
    isolation is mathematically equivalent but can cross the twelfth-digit
    serialization boundary, so the add/remove order is part of inference
    compatibility. This is a frozen numerical routine, not a pandas runtime
    dependency.
    """

    if window <= 0 or len(values) < window:
        raise ValueError("rolling population standard deviation needs a full window")
    inverse_condition_tolerance = sys.float_info.epsilon * 1e3
    nobs = mean = squared_deviations = 0.0
    compensation_add = compensation_remove = 0.0
    numerically_unstable = False
    previous_start = previous_end = 0

    def add(value: float) -> None:
        nonlocal nobs, mean, squared_deviations, compensation_add
        nonlocal numerically_unstable
        if math.isnan(value):
            return
        previous_m2 = squared_deviations
        nobs += 1.0
        previous_mean = mean - compensation_add
        adjusted = value - compensation_add
        delta = adjusted - mean
        compensation_add = delta + mean - adjusted
        mean += delta / nobs
        squared_deviations += (value - previous_mean) * (value - mean)
        if previous_m2 * inverse_condition_tolerance > squared_deviations:
            numerically_unstable = True

    def remove(value: float) -> None:
        nonlocal nobs, mean, squared_deviations, compensation_remove
        nonlocal numerically_unstable
        if math.isnan(value):
            return
        previous_m2 = squared_deviations
        nobs -= 1.0
        if nobs:
            previous_mean = mean - compensation_remove
            adjusted = value - compensation_remove
            delta = adjusted - mean
            compensation_remove = delta + mean - adjusted
            mean -= delta / nobs
            squared_deviations -= (value - previous_mean) * (value - mean)
            if previous_m2 * inverse_condition_tolerance > squared_deviations:
                numerically_unstable = True
        else:
            mean = squared_deviations = 0.0
            numerically_unstable = False

    for index in range(len(values)):
        start = max(0, index - window + 1)
        end = index + 1
        requires_recompute = index == 0 or start >= previous_end
        if not requires_recompute:
            for position in range(previous_start, start):
                remove(values[position])
            for position in range(previous_end, end):
                add(values[position])
        if requires_recompute or numerically_unstable:
            nobs = mean = squared_deviations = 0.0
            compensation_add = compensation_remove = 0.0
            for position in range(start, end):
                add(values[position])
            numerically_unstable = False
        previous_start, previous_end = start, end

    if nobs < window:
        raise ValueError("rolling population standard deviation has missing observations")
    variance = squared_deviations / nobs
    if variance < 0:
        raise ValueError("rolling population variance became negative")
    return math.sqrt(variance)


def _serialize_12g(value: float | None, name: str) -> float | None:
    """Apply the historical feature-store ``float_format='%.12g'`` boundary."""

    if value is None:
        return None
    if not math.isfinite(value):
        raise ValueError(f"Non-finite derived feature: {name}")
    return float(format(value, ".12g"))


def _validated_histories(
    records: Sequence[RateRecord], cutoff: date
) -> dict[str, list[RateRecord]]:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence of RateRecord values")
    histories: dict[str, list[RateRecord]] = {}
    seen: set[tuple[str, date]] = set()
    for record in records:
        if not isinstance(record, RateRecord):
            raise ValueError("records must contain only RateRecord values")
        numbers = (record.nominal, record.value_rub, record.rub_per_unit)
        if (
            any(not math.isfinite(value) or value <= 0 for value in numbers)
            or not record.nominal.is_integer()
            or not math.isclose(
                record.value_rub / record.nominal,
                record.rub_per_unit,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"Invalid normalized rate record for {record.currency} on {record.effective_date}"
            )
        key = (record.currency, record.effective_date)
        if key in seen:
            raise ValueError(
                f"Duplicate rate record for {record.currency} on {record.effective_date}"
            )
        seen.add(key)
        if record.effective_date <= cutoff:
            histories.setdefault(record.currency, []).append(record)
    for history in histories.values():
        history.sort(key=lambda item: item.effective_date)
    return histories


def _current_history(
    histories: dict[str, list[RateRecord]], currency: str, effective_date: date
) -> list[RateRecord]:
    history = histories.get(currency, [])
    if not history or history[-1].effective_date != effective_date:
        raise IncompleteSources(f"Missing synchronous {currency} quotation for {effective_date}")
    return history


def build_features(
    records: Sequence[RateRecord], currency: str, effective_date: date
) -> dict[str, float | None]:
    """Return the serialized B1 base14 row for one target publication event.

    ``effective_date - 1 day`` is the registered research publication-day
    convention used only for the calendar features. It is not a live source
    observation timestamp; the runner owns first-seen and decision clocks.
    """

    if currency not in TARGET_CURRENCIES:
        raise ValueError(f"Unsupported target currency: {currency!r}")
    if type(effective_date) is not date:
        raise ValueError("effective_date must be a date, not a datetime")

    histories = _validated_histories(records, effective_date)
    target = _current_history(histories, currency, effective_date)
    if len(target) < TARGET_HISTORY_REQUIRED:
        raise InsufficientHistory(
            f"{currency} needs {TARGET_HISTORY_REQUIRED} publications through "
            f"{effective_date}; got {len(target)}"
        )

    target_rates = [record.rub_per_unit for record in target]
    target_logs = [math.log(value) for value in target_rates]
    one_step = _move(target_logs, 1)

    # Exact B1 subset of alfa_features_v3._base_rate_features.
    raw: dict[str, float] = {
        "corridor_move_1_bps": one_step,
        "corridor_move_5_bps": _move(target_logs, 5),
        "corridor_move_20_bps": _move(target_logs, 20),
        "corridor_level_percentile_20": _percentile(target_rates, 20),
        "corridor_level_percentile_120": _percentile(target_rates, 120),
    }
    one_step_returns = [math.nan] + [
        1e4 * (target_logs[index] - target_logs[index - 1]) for index in range(1, len(target_logs))
    ]
    raw["corridor_volatility_20_bps"] = _rolling_population_std(one_step_returns, 20)

    # shift(1) - shift(4) is the three-update move immediately before t.
    prior_move_3 = 1e4 * (target_logs[-2] - target_logs[-5])
    prior_sign = 1.0 if prior_move_3 > 0 else -1.0 if prior_move_3 < 0 else 0.0
    raw["corridor_reversal_score_3_bps"] = -prior_sign * one_step

    # Each control move is computed on that control's own event sequence, then
    # joined to the target by exact effective date, as in _control_context.
    control_moves = {window: [] for window in (1, 5, 20)}
    for control_currency in CONTROL_CURRENCIES:
        control = _current_history(histories, control_currency, effective_date)
        if len(control) < CONTROL_HISTORY_REQUIRED:
            raise InsufficientHistory(
                f"{control_currency} needs {CONTROL_HISTORY_REQUIRED} publications "
                f"through {effective_date}; got {len(control)}"
            )
        control_logs = [math.log(record.rub_per_unit) for record in control]
        for window in control_moves:
            control_moves[window].append(_move(control_logs, window))
    for window, values in control_moves.items():
        raw[f"rub_common_move_{window}_bps"] = math.fsum(values) / len(values)

    try:
        publication_date = effective_date - timedelta(days=1)
    except OverflowError as exc:
        raise ValueError("effective_date has no preceding publication day") from exc
    prior_publication_date = target[-2].effective_date - timedelta(days=1)
    raw["days_since_prior_publication"] = float((publication_date - prior_publication_date).days)
    raw["publication_is_friday"] = float(publication_date.weekday() == 4)
    angle = math.tau * (publication_date.month - 1) / 12
    raw["publication_month_sin"] = math.sin(angle)
    raw["publication_month_cos"] = math.cos(angle)

    if set(raw) != set(FEATURES):
        missing = sorted(set(FEATURES) - set(raw))
        extra = sorted(set(raw) - set(FEATURES))
        raise AssertionError(f"B1 base14 schema drift: missing={missing}, extra={extra}")
    return {name: _serialize_12g(raw[name], name) for name in FEATURES}


__all__ = [
    "CONTROL_HISTORY_REQUIRED",
    "IncompleteSources",
    "InsufficientHistory",
    "TARGET_HISTORY_REQUIRED",
    "build_features",
]
