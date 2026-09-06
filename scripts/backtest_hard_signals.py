#!/usr/bin/env python3
"""Causal backtest for spike, low-level and stable-corridor signals.

The product experiment contains no calendar reminders and no cross-family
combination. Every feature at T uses observations available no later than T;
future data only scores an already emitted signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_CURRENCIES = ("AMD", "KGS", "KZT", "TJS", "UZS")
DEFAULT_HORIZONS = (2, 3, 4, 5)
DEFAULT_RECENT_YEARS = (3,)
DEFAULT_RESEARCH_HORIZONS = (2, 3, 4, 5)
DEFAULT_EVALUATION_YEARS = 3
MOMENTUM_STREAKS = (2, 3, 4, 5)
EXCLUDED_MOMENTUM_STREAKS = (10, 20)


@dataclass(frozen=True)
class Candidate:
    signal_type: str
    lookback: int
    threshold: float

    @property
    def candidate_id(self) -> str:
        if self.signal_type == "level":
            return f"level_w{self.lookback}_p{int(self.threshold * 100)}"
        if self.signal_type == "stable":
            return f"stable_w{self.lookback}_r{int(self.threshold)}bp"
        return f"momentum_streak_{self.lookback}d"

    @property
    def parameters(self) -> str:
        if self.signal_type == "spike":
            return json.dumps(
                {"consecutive_declines": self.lookback},
                ensure_ascii=False,
                sort_keys=True,
            )
        threshold_name = {
            "level": "higher_share",
            "stable": "max_range_bps",
        }[self.signal_type]
        return json.dumps(
            {"lookback": self.lookback, threshold_name: self.threshold},
            ensure_ascii=False,
            sort_keys=True,
        )


@dataclass(frozen=True)
class SpikePolicy:
    streak_days: int

    @property
    def policy_id(self) -> str:
        return f"momentum_streak_{self.streak_days}d"

    @property
    def parameters(self) -> str:
        return json.dumps(
            {"consecutive_declines": self.streak_days},
            ensure_ascii=False,
            sort_keys=True,
        )


@dataclass(frozen=True)
class LevelPolicy:
    lookback: int
    higher_share: float

    @property
    def policy_id(self) -> str:
        return f"level_L{self.lookback}_p{int(self.higher_share * 100)}"

    @property
    def parameters(self) -> str:
        return json.dumps(
            {"lookback": self.lookback, "higher_share": self.higher_share},
            ensure_ascii=False,
            sort_keys=True,
        )


@dataclass(frozen=True)
class CorridorExitPolicy:
    lookback: int
    max_range_bps: float
    breakout_bps: float
    direction: str

    @property
    def policy_id(self) -> str:
        return (
            f"corridor_exit_{self.direction}_L{self.lookback}"
            f"_r{int(self.max_range_bps)}_b{int(self.breakout_bps)}bp"
        )

    @property
    def parameters(self) -> str:
        return json.dumps(
            {
                "lookback": self.lookback,
                "max_range_bps": self.max_range_bps,
                "breakout_bps": self.breakout_bps,
                "direction": self.direction,
            },
            ensure_ascii=False,
            sort_keys=True,
        )


def candidate_grid() -> list[Candidate]:
    return [Candidate("spike", streak_days, 0.0) for streak_days in MOMENTUM_STREAKS]


def spike_policy_grid() -> list[SpikePolicy]:
    """Literal case baseline: N consecutive published-day declines."""
    return [SpikePolicy(streak_days) for streak_days in MOMENTUM_STREAKS]


def level_policy_grid() -> list[LevelPolicy]:
    return [
        LevelPolicy(lookback, share)
        for lookback in (20, 60, 120, 250)
        for share in (0.70, 0.80, 0.90, 0.95)
    ]


def corridor_exit_policy_grid() -> list[CorridorExitPolicy]:
    return [
        CorridorExitPolicy(lookback, max_range, breakout, direction)
        for lookback in (5, 10, 20)
        for max_range in (50.0, 100.0, 200.0, 400.0)
        for breakout in (0.0, 25.0, 50.0)
        for direction in ("down", "up")
    ]


def latest_default_input() -> Path:
    candidates = sorted(Path("data/processed").glob("cbr_fx_daily_2010-01-01_*.csv"))
    if not candidates:
        raise FileNotFoundError("No data/processed/cbr_fx_daily_2010-01-01_*.csv file found")
    return candidates[-1]


def load_prices(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "currency", "rub_per_unit"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame.loc[frame["currency"].isin(TARGET_CURRENCIES), list(required)].copy()
    frame = frame.sort_values(["currency", "date"])
    if frame.duplicated(["currency", "date"]).any():
        raise ValueError("Duplicate currency/date observations found")
    if (frame["rub_per_unit"] <= 0).any():
        raise ValueError("All rub_per_unit values must be positive")
    return frame


def compute_candidate(series: pd.Series, candidate: Candidate) -> pd.DataFrame:
    favourable_speed_bps = -series.pct_change() * 10_000

    if candidate.signal_type == "level":
        # Compare T with exactly Y observations preceding T.  A value of 0.9
        # means that 90% of those historical prices were higher (worse).
        value = series.rolling(candidate.lookback + 1, min_periods=candidate.lookback + 1).apply(
            lambda values: float(np.mean(values[:-1] > values[-1])), raw=True
        )
        raw_signal = value >= candidate.threshold
        strength = (value - candidate.threshold) * 10_000
    elif candidate.signal_type == "stable":
        rolling = series.rolling(candidate.lookback, min_periods=candidate.lookback)
        value = (rolling.max() / rolling.min() - 1.0) * 10_000
        raw_signal = value <= candidate.threshold
        strength = candidate.threshold - value
    elif candidate.signal_type == "spike":
        daily_decline = series.lt(series.shift(1))
        streak_group = (~daily_decline).cumsum()
        streak_length = daily_decline.groupby(streak_group).cumsum().astype(float)
        # Fire once when the current run first reaches N published declines.
        value = streak_length
        raw_signal = streak_length.eq(candidate.lookback)
        # Strength is descriptive only; it never replaces the consecutive-day rule.
        strength = (series.shift(candidate.lookback) / series - 1.0) * 10_000
    else:
        raise ValueError(f"Unknown signal type: {candidate.signal_type}")

    return pd.DataFrame(
        {
            "indicator_value": value,
            "strength": strength,
            "speed_bps": favourable_speed_bps,
            "raw_signal": raw_signal.fillna(False),
        },
        index=series.index,
    )


def thin_with_cooldown(mask: pd.Series, cooldown_observations: int) -> list[int]:
    selected: list[int] = []
    last_position = -10**9
    for position in np.flatnonzero(mask.to_numpy(dtype=bool)):
        if position - last_position > cooldown_observations:
            selected.append(int(position))
            last_position = int(position)
    return selected


def push_text(currency: str, candidate: Candidate, value: float, price: float) -> str:
    return f"Курс {currency} снижается {candidate.lookback}-й опубликованный день подряд."


def make_signal_row(
    *,
    date: pd.Timestamp,
    currency: str,
    price: float,
    strategy_id: str,
    strategy_family: str,
    policy: str,
    candidate: Candidate,
    feature_row: pd.Series,
    strategy_parameters: str | None = None,
    components_triggered: str | None = None,
) -> dict[str, object]:
    indicator_value = float(feature_row["indicator_value"])
    strength = float(feature_row["strength"])
    speed_bps = float(feature_row["speed_bps"])
    return {
        "date": date,
        "currency": currency,
        "strategy_id": strategy_id,
        "strategy_family": strategy_family,
        "policy": policy,
        "indicator": candidate.signal_type,
        "parameters": strategy_parameters or candidate.parameters,
        "components_triggered": components_triggered or candidate.signal_type,
        "direction": "lower_rub_per_unit_is_better",
        "indicator_value": indicator_value,
        "strength": strength,
        "speed_bps": speed_bps,
        "rub_per_unit": price,
        "recommended_scenario": "favourable_now",
        "push_text": push_text(currency, candidate, indicator_value, price),
    }


def build_signals(
    prices: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    cooldown_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build pure spike streams; no calendar or other signal family is mixed in."""
    signal_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []

    for currency, group in prices.groupby("currency", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        series = group["rub_per_unit"]
        dates = pd.DatetimeIndex(group["date"])
        spike_candidates = candidate_grid()
        feature_cache = {
            candidate.candidate_id: compute_candidate(series, candidate)
            for candidate in spike_candidates
        }

        for policy in spike_policy_grid():
            candidate = Candidate("spike", policy.streak_days, 0.0)
            feature = feature_cache[candidate.candidate_id]
            eligible = feature["raw_signal"] & (group["date"] >= evaluation_start)
            # raw_signal is already a first-completion event (streak == N),
            # not a daily state. Apply no extra thinning before comparing the
            # fast and slow base indicators; cooldown belongs to a later
            # combined communication stream.
            positions = np.flatnonzero(eligible.to_numpy(dtype=bool)).tolist()
            strategy_id = f"{policy.policy_id}__spike_only"
            policy_rows.append(
                {
                    "candidate_id": policy.policy_id,
                    "signal_type": "spike_policy",
                    "lookback": policy.streak_days,
                    "threshold": np.nan,
                    "parameters": policy.parameters,
                }
            )

            for position in positions:
                signal_rows.append(
                    make_signal_row(
                        date=dates[position],
                        currency=currency,
                        price=float(series.iloc[position]),
                        strategy_id=strategy_id,
                        strategy_family="spike_policy",
                        policy="spike_only",
                        candidate=candidate,
                        feature_row=feature.iloc[position],
                        strategy_parameters=policy.parameters,
                        components_triggered=f"streak_{policy.streak_days}d",
                    )
                )

    signals = pd.DataFrame(signal_rows).sort_values(["currency", "strategy_id", "date"])
    policies = pd.DataFrame(policy_rows).drop_duplicates().sort_values("candidate_id")
    return signals, policies


def outcome_frame(series: pd.Series, dates: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    past = pd.concat([series.shift(step) for step in range(1, horizon + 1)], axis=1)
    future = pd.concat([series.shift(-step) for step in range(1, horizon + 1)], axis=1)
    symmetric = pd.concat(
        [series.shift(step) for step in range(horizon, -horizon - 1, -1)], axis=1
    )
    valid = past.notna().all(axis=1) & future.notna().all(axis=1)
    past_min = past.min(axis=1)
    future_min = future.min(axis=1)
    # Positive benefit means a lower-than-average foreign-currency price at T.
    benefit_bps = (symmetric.mean(axis=1) / series - 1.0) * 10_000
    future_regret_bps = np.maximum(series / future_min - 1.0, 0.0) * 10_000
    local_window_min = pd.concat([past_min, future_min], axis=1).min(axis=1)
    local_regret_bps = np.maximum(series / local_window_min - 1.0, 0.0) * 10_000
    iso = dates.isocalendar()
    label_available_date = pd.Series(dates).shift(-horizon)
    return pd.DataFrame(
        {
            "date": dates,
            # Primary product truth: no better price appears after T.
            "hit": (future_min >= series).astype(float),
            "hit_10bp": (future_regret_bps <= 10.0).astype(float),
            "hit_25bp": (future_regret_bps <= 25.0).astype(float),
            "forward_hold_hit": (future_min >= series).astype(float),
            "forward_hold_hit_10bp": (future_regret_bps <= 10.0).astype(float),
            "forward_hold_hit_25bp": (future_regret_bps <= 25.0).astype(float),
            # Secondary diagnostic matching the symmetric ML label.
            "local_min_hit": ((past_min >= series) & (future_min >= series)).astype(float),
            "local_min_hit_10bp": (local_regret_bps <= 10.0).astype(float),
            "local_min_hit_25bp": (local_regret_bps <= 25.0).astype(float),
            "benefit_bps": benefit_bps,
            "forward_change_bps": (series.shift(-horizon) / series - 1.0) * 10_000,
            "future_regret_bps": future_regret_bps,
            "iso_year": iso.year.to_numpy(),
            "iso_week": iso.week.to_numpy(),
            "label_available_date": label_available_date.to_numpy(),
            "valid": valid,
        }
    ).set_index("date")


def deterministic_seed(base_seed: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**32)


def monthly_block_bootstrap_mean(
    values: pd.Series,
    repeats: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Percentile CI and one-sided bootstrap tail probability for mean > 0.

    Whole calendar months are resampled to retain dependence between nearby
    and overlapping signal outcomes.  The one-sided value is the smoothed
    bootstrap probability of a non-positive mean. Computing it from the same
    distribution as the percentile interval keeps both inference summaries
    coherent when calendar blocks contain different numbers of signals.
    """
    clean = values.dropna().astype(float).sort_index()
    if clean.empty:
        return {"ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    observed = float(clean.mean())
    month = pd.DatetimeIndex(clean.index).to_period("M")
    grouped = clean.groupby(month)
    block_sums = grouped.sum().to_numpy(float)
    block_counts = grouped.size().to_numpy(float)
    block_count = len(block_sums)
    if block_count < 2:
        return {"ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, block_count, size=(repeats, block_count))
    denominators = block_counts[sampled].sum(axis=1)
    bootstrap_means = block_sums[sampled].sum(axis=1) / denominators
    alpha = 1.0 - confidence
    percentile_low, percentile_high = np.quantile(
        bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0]
    )

    # Test H0: mean <= 0 by centring the block-bootstrap distribution at zero.
    # Using P(bootstrap_mean <= 0) would leave the distribution centred at the
    # observed effect and is a confidence-tail measure, not a null test.
    null_means = bootstrap_means - observed
    p_value = (1 + (null_means >= observed).sum()) / (repeats + 1)
    # Use the same centred error distribution for the interval and the test.
    ci_low = 2.0 * observed - percentile_high
    ci_high = 2.0 * observed - percentile_low
    return {
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
    }


def period_random(
    outcomes: pd.DataFrame,
    signal_count: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """
    Primary random-day baseline.

    Draw the same number of random published CBR dates from the same
    currency and evaluation period as the strategy signals.

    Unlike week_matched_random, this baseline does NOT know which weeks
    the strategy selected.
    """
    columns = [
        "hit",
        "hit_10bp",
        "hit_25bp",
        "forward_hold_hit",
        "forward_hold_hit_10bp",
        "forward_hold_hit_25bp",
        "local_min_hit",
        "local_min_hit_10bp",
        "local_min_hit_25bp",
        "benefit_bps",
        "forward_change_bps",
        "future_regret_bps",
    ]

    pool = outcomes[columns].to_numpy(dtype=float)

    if signal_count <= 0:
        raise ValueError("signal_count must be positive")

    if signal_count > len(pool):
        raise ValueError(
            f"Cannot draw {signal_count} dates from pool of {len(pool)}"
        )

    rng = np.random.default_rng(seed)

    if signal_count == len(pool):
        means = np.repeat(
            pool.mean(axis=0, keepdims=True),
            repeats,
            axis=0,
        )
    else:
        # Uniform sampling without replacement for every Monte Carlo run.
        random_keys = rng.random((repeats, len(pool)))
        chosen = np.argpartition(
            random_keys,
            kth=signal_count - 1,
            axis=1,
        )[:, :signal_count]

        means = pool[chosen].mean(axis=1)

    return pd.DataFrame(
        means,
        columns=[
            "hit_rate",
            "hit_rate_10bp",
            "hit_rate_25bp",
            "forward_hold_rate",
            "forward_hold_rate_10bp",
            "forward_hold_rate_25bp",
            "local_min_hit_rate",
            "local_min_hit_rate_10bp",
            "local_min_hit_rate_25bp",
            "benefit_bps",
            "forward_change_bps",
            "future_regret_bps",
        ],
    )


def week_matched_random(
    pools: dict[tuple[int, int], np.ndarray],
    signal_outcomes: pd.DataFrame,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    counts = signal_outcomes.groupby(["iso_year", "iso_week"]).size()
    rng = np.random.default_rng(seed)
    totals = np.zeros((repeats, 12), dtype=float)
    total_count = int(counts.sum())
    for key, count_value in counts.items():
        pool = pools[(int(key[0]), int(key[1]))]
        count = int(count_value)
        if count == len(pool):
            totals += pool.sum(axis=0)
            continue
        # The smallest random keys form a uniform sample without replacement.
        random_keys = rng.random((repeats, len(pool)))
        chosen = np.argpartition(random_keys, kth=count - 1, axis=1)[:, :count]
        totals += pool[chosen].sum(axis=1)
    means = totals / total_count
    return pd.DataFrame(
        means,
        columns=[
            "hit_rate",
            "hit_rate_10bp",
            "hit_rate_25bp",
            "forward_hold_rate",
            "forward_hold_rate_10bp",
            "forward_hold_rate_25bp",
            "local_min_hit_rate",
            "local_min_hit_rate_10bp",
            "local_min_hit_rate_25bp",
            "benefit_bps",
            "forward_change_bps",
            "future_regret_bps",
        ],
    )


def evaluate_strategies(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    horizons: tuple[int, ...],
    repeats: int,
    base_seed: int,
    cooldown_observations: int,
    evaluation_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for currency, price_group in prices.groupby("currency", sort=True):
        price_group = price_group.sort_values("date").reset_index(drop=True)

        series = price_group["rub_per_unit"]
        dates = pd.DatetimeIndex(price_group["date"])

        currency_signals = signals[
            signals["currency"] == currency
        ]

        period_date_mask = dates >= evaluation_start
        if evaluation_end is not None:
            period_date_mask &= dates < evaluation_end
        period_dates = dates[period_date_mask]
        period_iso = period_dates.isocalendar()
        week_index = pd.MultiIndex.from_arrays(
            [period_iso.year.to_numpy(), period_iso.week.to_numpy()],
            names=["iso_year", "iso_week"],
        ).drop_duplicates()
        total_weeks = len(week_index)

        for horizon in horizons:
            outcomes = outcome_frame(
                series,
                dates,
                horizon,
            )

            outcomes = outcomes[
                outcomes["valid"]
                & (outcomes.index >= evaluation_start)
                & (
                    True
                    if evaluation_end is None
                    else outcomes.index < evaluation_end
                )
            ].copy()

            if outcomes.empty:
                continue

            # Week-matched baseline stays as an additional robustness test.
            outcome_pools = {
                (int(key[0]), int(key[1])): group[
                    [
                        "hit",
                        "hit_10bp",
                        "hit_25bp",
                        "forward_hold_hit",
                        "forward_hold_hit_10bp",
                        "forward_hold_hit_25bp",
                        "local_min_hit",
                        "local_min_hit_10bp",
                        "local_min_hit_25bp",
                        "benefit_bps",
                        "forward_change_bps",
                        "future_regret_bps",
                    ]
                ].to_numpy()
                for key, group in outcomes.groupby(
                    ["iso_year", "iso_week"],
                    sort=False,
                )
            }

            for strategy_id, strategy_signals in currency_signals.groupby(
                "strategy_id",
                sort=True,
            ):
                period_signals = strategy_signals.loc[
                    strategy_signals["date"].ge(evaluation_start)
                    & (
                        True
                        if evaluation_end is None
                        else strategy_signals["date"].lt(evaluation_end)
                    )
                ].copy()
                if period_signals.empty:
                    continue
                signal_iso = pd.DatetimeIndex(period_signals["date"]).isocalendar()
                period_signals["iso_year"] = signal_iso.year.to_numpy()
                period_signals["iso_week"] = signal_iso.week.to_numpy()
                selected_all = (
                    strategy_signals
                    .set_index("date")
                    .join(
                        outcomes,
                        how="inner",
                        rsuffix="_outcome",
                    )
                )

                if selected_all.empty:
                    continue

                quality_scope = "pure_spike_signals"
                selected = selected_all
                if selected.empty:
                    continue

                total_push_count = len(period_signals)
                evaluated_signal_count = len(selected)

                # ---------------------------------------------------------
                # PRIMARY baseline:
                # same number of random dates from the same full period.
                # ---------------------------------------------------------
                random = period_random(
                    outcomes=outcomes,
                    signal_count=evaluated_signal_count,
                    repeats=repeats,
                    seed=deterministic_seed(
                        base_seed,
                        "period_random",
                        currency,
                        strategy_id,
                        horizon,
                    ),
                )

                # ---------------------------------------------------------
                # SECONDARY robustness baseline:
                # random dates inside exactly the same weeks.
                # ---------------------------------------------------------
                week_random = week_matched_random(
                    outcome_pools,
                    selected,
                    repeats,
                    deterministic_seed(
                        base_seed,
                        "week_random",
                        currency,
                        strategy_id,
                        horizon,
                    ),
                )

                hit_rate = float(
                    selected["hit"].mean()
                )
                hit_rate_10bp = float(selected["hit_10bp"].mean())
                hit_rate_25bp = float(selected["hit_25bp"].mean())
                forward_hold_rate = float(selected["forward_hold_hit"].mean())
                forward_hold_rate_10bp = float(
                    selected["forward_hold_hit_10bp"].mean()
                )
                forward_hold_rate_25bp = float(
                    selected["forward_hold_hit_25bp"].mean()
                )
                local_min_hit_rate = float(selected["local_min_hit"].mean())
                local_min_hit_rate_10bp = float(
                    selected["local_min_hit_10bp"].mean()
                )
                local_min_hit_rate_25bp = float(
                    selected["local_min_hit_25bp"].mean()
                )

                benefit_bps = float(
                    selected["benefit_bps"].mean()
                )
                benefit_inference = monthly_block_bootstrap_mean(
                    selected["benefit_bps"],
                    repeats,
                    deterministic_seed(
                        base_seed,
                        "benefit_monthly_bootstrap",
                        currency,
                        strategy_id,
                        horizon,
                        evaluation_start,
                        evaluation_end,
                    ),
                )

                random_hit = float(
                    random["hit_rate"].mean()
                )
                random_hit_10bp = float(random["hit_rate_10bp"].mean())
                random_hit_25bp = float(random["hit_rate_25bp"].mean())
                random_forward_hold = float(random["forward_hold_rate"].mean())
                random_forward_hold_10bp = float(
                    random["forward_hold_rate_10bp"].mean()
                )
                random_forward_hold_25bp = float(
                    random["forward_hold_rate_25bp"].mean()
                )
                random_local_min_hit = float(random["local_min_hit_rate"].mean())
                random_local_min_hit_10bp = float(
                    random["local_min_hit_rate_10bp"].mean()
                )
                random_local_min_hit_25bp = float(
                    random["local_min_hit_rate_25bp"].mean()
                )

                week_random_hit = float(
                    week_random["hit_rate"].mean()
                )

                # Main metric from the task.
                hit_lift = (
                    hit_rate / random_hit
                    if random_hit > 0
                    else np.nan
                )
                hit_lift_10bp = (
                    hit_rate_10bp / random_hit_10bp
                    if random_hit_10bp > 0
                    else np.nan
                )
                hit_lift_25bp = (
                    hit_rate_25bp / random_hit_25bp
                    if random_hit_25bp > 0
                    else np.nan
                )
                forward_hold_lift = (
                    forward_hold_rate / random_forward_hold
                    if random_forward_hold > 0
                    else np.nan
                )
                forward_hold_lift_10bp = (
                    forward_hold_rate_10bp / random_forward_hold_10bp
                    if random_forward_hold_10bp > 0
                    else np.nan
                )
                forward_hold_lift_25bp = (
                    forward_hold_rate_25bp / random_forward_hold_25bp
                    if random_forward_hold_25bp > 0
                    else np.nan
                )
                local_min_lift = (
                    local_min_hit_rate / random_local_min_hit
                    if random_local_min_hit > 0
                    else np.nan
                )
                local_min_lift_10bp = (
                    local_min_hit_rate_10bp / random_local_min_hit_10bp
                    if random_local_min_hit_10bp > 0
                    else np.nan
                )
                local_min_lift_25bp = (
                    local_min_hit_rate_25bp / random_local_min_hit_25bp
                    if random_local_min_hit_25bp > 0
                    else np.nan
                )

                week_matched_hit_lift = (
                    hit_rate / week_random_hit
                    if week_random_hit > 0
                    else np.nan
                )

                # Monte Carlo significance of the MAIN metric.
                hit_p_value = float(
                    (
                        1
                        + (
                            random["hit_rate"]
                            >= hit_rate
                        ).sum()
                    )
                    / (repeats + 1)
                )

                week_matched_hit_p_value = float(
                    (
                        1
                        + (
                            week_random["hit_rate"]
                            >= hit_rate
                        ).sum()
                    )
                    / (repeats + 1)
                )

                # ---------------------------------------------------------
                # Frequency / clustering
                # ---------------------------------------------------------
                signals_per_week = (
                    total_push_count / total_weeks
                    if total_weeks > 0
                    else np.nan
                )

                market_signals_per_week = (
                    evaluated_signal_count / total_weeks
                    if total_weeks > 0
                    else np.nan
                )

                selected_week_counts = (
                    period_signals
                    .groupby(
                        ["iso_year", "iso_week"]
                    )
                    .size()
                    .reindex(
                        week_index,
                        fill_value=0,
                    )
                )

                active_weeks = int(
                    (selected_week_counts > 0).sum()
                )

                active_week_share = (
                    active_weeks / total_weeks
                    if total_weeks > 0
                    else np.nan
                )

                weekly_mean = float(
                    selected_week_counts.mean()
                )

                weekly_std = float(
                    selected_week_counts.std(ddof=0)
                )

                weekly_count_cv = (
                    weekly_std / weekly_mean
                    if weekly_mean > 0
                    else np.nan
                )

                signal_dates = pd.DatetimeIndex(
                    period_signals["date"]
                ).sort_values()

                if len(signal_dates) > 1:
                    gap_days = (
                        np.diff(
                            signal_dates.values
                        )
                        .astype("timedelta64[D]")
                        .astype(int)
                    )

                    median_gap_days = float(
                        np.median(gap_days)
                    )

                    max_gap_days = float(
                        np.max(gap_days)
                    )

                    p90_gap_days = float(
                        np.quantile(
                            gap_days,
                            0.90,
                        )
                    )
                else:
                    median_gap_days = np.nan
                    max_gap_days = np.nan
                    p90_gap_days = np.nan

                first = strategy_signals.iloc[0]
                selected_signal_family = (
                    first["selected_signal_family"]
                    if "selected_signal_family" in strategy_signals.columns
                    else first["strategy_family"]
                )

                rows.append(
                    {
                        "currency": currency,
                        "strategy_id": strategy_id,
                        "strategy_family": first[
                            "strategy_family"
                        ],
                        "selected_signal_family": selected_signal_family,
                        "policy": first["policy"],
                        "horizon": horizon,

                        # Counts / frequency
                        # Legacy names keep notebook compatibility: signal_count
                        # and signals_per_week refer to the full communication
                        # stream, while hit/benefit use evaluated market signals.
                        "signal_count": total_push_count,
                        "total_push_count": total_push_count,
                        "evaluated_signal_count": evaluated_signal_count,
                        "signals_per_week": signals_per_week,
                        "market_signals_per_week": market_signals_per_week,
                        "quality_scope": quality_scope,
                        "active_weeks": active_weeks,
                        "active_week_share": active_week_share,
                        "weekly_count_cv": weekly_count_cv,
                        "median_gap_days": median_gap_days,
                        "p90_gap_days": p90_gap_days,
                        "max_gap_days": max_gap_days,

                        # MAIN hit metrics
                        "hit_rate": hit_rate,
                        "hit_target": "no_lower_price_in_next_h_observations",
                        "random_hit_rate_mean": random_hit,
                        "random_hit_rate_p05": float(
                            random[
                                "hit_rate"
                            ].quantile(0.05)
                        ),
                        "random_hit_rate_p95": float(
                            random[
                                "hit_rate"
                            ].quantile(0.95)
                        ),
                        "hit_lift": hit_lift,
                        "hit_uplift_ratio": hit_lift,
                        "hit_rate_10bp": hit_rate_10bp,
                        "random_hit_rate_10bp_mean": random_hit_10bp,
                        "hit_uplift_ratio_10bp": hit_lift_10bp,
                        "hit_rate_25bp": hit_rate_25bp,
                        "random_hit_rate_25bp_mean": random_hit_25bp,
                        "hit_uplift_ratio_25bp": hit_lift_25bp,
                        "forward_hold_rate": forward_hold_rate,
                        "random_forward_hold_rate_mean": random_forward_hold,
                        "forward_hold_uplift_ratio": forward_hold_lift,
                        "forward_hold_rate_10bp": forward_hold_rate_10bp,
                        "random_forward_hold_rate_10bp_mean": random_forward_hold_10bp,
                        "forward_hold_uplift_ratio_10bp": forward_hold_lift_10bp,
                        "forward_hold_rate_25bp": forward_hold_rate_25bp,
                        "random_forward_hold_rate_25bp_mean": random_forward_hold_25bp,
                        "forward_hold_uplift_ratio_25bp": forward_hold_lift_25bp,
                        "local_min_hit_rate": local_min_hit_rate,
                        "random_local_min_hit_rate_mean": random_local_min_hit,
                        "local_min_uplift_ratio": local_min_lift,
                        "local_min_hit_rate_10bp": local_min_hit_rate_10bp,
                        "random_local_min_hit_rate_10bp_mean": random_local_min_hit_10bp,
                        "local_min_uplift_ratio_10bp": local_min_lift_10bp,
                        "local_min_hit_rate_25bp": local_min_hit_rate_25bp,
                        "random_local_min_hit_rate_25bp_mean": random_local_min_hit_25bp,
                        "local_min_uplift_ratio_25bp": local_min_lift_25bp,
                        "hit_p_value": hit_p_value,

                        # Week-matched robustness
                        "week_matched_random_hit_rate_mean":
                            week_random_hit,
                        "week_matched_hit_lift":
                            week_matched_hit_lift,
                        "week_matched_hit_p_value":
                            week_matched_hit_p_value,

                        # Benefit
                        "benefit_bps": benefit_bps,
                        "benefit_ci_low_95": benefit_inference["ci_low"],
                        "benefit_ci_high_95": benefit_inference["ci_high"],
                        "benefit_p_value_vs_zero": benefit_inference["p_value"],

                        # Additional diagnostics
                        "forward_change_bps": float(
                            selected[
                                "forward_change_bps"
                            ].mean()
                        ),
                        "random_forward_change_bps_mean":
                            float(
                                random[
                                    "forward_change_bps"
                                ].mean()
                            ),
                        "future_regret_bps": float(
                            selected[
                                "future_regret_bps"
                            ].mean()
                        ),
                        "random_future_regret_bps_mean":
                            float(
                                random[
                                    "future_regret_bps"
                                ].mean()
                            ),

                        "first_signal": signal_dates.min(),
                        "last_signal": signal_dates.max(),
                    }
                )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "horizon",
                "currency",
                "strategy_family",
                "strategy_id",
            ]
        )
        .reset_index(drop=True)
    )


def build_walk_forward_signals(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    selection_horizon: int,
    min_train_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a policy on matured history and deploy it for one year.

    Candidate outcomes whose horizon has not finished by the selection date
    are excluded. The ranking first enforces the product frequency and sample
    size gates, then positive benefit and lift. This prevents a rare, lucky
    rule from silently winning a large grid search.
    """
    deployed_rows: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    first_eligible_date = evaluation_start + pd.DateOffset(years=min_train_years)
    starts_mid_year = first_eligible_date.month != 1 or first_eligible_date.day != 1
    first_deployment = pd.Timestamp(
        year=first_eligible_date.year + int(starts_mid_year),
        month=1,
        day=1,
    )

    for currency, price_group in prices.groupby("currency", sort=True):
        price_group = price_group.sort_values("date").reset_index(drop=True)
        outcomes = outcome_frame(
            price_group["rub_per_unit"],
            pd.DatetimeIndex(price_group["date"]),
            selection_horizon,
        )
        currency_signals = signals.loc[signals["currency"] == currency]
        last_date = pd.Timestamp(price_group["date"].max())

        fold_start = pd.Timestamp(
            year=first_deployment.year, month=1, day=1
        )
        while fold_start <= last_date:
            fold_end = fold_start + pd.DateOffset(years=1)
            train_outcomes = outcomes.loc[
                outcomes["valid"]
                & (outcomes.index >= evaluation_start)
                & (outcomes["label_available_date"] < fold_start)
            ].copy()
            if train_outcomes.empty:
                fold_start = fold_end
                continue

            train_weeks = len(
                train_outcomes[["iso_year", "iso_week"]].drop_duplicates()
            )
            baseline_hit = float(train_outcomes["hit"].mean())

            experiment_policies = (("spike", "spike_only"),)
            for signal_family, policy in experiment_policies:
                policy_signals = currency_signals.loc[
                    (currency_signals["policy"] == policy)
                    & (currency_signals["strategy_family"] == "spike_policy")
                ]
                candidates: list[dict[str, object]] = []
                for strategy_id, strategy_signals in policy_signals.groupby(
                    "strategy_id", sort=True
                ):
                    train_all = (
                        strategy_signals.set_index("date")
                        .join(train_outcomes, how="inner", rsuffix="_outcome")
                    )
                    train_market = train_all
                    if train_market.empty:
                        continue
                    total_frequency = len(train_all) / train_weeks
                    hit_rate = float(train_market["hit"].mean())
                    hit_lift = hit_rate / baseline_hit if baseline_hit > 0 else np.nan
                    benefit_bps = float(train_market["benefit_bps"].mean())
                    frequency_distance = max(
                        1.0 - total_frequency, total_frequency - 2.0, 0.0
                    )
                    sample_gate = len(train_market) >= 30
                    frequency_gate = 1.0 <= total_frequency <= 2.0
                    lift_gate = hit_lift >= 1.20
                    benefit_gate = benefit_bps >= -5.0
                    gates_met = sum((sample_gate, lift_gate, benefit_gate))
                    candidates.append(
                        {
                            "strategy_id": strategy_id,
                            "train_total_push_count": len(train_all),
                            "train_evaluated_signal_count": len(train_market),
                            "train_signals_per_week": total_frequency,
                            "train_hit_rate": hit_rate,
                            "train_random_hit_rate": baseline_hit,
                            "train_hit_lift": hit_lift,
                            "train_benefit_bps": benefit_bps,
                            "frequency_distance": frequency_distance,
                            "sample_gate": sample_gate,
                            "frequency_gate": frequency_gate,
                            "lift_gate": lift_gate,
                            "benefit_gate": benefit_gate,
                            "gates_met": gates_met,
                        }
                    )

                if not candidates:
                    continue
                ranking = pd.DataFrame(candidates).sort_values(
                    [
                        "sample_gate",
                        "lift_gate",
                        "benefit_gate",
                        "train_hit_lift",
                        "train_benefit_bps",
                    ],
                    ascending=[False, False, False, False, False],
                )
                chosen = ranking.iloc[0].to_dict()
                chosen_id = str(chosen["strategy_id"])
                chosen["currency"] = currency
                chosen["policy"] = policy
                chosen["signal_family"] = signal_family
                chosen["selection_horizon"] = selection_horizon
                chosen["fold_start"] = fold_start
                chosen["fold_end"] = min(fold_end, last_date + pd.offsets.Day(1))
                chosen["meets_all_train_gates"] = bool(chosen["gates_met"] == 3)
                selection_rows.append(chosen)

                deployed = policy_signals.loc[
                    (policy_signals["strategy_id"] == chosen_id)
                    & (policy_signals["date"] >= fold_start)
                    & (policy_signals["date"] < fold_end)
                ].copy()
                if deployed.empty:
                    continue
                deployed["selected_candidate_id"] = chosen_id
                deployed["selected_signal_family"] = signal_family
                deployed["selection_horizon"] = selection_horizon
                deployed["selection_date"] = fold_start
                deployed["fold_year"] = fold_start.year
                deployed["strategy_id"] = (
                    f"walk_forward__{signal_family}__{policy}__h{selection_horizon}"
                )
                deployed["strategy_family"] = "walk_forward"
                deployed_rows.append(deployed)

            fold_start = fold_end

    walk_forward_signals = (
        pd.concat(deployed_rows, ignore_index=True)
        if deployed_rows
        else pd.DataFrame()
    )
    selections = pd.DataFrame(selection_rows)
    return walk_forward_signals, selections


def signals_as_of(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    policy_start: pd.Timestamp,
    cooldown_observations: int = 3,
) -> pd.DataFrame:
    """Return signals visible on each currency's latest publication by ``as_of``."""
    history = prices.loc[prices["date"] <= as_of].copy()
    if history.empty:
        return pd.DataFrame()
    signals, _ = build_signals(history, policy_start, cooldown_observations)
    latest_dates = history.groupby("currency")["date"].max().rename("latest_date")
    visible = signals.join(latest_dates, on="currency")
    return visible.loc[visible["date"] == visible["latest_date"]].drop(
        columns="latest_date"
    )


def evaluate_walk_forward_folds(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    horizons: tuple[int, ...],
    repeats: int,
    base_seed: int,
    cooldown_observations: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if signals.empty:
        return pd.DataFrame()
    for fold_year, fold_signals in signals.groupby("fold_year", sort=True):
        fold_start = pd.Timestamp(year=int(fold_year), month=1, day=1)
        fold_end = fold_start + pd.DateOffset(years=1)
        metrics = evaluate_strategies(
            prices,
            fold_signals,
            fold_start,
            horizons,
            repeats,
            base_seed,
            cooldown_observations,
            evaluation_end=fold_end,
        )
        if not metrics.empty:
            metrics.insert(0, "fold_year", int(fold_year))
            rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def evaluate_recent_walk_forward(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    recent_years: tuple[int, ...],
    horizons: tuple[int, ...],
    repeats: int,
    base_seed: int,
    cooldown_observations: int,
) -> pd.DataFrame:
    """Score already-deployed walk-forward signals on recent rolling periods."""
    if signals.empty:
        return pd.DataFrame()
    data_as_of = pd.Timestamp(prices["date"].max())
    rows: list[pd.DataFrame] = []
    for years in recent_years:
        period_start = data_as_of - pd.DateOffset(years=years)
        metrics = evaluate_strategies(
            prices,
            signals,
            period_start,
            horizons,
            repeats,
            base_seed,
            cooldown_observations,
        )
        if metrics.empty:
            continue
        metrics.insert(0, "recent_years", years)
        metrics.insert(1, "requested_period_start", period_start)
        metrics.insert(2, "data_as_of", data_as_of)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_level_corridor_signals(
    prices: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    cooldown_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build level and causal first-exit-from-corridor signal streams."""
    rows: list[dict[str, object]] = []
    policies: list[dict[str, object]] = []

    for currency, group in prices.groupby("currency", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        series = group["rub_per_unit"]
        dates = pd.DatetimeIndex(group["date"])

        for policy in level_policy_grid():
            candidate = Candidate("level", policy.lookback, policy.higher_share)
            feature = compute_candidate(series, candidate)
            eligible = feature["raw_signal"] & group["date"].ge(evaluation_start)
            positions = thin_with_cooldown(eligible, cooldown_observations)
            policies.append(
                {
                    "strategy_id": policy.policy_id,
                    "signal_family": "level",
                    "scenario": "favourable_now",
                    "parameters": policy.parameters,
                }
            )
            for position in positions:
                value = float(feature.iloc[position]["indicator_value"])
                rows.append(
                    {
                        "date": dates[position],
                        "currency": currency,
                        "strategy_id": policy.policy_id,
                        "signal_family": "level",
                        "scenario": "favourable_now",
                        "parameters": policy.parameters,
                        "rub_per_unit": float(series.iloc[position]),
                        "indicator_value": value,
                        "strength": float(feature.iloc[position]["strength"]),
                        "direction": "lower_rub_per_unit_is_better",
                        "push_text": (
                            f"Курс {currency} ниже, чем в {value * 100:.0f}% опубликованных "
                            f"дней за последние {policy.lookback} наблюдений."
                        ),
                    }
                )

        for policy in corridor_exit_policy_grid():
            prior = series.shift(1).rolling(
                policy.lookback, min_periods=policy.lookback
            )
            prior_low = prior.min()
            prior_high = prior.max()
            prior_range_bps = (prior_high / prior_low - 1.0) * 10_000
            was_stable = prior_range_bps <= policy.max_range_bps
            if policy.direction == "down":
                boundary = prior_low * (1.0 - policy.breakout_bps / 10_000)
                raw = was_stable & series.lt(boundary)
                family = "corridor_exit_down"
                scenario = "favourable_now"
                direction = "lower_breakout"
            else:
                boundary = prior_high * (1.0 + policy.breakout_bps / 10_000)
                raw = was_stable & series.gt(boundary)
                family = "corridor_exit_up"
                scenario = "window_closing"
                direction = "upper_breakout"
            positions = thin_with_cooldown(
                raw.fillna(False) & group["date"].ge(evaluation_start),
                cooldown_observations,
            )
            policies.append(
                {
                    "strategy_id": policy.policy_id,
                    "signal_family": family,
                    "scenario": scenario,
                    "parameters": policy.parameters,
                }
            )
            for position in positions:
                exit_bps = (
                    (prior_low.iloc[position] / series.iloc[position] - 1.0) * 10_000
                    if policy.direction == "down"
                    else (series.iloc[position] / prior_high.iloc[position] - 1.0) * 10_000
                )
                rows.append(
                    {
                        "date": dates[position],
                        "currency": currency,
                        "strategy_id": policy.policy_id,
                        "signal_family": family,
                        "scenario": scenario,
                        "parameters": policy.parameters,
                        "rub_per_unit": float(series.iloc[position]),
                        "indicator_value": float(prior_range_bps.iloc[position]),
                        "strength": float(exit_bps),
                        "direction": direction,
                        "prior_corridor_low": float(prior_low.iloc[position]),
                        "prior_corridor_high": float(prior_high.iloc[position]),
                        "push_text": (
                            f"Курс {currency} вышел "
                            f"{'ниже' if policy.direction == 'down' else 'выше'} "
                            f"диапазона последних {policy.lookback} публикаций."
                        ),
                    }
                )

    signals = pd.DataFrame(rows).sort_values(
        ["currency", "signal_family", "strategy_id", "date"]
    )
    policy_frame = pd.DataFrame(policies).drop_duplicates().sort_values(
        ["signal_family", "strategy_id"]
    )
    return signals, policy_frame


def scenario_outcomes(outcomes: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Add scenario-specific hit; benefit always follows the case definition."""
    result = outcomes.copy()
    if scenario == "favourable_now":
        result["target_hit"] = result["hit"]
        result["target_definition"] = "no_lower_price_in_next_h_observations"
    elif scenario == "window_closing":
        result["target_hit"] = (result["forward_change_bps"] > 0).astype(float)
        result["target_definition"] = "rub_per_unit_at_T_plus_h_is_higher_than_at_T"
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return result


def level_persistence_outcomes(
    outcomes: pd.DataFrame,
    series: pd.Series,
    dates: pd.DatetimeIndex,
    parameters: str,
    horizon: int,
) -> pd.DataFrame:
    """Literal level-message hit for T and every publication through T+h.

    At each date the percentile condition is recomputed from only the preceding
    L observations.  Future values are used solely to score a signal after it
    was emitted, never to construct the trigger at T.
    """
    parsed = json.loads(parameters)
    lookback = int(parsed["lookback"])
    higher_share = float(parsed["higher_share"])
    level_value = series.rolling(
        lookback + 1, min_periods=lookback + 1
    ).apply(
        lambda values: float(np.mean(values[:-1] > values[-1])), raw=True
    )
    level_value.index = dates
    condition_values = pd.concat(
        [level_value.shift(-step) for step in range(horizon + 1)],
        axis=1,
    )
    condition_valid = condition_values.notna().all(axis=1)
    condition_holds = condition_values.ge(higher_share).all(axis=1)
    result = outcomes.loc[
        condition_valid.reindex(outcomes.index).fillna(False).to_numpy()
    ].copy()
    result["target_hit"] = condition_holds.reindex(result.index).astype(float)
    result["target_definition"] = (
        "level_condition_holds_at_T_and_every_observation_through_T_plus_h"
    )
    return result


def excluded_momentum_audit(
    prices: pd.DataFrame,
    evaluation_start: pd.Timestamp,
) -> pd.DataFrame:
    """Count deliberately excluded long streaks without evaluating outcomes."""
    rows: list[dict[str, object]] = []
    for currency, group in prices.groupby("currency", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        for streak_days in EXCLUDED_MOMENTUM_STREAKS:
            feature = compute_candidate(
                group["rub_per_unit"], Candidate("spike", streak_days, 0.0)
            )
            count = int(
                (
                    feature["raw_signal"]
                    & group["date"].ge(evaluation_start)
                ).sum()
            )
            rows.append(
                {
                    "currency": currency,
                    "signal_family": "spike",
                    "strategy_id": f"momentum_streak_{streak_days}d__spike_only",
                    "parameters": json.dumps(
                        {"consecutive_declines": streak_days},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "signal_count": count,
                    "evaluation_status": "excluded_insufficient_events",
                    "exclusion_reason": (
                        "zero events"
                        if count == 0
                        else "too few events for statistical evaluation"
                    ),
                }
            )
    return pd.DataFrame(rows)


def policy_coverage(
    signals: pd.DataFrame,
    policies: pd.DataFrame,
    signal_source: str,
) -> pd.DataFrame:
    """Inventory policy/corridor combinations, including zero-signal cells."""
    if signal_source == "spike":
        policy_index = policies.rename(
            columns={"candidate_id": "base_strategy_id"}
        ).assign(
            strategy_id=lambda frame: frame["base_strategy_id"] + "__spike_only",
            signal_family="spike",
            scenario="favourable_now",
        )[["strategy_id", "signal_family", "scenario", "parameters"]]
    elif signal_source == "alternative":
        policy_index = policies[
            ["strategy_id", "signal_family", "scenario", "parameters"]
        ].drop_duplicates()
    else:
        raise ValueError(f"Unknown signal source: {signal_source}")

    currencies = pd.DataFrame({"currency": TARGET_CURRENCIES})
    inventory = policy_index.merge(currencies, how="cross")
    counts = (
        signals.groupby(["currency", "strategy_id"], as_index=False)
        .size()
        .rename(columns={"size": "signal_count"})
    )
    inventory = inventory.merge(counts, on=["currency", "strategy_id"], how="left")
    inventory["signal_count"] = inventory["signal_count"].fillna(0).astype(int)
    inventory["evaluation_status"] = np.where(
        inventory["signal_count"].gt(0), "evaluated", "excluded_zero_signals"
    )
    inventory["exclusion_reason"] = np.where(
        inventory["signal_count"].gt(0), "", "no signals in this corridor"
    )
    return inventory.sort_values(
        ["signal_family", "currency", "strategy_id"]
    ).reset_index(drop=True)


def random_sample_means(
    pool: np.ndarray,
    signal_count: int,
    repeats: int,
    seed: int,
) -> np.ndarray:
    if signal_count <= 0 or signal_count > len(pool):
        raise ValueError("Invalid random sample size")
    if signal_count == len(pool):
        return np.repeat(pool.mean(axis=0, keepdims=True), repeats, axis=0)
    rng = np.random.default_rng(seed)
    keys = rng.random((repeats, len(pool)))
    chosen = np.argpartition(keys, kth=signal_count - 1, axis=1)[:, :signal_count]
    return pool[chosen].mean(axis=1)


def evaluate_level_corridor(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    horizons: tuple[int, ...],
    repeats: int,
    seed: int,
    evaluation_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Evaluate hits and case-defined benefit against identical random dates."""
    rows: list[dict[str, object]] = []
    for currency, price_group in prices.groupby("currency", sort=True):
        price_group = price_group.sort_values("date").reset_index(drop=True)
        dates = pd.DatetimeIndex(price_group["date"])
        currency_signals = signals.loc[signals["currency"].eq(currency)]
        period_date_mask = dates >= evaluation_start
        if evaluation_end is not None:
            period_date_mask &= dates < evaluation_end
        period_weeks = pd.MultiIndex.from_arrays(
            [
                dates[period_date_mask].isocalendar().year.to_numpy(),
                dates[period_date_mask].isocalendar().week.to_numpy(),
            ]
        ).drop_duplicates()
        weeks = len(period_weeks)
        for horizon in horizons:
            base_all = outcome_frame(price_group["rub_per_unit"], dates, horizon)
            base = base_all.loc[
                base_all["valid"]
                & base_all.index.to_series().ge(evaluation_start).to_numpy()
                & (True if evaluation_end is None else base_all.index < evaluation_end)
            ]
            for strategy_id, strategy_signals in currency_signals.groupby("strategy_id"):
                first = strategy_signals.iloc[0]
                scenario = str(first["scenario"])
                outcomes = scenario_outcomes(base, scenario)
                period_signals = strategy_signals.loc[
                    strategy_signals["date"].ge(evaluation_start)
                    & (
                        True
                        if evaluation_end is None
                        else strategy_signals["date"].lt(evaluation_end)
                    )
                ]
                if period_signals.empty:
                    continue
                selected = strategy_signals.set_index("date").join(outcomes, how="inner")
                if selected.empty:
                    continue
                columns = ["target_hit", "forward_change_bps"]
                random = random_sample_means(
                    outcomes[columns].to_numpy(float),
                    len(selected),
                    repeats,
                    deterministic_seed(
                        seed, "alternative", currency, strategy_id, horizon,
                        evaluation_start, evaluation_end,
                    ),
                )
                hit_rate = float(selected["target_hit"].mean())
                random_hit = float(random[:, 0].mean())
                benefit = float(selected["benefit_bps"].mean())
                benefit_inference = monthly_block_bootstrap_mean(
                    selected["benefit_bps"],
                    repeats,
                    deterministic_seed(
                        seed, "alternative_benefit_monthly_bootstrap", currency,
                        strategy_id, horizon, evaluation_start, evaluation_end,
                    ),
                )
                signal_dates = pd.DatetimeIndex(period_signals["date"]).sort_values()
                gaps = np.diff(signal_dates.values).astype("timedelta64[D]").astype(int)
                rows.append(
                    {
                        "currency": currency,
                        "strategy_id": strategy_id,
                        "signal_family": first["signal_family"],
                        "scenario": scenario,
                        "parameters": first["parameters"],
                        "horizon": horizon,
                        "target_definition": outcomes["target_definition"].iloc[0],
                        "signal_count": len(period_signals),
                        "evaluated_signal_count": len(selected),
                        "signals_per_week": len(period_signals) / weeks,
                        "hit_rate": hit_rate,
                        "random_hit_rate_mean": random_hit,
                        "hit_lift": hit_rate / random_hit if random_hit else np.nan,
                        "hit_p_value": (1 + (random[:, 0] >= hit_rate).sum()) / (repeats + 1),
                        # Required benefit from the case: T vs mean(T-h ... T+h).
                        "benefit_bps": benefit,
                        "benefit_ci_low_95": benefit_inference["ci_low"],
                        "benefit_ci_high_95": benefit_inference["ci_high"],
                        "benefit_p_value_vs_zero": benefit_inference["p_value"],
                        # Optional waiting-cost diagnostic, never the main benefit.
                        "forward_change_bps": float(selected["forward_change_bps"].mean()),
                        "random_forward_change_bps_mean": float(random[:, 1].mean()),
                        "median_gap_days": float(np.median(gaps)) if len(gaps) else np.nan,
                        "max_gap_days": float(np.max(gaps)) if len(gaps) else np.nan,
                        "first_signal": signal_dates.min(),
                        "last_signal": signal_dates.max(),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["horizon", "signal_family", "currency", "strategy_id"]
    ).reset_index(drop=True)


def build_level_corridor_walk_forward_signals(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    selection_horizon: int,
    min_train_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    deployed: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []
    first_fold = pd.Timestamp(
        year=(evaluation_start + pd.DateOffset(years=min_train_years)).year,
        month=1,
        day=1,
    )
    groups = (
        ("level", "favourable_now"),
        ("corridor_exit_down", "favourable_now"),
        ("corridor_exit_up", "window_closing"),
    )
    for currency, price_group in prices.groupby("currency", sort=True):
        price_group = price_group.sort_values("date").reset_index(drop=True)
        base = outcome_frame(
            price_group["rub_per_unit"],
            pd.DatetimeIndex(price_group["date"]),
            selection_horizon,
        )
        last_date = pd.Timestamp(price_group["date"].max())
        currency_signals = signals.loc[signals["currency"].eq(currency)]
        fold_start = first_fold
        while fold_start <= last_date:
            fold_end = fold_start + pd.DateOffset(years=1)
            matured = base.loc[
                base["valid"]
                & base.index.to_series().ge(evaluation_start).to_numpy()
                & base["label_available_date"].lt(fold_start)
            ]
            train_weeks = len(matured[["iso_year", "iso_week"]].drop_duplicates())
            for family, scenario in groups:
                family_signals = currency_signals.loc[
                    currency_signals["signal_family"].eq(family)
                ]
                candidates: list[dict[str, object]] = []
                for strategy_id, strategy_signals in family_signals.groupby("strategy_id"):
                    first = strategy_signals.iloc[0]
                    outcomes = scenario_outcomes(matured, scenario)
                    if outcomes.empty:
                        continue
                    baseline_hit = float(outcomes["target_hit"].mean())
                    train = strategy_signals.set_index("date").join(outcomes, how="inner")
                    if train.empty:
                        continue
                    frequency = len(train) / train_weeks
                    hit_rate = float(train["target_hit"].mean())
                    hit_lift = hit_rate / baseline_hit if baseline_hit else np.nan
                    benefit = float(train["benefit_bps"].mean())
                    candidates.append(
                        {
                            "strategy_id": strategy_id,
                            "train_count": len(train),
                            "train_signals_per_week": frequency,
                            "train_hit_rate": hit_rate,
                            "train_random_hit_rate": baseline_hit,
                            "train_hit_lift": hit_lift,
                            "train_benefit_bps": benefit,
                            "frequency_gate": 1 <= frequency <= 2,
                            "sample_gate": len(train) >= 30,
                            "benefit_gate": benefit >= -5,
                            "lift_gate": hit_lift >= 1.2,
                            "frequency_distance": max(1 - frequency, frequency - 2, 0),
                        }
                    )
                if not candidates:
                    continue
                ranking = pd.DataFrame(candidates).sort_values(
                    ["sample_gate", "lift_gate", "benefit_gate",
                     "train_hit_lift", "train_benefit_bps"],
                    ascending=[False, False, False, False, False],
                )
                chosen = ranking.iloc[0].to_dict()
                chosen.update(
                    {
                        "currency": currency,
                        "signal_family": family,
                        "scenario": scenario,
                        "selection_horizon": selection_horizon,
                        "fold_start": fold_start,
                        "fold_end": min(fold_end, last_date + pd.offsets.Day(1)),
                    }
                )
                selections.append(chosen)
                fold_signals = family_signals.loc[
                    family_signals["strategy_id"].eq(chosen["strategy_id"])
                    & family_signals["date"].ge(fold_start)
                    & family_signals["date"].lt(fold_end)
                ].copy()
                if fold_signals.empty:
                    continue
                fold_signals["selected_strategy_id"] = chosen["strategy_id"]
                fold_signals["selection_horizon"] = selection_horizon
                fold_signals["selection_date"] = fold_start
                fold_signals["fold_year"] = fold_start.year
                fold_signals["strategy_id"] = (
                    f"walk_forward__{family}__h{selection_horizon}"
                )
                deployed.append(fold_signals)
            fold_start = fold_end
    return (
        pd.concat(deployed, ignore_index=True) if deployed else pd.DataFrame(),
        pd.DataFrame(selections),
    )


def evaluate_recent_level_corridor(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    recent_years: tuple[int, ...],
    horizons: tuple[int, ...],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    data_as_of = pd.Timestamp(prices["date"].max())
    rows: list[pd.DataFrame] = []
    for years in recent_years:
        period_start = data_as_of - pd.DateOffset(years=years)
        metrics = evaluate_level_corridor(
            prices, signals, period_start, horizons, repeats, seed
        )
        if metrics.empty:
            continue
        metrics.insert(0, "recent_years", years)
        metrics.insert(1, "requested_period_start", period_start)
        metrics.insert(2, "data_as_of", data_as_of)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def screen_walk_forward_families(
    spike_metrics: pd.DataFrame,
    alternative_metrics: pd.DataFrame,
    currencies: tuple[str, ...] = TARGET_CURRENCIES,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    lift_floor: float = 1.2,
    benefit_floor_bps: float = -5.0,
) -> pd.DataFrame:
    """Apply the current hard-signal gate to honest walk-forward results.

    Frequency is deliberately not part of this screen.  A family passes only
    when every requested currency/horizon cell exists and both quality floors
    hold in every cell.  The hit rates shown in the result are taken from the
    exact cell that determines the worst lift, so their ratio is interpretable.
    """
    spike = spike_metrics.copy()
    if not spike.empty:
        spike["signal_family"] = "momentum"
    metrics = pd.concat([spike, alternative_metrics.copy()], ignore_index=True)
    required = metrics.loc[
        metrics["currency"].isin(currencies)
        & metrics["horizon"].isin(horizons)
    ].copy()
    expected_cells = len(currencies) * len(horizons)
    rows: list[dict[str, object]] = []
    for family, group in required.groupby("signal_family", sort=True):
        group = group.drop_duplicates(["currency", "horizon"], keep="last")
        worst_lift_row = group.loc[group["hit_lift"].idxmin()]
        worst_benefit_row = group.loc[group["benefit_bps"].idxmin()]
        complete = (
            len(group) == expected_cells
            and group["currency"].nunique() == len(currencies)
            and group["horizon"].nunique() == len(horizons)
        )
        min_lift = float(worst_lift_row["hit_lift"])
        min_benefit = float(worst_benefit_row["benefit_bps"])
        rows.append(
            {
                "signal_family": family,
                "evaluated_cells": len(group),
                "expected_cells": expected_cells,
                "coverage_complete": complete,
                "min_hit_lift": min_lift,
                "median_hit_lift": float(group["hit_lift"].median()),
                "worst_lift_currency": worst_lift_row["currency"],
                "worst_lift_horizon": int(worst_lift_row["horizon"]),
                "hit_rate_signal_at_worst_lift": float(worst_lift_row["hit_rate"]),
                "hit_rate_random_at_worst_lift": float(
                    worst_lift_row["random_hit_rate_mean"]
                ),
                "min_benefit_bps": min_benefit,
                "median_benefit_bps": float(group["benefit_bps"].median()),
                "worst_benefit_currency": worst_benefit_row["currency"],
                "worst_benefit_horizon": int(worst_benefit_row["horizon"]),
                "lift_floor": lift_floor,
                "benefit_floor_bps": benefit_floor_bps,
                "passes_hard_screen": bool(
                    complete
                    and min_lift >= lift_floor
                    and min_benefit >= benefit_floor_bps
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["passes_hard_screen", "min_hit_lift"], ascending=[False, False]
    ).reset_index(drop=True)


def relaxed_walk_forward_candidates(
    spike_metrics: pd.DataFrame,
    alternative_metrics: pd.DataFrame,
    lift_floor: float = 1.15,
) -> pd.DataFrame:
    """Return honest OOT cells passing a relaxed lift and positive mean benefit."""
    spike = spike_metrics.copy()
    if not spike.empty:
        spike["signal_family"] = "momentum"
    metrics = pd.concat([spike, alternative_metrics.copy()], ignore_index=True)
    selected = metrics.loc[
        metrics["hit_lift"].ge(lift_floor) & metrics["benefit_bps"].ge(0)
    ].copy()
    selected["lift_floor"] = lift_floor
    selected["benefit_is_significant"] = (
        selected["benefit_ci_low_95"].gt(0)
        & selected["benefit_p_value_vs_zero"].lt(0.05)
    )
    columns = [
        "currency", "signal_family", "horizon", "hit_rate",
        "random_hit_rate_mean", "hit_lift", "benefit_bps",
        "benefit_ci_low_95", "benefit_ci_high_95",
        "benefit_p_value_vs_zero", "benefit_is_significant",
        "signals_per_week", "evaluated_signal_count", "lift_floor",
    ]
    return selected[columns].sort_values(
        ["benefit_is_significant", "horizon", "currency", "hit_lift"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)


def screen_fixed_level_policies(
    metrics: pd.DataFrame,
    currencies: tuple[str, ...] = TARGET_CURRENCIES,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    lift_floor: float = 1.2,
    benefit_floor_bps: float = -5.0,
) -> pd.DataFrame:
    """Shortlist fixed level parameters across every currency/horizon cell.

    ``passes_hard_screen`` follows the team's exploratory tolerances.  The
    stricter ``passes_case_screen`` additionally applies the case's 1.3 lift
    target and requires the benefit 95% lower bound to be above zero in every
    cell.  This retrospective screen is descriptive, not a substitute for
    walk-forward validation.
    """
    level = metrics.loc[
        metrics["signal_family"].eq("level")
        & metrics["currency"].isin(currencies)
        & metrics["horizon"].isin(horizons)
    ].copy()
    expected_cells = len(currencies) * len(horizons)
    rows: list[dict[str, object]] = []
    for strategy_id, group in level.groupby("strategy_id", sort=True):
        group = group.drop_duplicates(["currency", "horizon"], keep="last")
        lift_row = group.loc[group["hit_lift"].idxmin()]
        benefit_row = group.loc[group["benefit_bps"].idxmin()]
        complete = (
            len(group) == expected_cells
            and group["currency"].nunique() == len(currencies)
            and group["horizon"].nunique() == len(horizons)
        )
        min_lift = float(lift_row["hit_lift"])
        min_benefit = float(benefit_row["benefit_bps"])
        min_ci_low = float(group["benefit_ci_low_95"].min())
        rows.append(
            {
                "strategy_id": strategy_id,
                "parameters": group["parameters"].iloc[0],
                "evaluated_cells": len(group),
                "expected_cells": expected_cells,
                "coverage_complete": complete,
                "min_hit_lift": min_lift,
                "median_hit_lift": float(group["hit_lift"].median()),
                "worst_lift_currency": lift_row["currency"],
                "worst_lift_horizon": int(lift_row["horizon"]),
                "min_benefit_bps": min_benefit,
                "median_benefit_bps": float(group["benefit_bps"].median()),
                "min_benefit_ci_low_95": min_ci_low,
                "worst_benefit_currency": benefit_row["currency"],
                "worst_benefit_horizon": int(benefit_row["horizon"]),
                "passes_hard_screen": bool(
                    complete
                    and min_lift >= lift_floor
                    and min_benefit >= benefit_floor_bps
                ),
                "passes_case_screen": bool(
                    complete and min_lift >= 1.3 and min_ci_low > 0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["passes_hard_screen", "min_hit_lift"], ascending=[False, False]
    ).reset_index(drop=True)


def screen_fixed_level_policies_by_currency(
    metrics: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    lift_floor: float = 1.2,
    benefit_floor_bps: float = -5.0,
) -> pd.DataFrame:
    """Find corridor-specific candidates that pass every requested horizon."""
    level = metrics.loc[
        metrics["signal_family"].eq("level")
        & metrics["horizon"].isin(horizons)
    ].copy()
    rows: list[dict[str, object]] = []
    for (currency, strategy_id), group in level.groupby(
        ["currency", "strategy_id"], sort=True
    ):
        group = group.drop_duplicates("horizon", keep="last")
        complete = len(group) == len(horizons)
        min_lift = float(group["hit_lift"].min())
        min_benefit = float(group["benefit_bps"].min())
        min_ci_low = float(group["benefit_ci_low_95"].min())
        rows.append(
            {
                "currency": currency,
                "strategy_id": strategy_id,
                "parameters": group["parameters"].iloc[0],
                "evaluated_horizons": len(group),
                "min_hit_lift": min_lift,
                "median_hit_lift": float(group["hit_lift"].median()),
                "min_benefit_bps": min_benefit,
                "min_benefit_ci_low_95": min_ci_low,
                "signals_per_week": float(group["signals_per_week"].iloc[0]),
                "passes_hard_screen": bool(
                    complete
                    and min_lift >= lift_floor
                    and min_benefit >= benefit_floor_bps
                ),
                "passes_case_screen": bool(
                    complete and min_lift >= 1.3 and min_ci_low > 0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["passes_hard_screen", "currency", "min_hit_lift"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def build_corridor_qualified_level_union(
    prices: pd.DataFrame,
    level_signals: pd.DataFrame,
    corridor_screen: pd.DataFrame,
    report_start: pd.Timestamp,
    cooldown_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union corridor-specific soft-pass policies and report their frequency."""
    qualified = corridor_screen.loc[
        corridor_screen["passes_hard_screen"], ["currency", "strategy_id"]
    ].drop_duplicates()
    source = level_signals.merge(
        qualified.assign(_qualified=True),
        on=["currency", "strategy_id"],
        how="inner",
    )
    # Reuse the union implementation by supplying a global-looking screen for
    # the already corridor-filtered signal frame.
    pseudo_screen = pd.DataFrame(
        {"strategy_id": source["strategy_id"].unique(), "passes_hard_screen": True}
    )
    events, frequency = build_qualified_level_union(
        prices,
        source,
        pseudo_screen,
        report_start,
        cooldown_observations,
    )
    counts = qualified.groupby("currency").size()
    frequency["qualified_policy_count"] = (
        frequency["currency"].map(counts).fillna(0).astype(int)
    )
    return events, frequency


def build_qualified_level_union(
    prices: pd.DataFrame,
    level_signals: pd.DataFrame,
    policy_screen: pd.DataFrame,
    report_start: pd.Timestamp,
    cooldown_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union qualifying fixed level triggers, deduplicate dates, then cooldown."""
    qualified = set(
        policy_screen.loc[policy_screen["passes_hard_screen"], "strategy_id"]
    )
    source = level_signals.loc[
        level_signals["strategy_id"].isin(qualified)
        & level_signals["date"].ge(report_start)
    ].copy()
    if source.empty:
        events = pd.DataFrame(
            columns=["date", "currency", "matching_strategy_ids", "policy_count"]
        )
    else:
        union = (
            source.groupby(["currency", "date"], as_index=False)
            .agg(
                matching_strategy_ids=(
                    "strategy_id", lambda values: "|".join(sorted(set(values)))
                ),
                policy_count=("strategy_id", "nunique"),
            )
        )
        kept: list[pd.DataFrame] = []
        for currency, price_group in prices.groupby("currency", sort=True):
            market_dates = pd.DatetimeIndex(
                price_group.loc[price_group["date"].ge(report_start), "date"]
            )
            raw_dates = set(union.loc[union["currency"].eq(currency), "date"])
            mask = pd.Series(market_dates.isin(raw_dates))
            keep_positions = thin_with_cooldown(mask, cooldown_observations)
            keep_dates = set(market_dates[keep_positions])
            kept.append(
                union.loc[
                    union["currency"].eq(currency) & union["date"].isin(keep_dates)
                ]
            )
        events = pd.concat(kept, ignore_index=True).sort_values(["currency", "date"])

    frequency_rows: list[dict[str, object]] = []
    for currency, price_group in prices.groupby("currency", sort=True):
        market_dates = pd.DatetimeIndex(
            price_group.loc[price_group["date"].ge(report_start), "date"]
        )
        weeks = len(
            pd.MultiIndex.from_arrays(
                [market_dates.isocalendar().year, market_dates.isocalendar().week]
            ).drop_duplicates()
        )
        count = int(events["currency"].eq(currency).sum()) if not events.empty else 0
        frequency_rows.append(
            {
                "currency": currency,
                "qualified_policy_count": len(qualified),
                "push_count": count,
                "weeks": weeks,
                "pushes_per_week": count / weeks if weeks else np.nan,
            }
        )
    return events, pd.DataFrame(frequency_rows)


def parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(sorted({int(item) for item in value.split(",")}))
    if not horizons or min(horizons) <= 0:
        raise argparse.ArgumentTypeError("Horizons must be positive comma-separated integers")
    return horizons


def parse_recent_years(value: str) -> tuple[int, ...]:
    years = tuple(sorted({int(item) for item in value.split(",")}))
    if not years or min(years) <= 0:
        raise argparse.ArgumentTypeError("Recent years must be positive comma-separated integers")
    return years


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/hard_signals"))
    parser.add_argument("--evaluation-start", type=pd.Timestamp, default=pd.Timestamp("2014-01-01"))
    parser.add_argument("--horizons", type=parse_horizons, default=DEFAULT_HORIZONS)
    parser.add_argument(
        "--research-horizons",
        type=parse_horizons,
        default=DEFAULT_RESEARCH_HORIZONS,
        help="Supplemental horizons for exploratory policy trade-off plots",
    )
    parser.add_argument("--cooldown-observations", type=int, default=3)
    parser.add_argument("--random-repeats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-train-years", type=int, default=4)
    parser.add_argument(
        "--evaluation-years",
        type=int,
        default=DEFAULT_EVALUATION_YEARS,
        help="Publish metrics only for this many trailing years",
    )
    parser.add_argument(
        "--recent-years",
        type=parse_recent_years,
        default=DEFAULT_RECENT_YEARS,
        help="Rolling out-of-time reporting windows ending at the latest data date",
    )
    args = parser.parse_args()

    input_path = args.input or latest_default_input()
    prices = load_prices(input_path)
    data_as_of = pd.Timestamp(prices["date"].max())
    report_start = data_as_of - pd.DateOffset(years=args.evaluation_years)
    signals, candidates = build_signals(
        prices,
        args.evaluation_start,
        args.cooldown_observations,
    )
    metrics = evaluate_strategies(
        prices,
        signals,
        report_start,
        args.horizons,
        args.random_repeats,
        args.seed,
        args.cooldown_observations,
    )
    missing_research_horizons = tuple(
        horizon for horizon in args.research_horizons if horizon not in args.horizons
    )
    supplemental_metrics = (
        evaluate_strategies(
            prices,
            signals,
            report_start,
            missing_research_horizons,
            args.random_repeats,
            args.seed,
            args.cooldown_observations,
        )
        if missing_research_horizons
        else pd.DataFrame()
    )
    research_metrics = pd.concat(
        [
            metrics.loc[metrics["horizon"].isin(args.research_horizons)],
            supplemental_metrics,
        ],
        ignore_index=True,
    ).sort_values(["horizon", "currency", "strategy_id"])
    spike_deployments: list[pd.DataFrame] = []
    spike_selections: list[pd.DataFrame] = []
    spike_metric_parts: list[pd.DataFrame] = []
    spike_recent_parts: list[pd.DataFrame] = []
    for selection_horizon in args.horizons:
        deployed, selected = build_walk_forward_signals(
            prices,
            signals,
            args.evaluation_start,
            selection_horizon,
            args.min_train_years,
        )
        if not selected.empty:
            spike_selections.append(selected)
        if deployed.empty:
            continue
        spike_deployments.append(deployed)
        overall = evaluate_strategies(
            prices,
            deployed,
            report_start,
            (selection_horizon,),
            args.random_repeats,
            args.seed,
            args.cooldown_observations,
        )
        overall.insert(0, "fold_year", f"last_{args.evaluation_years}_years")
        spike_metric_parts.append(overall)
        spike_recent_parts.append(
            evaluate_recent_walk_forward(
                prices,
                deployed,
                args.recent_years,
                (selection_horizon,),
                args.random_repeats,
                args.seed,
                args.cooldown_observations,
            )
        )
    walk_forward_signals = (
        pd.concat(spike_deployments, ignore_index=True)
        if spike_deployments else pd.DataFrame()
    )
    walk_forward_selections = (
        pd.concat(spike_selections, ignore_index=True)
        if spike_selections else pd.DataFrame()
    )
    walk_forward_metrics = (
        pd.concat([part for part in spike_metric_parts if not part.empty], ignore_index=True)
        if any(not part.empty for part in spike_metric_parts) else pd.DataFrame()
    )
    recent_walk_forward_metrics = (
        pd.concat([part for part in spike_recent_parts if not part.empty], ignore_index=True)
        if any(not part.empty for part in spike_recent_parts) else pd.DataFrame()
    )

    alternative_signals, alternative_policies = build_level_corridor_signals(
        prices, args.evaluation_start, args.cooldown_observations
    )
    alternative_metrics = evaluate_level_corridor(
        prices,
        alternative_signals,
        report_start,
        args.horizons,
        args.random_repeats,
        args.seed,
    )
    alternative_supplemental = (
        evaluate_level_corridor(
            prices,
            alternative_signals,
            report_start,
            missing_research_horizons,
            args.random_repeats,
            args.seed,
        )
        if missing_research_horizons
        else pd.DataFrame()
    )
    alternative_research_metrics = pd.concat(
        [
            alternative_metrics.loc[
                alternative_metrics["horizon"].isin(args.research_horizons)
            ],
            alternative_supplemental,
        ],
        ignore_index=True,
    ).sort_values(["horizon", "signal_family", "currency", "strategy_id"])
    alternative_deployments: list[pd.DataFrame] = []
    alternative_selections: list[pd.DataFrame] = []
    alternative_metric_parts: list[pd.DataFrame] = []
    alternative_recent_parts: list[pd.DataFrame] = []
    for selection_horizon in args.horizons:
        deployed, selected = build_level_corridor_walk_forward_signals(
            prices,
            alternative_signals,
            args.evaluation_start,
            selection_horizon,
            args.min_train_years,
        )
        if not selected.empty:
            alternative_selections.append(selected)
        if deployed.empty:
            continue
        alternative_deployments.append(deployed)
        alternative_metric_parts.append(
            evaluate_level_corridor(
                prices,
                deployed,
                report_start,
                (selection_horizon,),
                args.random_repeats,
                args.seed,
            )
        )
        alternative_recent_parts.append(
            evaluate_recent_level_corridor(
                prices,
                deployed,
                args.recent_years,
                (selection_horizon,),
                args.random_repeats,
                args.seed,
            )
        )
    alternative_walk_forward_signals = (
        pd.concat(alternative_deployments, ignore_index=True)
        if alternative_deployments else pd.DataFrame()
    )
    alternative_walk_forward_selections = (
        pd.concat(alternative_selections, ignore_index=True)
        if alternative_selections else pd.DataFrame()
    )
    alternative_walk_forward_metrics = (
        pd.concat(
            [part for part in alternative_metric_parts if not part.empty],
            ignore_index=True,
        )
        if any(not part.empty for part in alternative_metric_parts) else pd.DataFrame()
    )
    alternative_recent_walk_forward_metrics = (
        pd.concat(
            [part for part in alternative_recent_parts if not part.empty],
            ignore_index=True,
        )
        if any(not part.empty for part in alternative_recent_parts) else pd.DataFrame()
    )

    coverage = pd.concat(
        [
            policy_coverage(
                signals.loc[signals["date"].ge(report_start)], candidates, "spike"
            ),
            policy_coverage(
                alternative_signals.loc[alternative_signals["date"].ge(report_start)],
                alternative_policies,
                "alternative",
            ),
            excluded_momentum_audit(prices, report_start),
        ],
        ignore_index=True,
    ).sort_values(["signal_family", "currency", "strategy_id"])
    hard_signal_screen = screen_walk_forward_families(
        walk_forward_metrics,
        alternative_walk_forward_metrics,
        TARGET_CURRENCIES,
        args.horizons,
    )
    relaxed_candidates = relaxed_walk_forward_candidates(
        walk_forward_metrics,
        alternative_walk_forward_metrics,
    )
    level_policy_screen = screen_fixed_level_policies(
        alternative_metrics,
        TARGET_CURRENCIES,
        args.horizons,
    )
    qualified_level_union, qualified_level_frequency = build_qualified_level_union(
        prices,
        alternative_signals.loc[alternative_signals["signal_family"].eq("level")],
        level_policy_screen,
        report_start,
        args.cooldown_observations,
    )
    corridor_level_policy_screen = screen_fixed_level_policies_by_currency(
        alternative_metrics,
        args.horizons,
    )
    corridor_level_union, corridor_level_frequency = (
        build_corridor_qualified_level_union(
            prices,
            alternative_signals.loc[
                alternative_signals["signal_family"].eq("level")
            ],
            corridor_level_policy_screen,
            report_start,
            args.cooldown_observations,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    signals.to_csv(
        args.output_dir / "signals.csv.gz",
        index=False,
        lineterminator="\n",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    metrics.to_csv(args.output_dir / "metrics.csv", index=False, lineterminator="\n")
    research_metrics.to_csv(
        args.output_dir / "research_metrics.csv", index=False, lineterminator="\n"
    )
    candidates.to_csv(args.output_dir / "candidates.csv", index=False, lineterminator="\n")
    walk_forward_signals.to_csv(
        args.output_dir / "walk_forward_signals.csv.gz",
        index=False,
        lineterminator="\n",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    walk_forward_selections.to_csv(
        args.output_dir / "walk_forward_selections.csv", index=False, lineterminator="\n"
    )
    walk_forward_metrics.to_csv(
        args.output_dir / "walk_forward_metrics.csv", index=False, lineterminator="\n"
    )
    recent_walk_forward_metrics.to_csv(
        args.output_dir / "recent_walk_forward_metrics.csv",
        index=False,
        lineterminator="\n",
    )
    alternative_signals.to_csv(
        args.output_dir / "alternative_signals.csv.gz",
        index=False,
        lineterminator="\n",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    alternative_policies.to_csv(
        args.output_dir / "alternative_policies.csv", index=False, lineterminator="\n"
    )
    alternative_metrics.to_csv(
        args.output_dir / "alternative_metrics.csv", index=False, lineterminator="\n"
    )
    alternative_research_metrics.to_csv(
        args.output_dir / "alternative_research_metrics.csv",
        index=False,
        lineterminator="\n",
    )
    alternative_walk_forward_signals.to_csv(
        args.output_dir / "alternative_walk_forward_signals.csv.gz",
        index=False,
        lineterminator="\n",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    alternative_walk_forward_selections.to_csv(
        args.output_dir / "alternative_walk_forward_selections.csv",
        index=False,
        lineterminator="\n",
    )
    alternative_walk_forward_metrics.to_csv(
        args.output_dir / "alternative_walk_forward_metrics.csv",
        index=False,
        lineterminator="\n",
    )
    alternative_recent_walk_forward_metrics.to_csv(
        args.output_dir / "alternative_recent_walk_forward_metrics.csv",
        index=False,
        lineterminator="\n",
    )
    coverage.to_csv(
        args.output_dir / "policy_coverage.csv", index=False, lineterminator="\n"
    )
    hard_signal_screen.to_csv(
        args.output_dir / "hard_signal_screen.csv", index=False, lineterminator="\n"
    )
    relaxed_candidates.to_csv(
        args.output_dir / "relaxed_walk_forward_candidates.csv",
        index=False,
        lineterminator="\n",
    )
    level_policy_screen.to_csv(
        args.output_dir / "level_policy_screen.csv", index=False, lineterminator="\n"
    )
    qualified_level_union.to_csv(
        args.output_dir / "qualified_level_union.csv", index=False, lineterminator="\n"
    )
    qualified_level_frequency.to_csv(
        args.output_dir / "qualified_level_frequency.csv",
        index=False,
        lineterminator="\n",
    )
    corridor_level_policy_screen.to_csv(
        args.output_dir / "corridor_level_policy_screen.csv",
        index=False,
        lineterminator="\n",
    )
    corridor_level_union.to_csv(
        args.output_dir / "corridor_level_union.csv", index=False, lineterminator="\n"
    )
    corridor_level_frequency.to_csv(
        args.output_dir / "corridor_level_frequency.csv",
        index=False,
        lineterminator="\n",
    )
    metadata = {
        "input": str(input_path),
        "history_start": args.evaluation_start.date().isoformat(),
        "report_start": report_start.date().isoformat(),
        "data_as_of": data_as_of.date().isoformat(),
        "evaluation_years": args.evaluation_years,
        "horizons": list(args.horizons),
        "research_horizons": list(args.research_horizons),
        "cooldown_observations": args.cooldown_observations,
        "momentum_cooldown_observations": 0,
        "momentum_trigger": "first day on which rub_per_unit has declined for N consecutive published observations",
        "random_repeats": args.random_repeats,
        "seed": args.seed,
        "random_baseline_primary": "same evaluated signal count in the full currency-period",
        "random_baseline_robustness": "same evaluated signal count in each ISO publication week",
        "quality_scope": "three separately selected families: spike, level, stable-corridor exit",
        "favourable_now_hit_target": "no lower rub_per_unit appears in T+1 ... T+h; used by momentum, level and corridor exit down",
        "level_hit_target": "no lower rub_per_unit appears in T+1 ... T+h; the level condition is the causal trigger, not the target",
        "local_min_diagnostic": "T is minimum rub_per_unit in T-h ... T+h; diagnostic only",
        "spike_policy_grid_size": len(spike_policy_grid()),
        "level_policy_grid_size": len(level_policy_grid()),
        "corridor_exit_policy_grid_size": len(corridor_exit_policy_grid()),
        "benefit_definition": "(mean rub_per_unit in T-h...T+h / rub_per_unit at T - 1) * 10000",
        "benefit_inference": "95% calendar-month block-bootstrap CI and one-sided centered-bootstrap p-value for mean benefit > 0",
        "window_closing_hit_target": "rub_per_unit at T+h is higher than at T",
        "waiting_cost_diagnostic": "forward_change_bps only; never replaces required benefit_bps",
        "selection_horizons": list(args.horizons),
        "min_train_years": args.min_train_years,
        "recent_years": list(args.recent_years),
        "walk_forward_selection": "annual and separate for every required horizon; only labels matured before fold start",
        "selection_ranking": "sample gate, lift >= 1.2, benefit >= -5 bp, then continuous lift and benefit; frequency is diagnostic only",
        "hard_signal_screen": {
            "source": "walk-forward metrics in the trailing evaluation window",
            "lift_floor": 1.2,
            "benefit_floor_bps": -5.0,
            "stability_rule": "both floors must hold in every requested currency/horizon cell",
            "frequency_used_as_filter": False,
        },
        "relaxed_walk_forward_candidates": {
            "lift_floor": 1.15,
            "mean_benefit_floor_bps": 0.0,
            "benefit_significance_flag": "95% lower bound > 0 and one-sided p-value < 0.05",
        },
        "fixed_level_policy_screen": {
            "scope": "all requested currencies and horizons in the trailing evaluation window",
            "exploratory_rule": "min lift >= 1.2 and min benefit >= -5 bp",
            "case_rule": "min lift >= 1.3 and benefit 95% lower bound > 0 in every cell",
            "warning": "retrospective development screen; not a replacement for walk-forward",
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {len(signals):,} spike and {len(alternative_signals):,} level/corridor "
        f"research signals; {len(walk_forward_signals):,} spike and "
        f"{len(alternative_walk_forward_signals):,} level/corridor walk-forward signals "
        f"from {input_path} to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
