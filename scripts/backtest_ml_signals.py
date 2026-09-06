#!/usr/bin/env python3
"""Nested walk-forward boosting over every published FX date.

Correctness properties:
- every feature at T uses data available by T;
- training and inner-validation labels must have matured before the decision cutoff;
- model family and per-currency thresholds are chosen only inside the
  inner historical validation period;
- the following outer period is used once for OOT evaluation;
- arbitrary-date inference is available through --as-of.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from backtest_hard_signals import (
    TARGET_CURRENCIES,
    deterministic_seed,
    latest_default_input,
    load_prices,
    monthly_block_bootstrap_mean,
    outcome_frame,
    random_sample_means,
)

# Required horizons from the case.
HORIZONS = (1, 3, 5, 10, 20)
COUNTRY_BY_CURRENCY = {"AMD": "AM", "KGS": "KG", "KZT": "KZ", "TJS": "TJ", "UZS": "UZ"}
CURRENCY_COPY = {
    "AMD": {"genitive": "армянского драма", "unit": "драм", "country": "Армению"},
    "KGS": {"genitive": "киргизского сома", "unit": "сом", "country": "Кыргызстан"},
    "KZT": {"genitive": "казахстанского тенге", "unit": "тенге", "country": "Казахстан"},
    "TJS": {"genitive": "таджикского сомони", "unit": "сомони", "country": "Таджикистан"},
    "UZS": {"genitive": "узбекского сума", "unit": "сум", "country": "Узбекистан"},
}
PRODUCTION_SIGNAL_COLUMNS = [
    "date",
    "corridor",
    "indicator",
    "direction",
    "strength",
    "speed",
    "recommended_scenario",
    "signal_source",
    "explanation_indicator",
    "template_id",
    "push_title",
    "push_text",
    "currency",
    "rub_per_unit",
]
MODEL_CONFIGS = ("long_history", "rolling_4y")
TRAINING_WINDOW_CONFIGS = (
    "rolling_1y",
    "rolling_2y",
    "rolling_4y",
    "rolling_6y",
    "long_history",
)
CANDIDATE_GATES = (
    "all_days",
    "level60_q70",
    "level60_q80",
    "level60_q90",
)
CANDIDATE_HISTORY_CONFIG = "long_history"

# Threshold candidates are quantiles of inner-validation model scores.
# They are ranking cutoffs, not calibrated probabilities.
PROBABILITY_QUANTILES = (0.0, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.975)
LOCAL_QUANTILES = (0.0, 0.20, 0.50, 0.70, 0.85, 0.95)
BENEFIT_QUANTILES = (0.0, 0.30)
REGRET_QUANTILES = (0.025, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0)
JOINT_POLICY_QUANTILES = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
JOINT_ARCHITECTURES = ("two_head", "joint_classifier")
SCORE_PERCENTILE_LOOKBACK = 252
INNER_OOF_YEARS = 3
TUNING_SCORE_LOOKBACKS = (120, 252)
TUNING_MODEL_CONFIGS = {
    "baseline_long": {
        "history": "long_history", "learning_rate": 0.10, "max_iter": 35,
        "max_leaf_nodes": 9, "min_samples_leaf": 25, "l2_regularization": 3,
    },
    "regularized_long": {
        "history": "long_history", "learning_rate": 0.05, "max_iter": 80,
        "max_leaf_nodes": 7, "min_samples_leaf": 40, "l2_regularization": 10,
    },
    "flexible_long": {
        "history": "long_history", "learning_rate": 0.05, "max_iter": 80,
        "max_leaf_nodes": 15, "min_samples_leaf": 25, "l2_regularization": 5,
    },
}

# Deliberately curated instead of a Cartesian grid.  The previous experiment
# identified flexible_long as the useful neighbourhood; these candidates vary
# one or two capacity/regularisation controls around it while retaining several
# controls.  This keeps the nightly run finite and makes each comparison
# interpretable.
NIGHTLY_MODEL_CONFIGS = {
    **TUNING_MODEL_CONFIGS,
    "flexible_more_trees": {
        "history": "long_history", "learning_rate": 0.03, "max_iter": 160,
        "max_leaf_nodes": 15, "min_samples_leaf": 25, "l2_regularization": 5,
    },
    "flexible_fast": {
        "history": "long_history", "learning_rate": 0.08, "max_iter": 60,
        "max_leaf_nodes": 15, "min_samples_leaf": 25, "l2_regularization": 5,
    },
    "deep_regularized": {
        "history": "long_history", "learning_rate": 0.04, "max_iter": 140,
        "max_leaf_nodes": 31, "min_samples_leaf": 25, "l2_regularization": 10,
        "max_features": 0.80,
    },
    "deep_smooth": {
        "history": "long_history", "learning_rate": 0.04, "max_iter": 140,
        "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 15,
        "max_features": 0.80,
    },
    "deep_low_regularization": {
        "history": "long_history", "learning_rate": 0.05, "max_iter": 100,
        "max_leaf_nodes": 31, "min_samples_leaf": 15, "l2_regularization": 2,
    },
    "shallow_more_trees": {
        "history": "long_history", "learning_rate": 0.04, "max_iter": 140,
        "max_leaf_nodes": 7, "min_samples_leaf": 15, "l2_regularization": 3,
    },
    "flexible_smooth": {
        "history": "long_history", "learning_rate": 0.05, "max_iter": 100,
        "max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 8,
    },
    "flexible_feature_subsample": {
        "history": "long_history", "learning_rate": 0.05, "max_iter": 100,
        "max_leaf_nodes": 15, "min_samples_leaf": 25, "l2_regularization": 5,
        "max_features": 0.65,
    },
    "flexible_unbalanced": {
        "history": "long_history", "learning_rate": 0.05, "max_iter": 100,
        "max_leaf_nodes": 15, "min_samples_leaf": 25, "l2_regularization": 5,
        "class_weight": None,
    },
}
NIGHTLY_SCORE_LOOKBACKS = (60, 120, 252, 504)
NIGHTLY_POLICY_QUANTILES = (
    0.50, 0.60, 0.70, 0.75, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975,
)
NIGHTLY_HORIZONS = (1, 3, 5, 7, 8, 9, 10, 11, 12, 15, 20)

# Frequency and cooldown are deliberately NOT selection constraints in the
# current experiment. Frequency is reported only as an outcome diagnostic.
SELECTION_COOLDOWN = 0
MIN_VALIDATION_SIGNALS = 20

TRUTH_LIFT_TARGET = 1.15
LOCAL_LIFT_TARGET = 1.30
RANDOM_REPEATS = 300

CONTEXT_CURRENCIES = ("USD", "EUR", "CNY")
CROSS_H_TRAIN_HORIZON = 11
CROSS_H_EVAL_HORIZON = 10
CROSS_H_LEGACY_CHOICES = {
    2023: {"model_config": "deep_regularized", "score_lookback": 60, "joint_quantile": 0.825},
    2024: {"model_config": "deep_smooth", "score_lookback": 504, "joint_quantile": 0.800},
    2025: {"model_config": "deep_smooth", "score_lookback": 504, "joint_quantile": 0.800},
    2026: {"model_config": "baseline_long", "score_lookback": 504, "joint_quantile": 0.700},
}


def load_ml_prices(path: Path) -> pd.DataFrame:
    """Load target and causal FX-context series used by the pooled ML model."""
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "currency", "rub_per_unit"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    allowed = set(TARGET_CURRENCIES) | set(CONTEXT_CURRENCIES)
    frame = frame.loc[frame["currency"].isin(allowed), list(required)].copy()
    frame = frame.sort_values(["currency", "date"]).reset_index(drop=True)
    if frame.duplicated(["currency", "date"]).any():
        raise ValueError("Duplicate currency/date observations found")
    if frame["rub_per_unit"].isna().any() or (frame["rub_per_unit"] <= 0).any():
        raise ValueError("All rub_per_unit values must be finite and positive")
    absent_targets = set(TARGET_CURRENCIES).difference(frame["currency"].unique())
    if absent_targets:
        raise ValueError(f"Missing target currencies: {sorted(absent_targets)}")
    return frame


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_brent_features(path: Path, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Legacy ablation helper; Brent is not used in the current model."""
    raw = pd.read_csv(path, parse_dates=["observation_date"])
    raw["brent"] = pd.to_numeric(raw["DCOILBRENTEU"], errors="coerce")
    raw = raw.dropna(subset=["brent"]).sort_values("observation_date")
    raw["available_date"] = raw["observation_date"].map(lambda value: value + timedelta(days=3))
    aligned = pd.merge_asof(
        pd.DataFrame({"date": dates}).sort_values("date"),
        raw[["available_date", "brent"]].sort_values("available_date"),
        left_on="date",
        right_on="available_date",
        direction="backward",
    ).set_index("date")["brent"]
    result = pd.DataFrame(index=dates)
    returns = np.log(aligned).diff()
    for lag in (0, 1, 2, 5, 10):
        result[f"brent_return_lag_{lag}"] = returns.shift(lag)
    return result


def decline_streak(returns: pd.Series) -> pd.Series:
    values = returns.lt(0).fillna(False).to_numpy()
    result = np.zeros(len(values), dtype=float)
    run = 0
    for index, falling in enumerate(values):
        run = run + 1 if falling else 0
        result[index] = run
    return pd.Series(result, index=returns.index)


def rolling_slope(log_price: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denominator = float(np.square(x).sum())
    return log_price.rolling(window).apply(
        lambda values: float(np.dot(values - values.mean(), x) / denominator),
        raw=True,
    )


def periods_since_rolling_min(price: pd.Series, window: int) -> pd.Series:
    return price.rolling(window).apply(
        lambda values: float(len(values) - 1 - np.argmin(values)), raw=True
    )


def recipient_holiday_features(dates: pd.DatetimeIndex, country: str) -> pd.DataFrame:
    """Calendar-known recipient-country holiday features."""
    years = range(int(dates.min().year) - 1, int(dates.max().year) + 2)
    calendar = holidays.country_holidays(country, years=years)
    holiday_dates = np.array(
        sorted(pd.Timestamp(day).to_datetime64() for day in calendar),
        dtype="datetime64[D]",
    )
    date_days = dates.to_numpy(dtype="datetime64[D]")

    # Defensive fallback for an unexpectedly empty holiday calendar.
    if len(holiday_dates) == 0:
        return pd.DataFrame(
            {
                "recipient_holiday_today": 0.0,
                "days_to_recipient_holiday": 31.0,
                "days_since_recipient_holiday": 31.0,
                "recipient_holiday_next_7d": 0.0,
                "recipient_holiday_previous_3d": 0.0,
            },
            index=dates,
        )

    right = np.searchsorted(holiday_dates, date_days, side="left")
    next_index = np.minimum(right, len(holiday_dates) - 1)
    previous_index = np.maximum(
        np.searchsorted(holiday_dates, date_days, side="right") - 1,
        0,
    )
    days_to = (holiday_dates[next_index] - date_days).astype("timedelta64[D]").astype(int)
    days_since = (date_days - holiday_dates[previous_index]).astype("timedelta64[D]").astype(int)
    is_holiday = np.isin(date_days, holiday_dates).astype(float)
    days_to = np.where(days_to < 0, 999, days_to)
    days_since = np.where(days_since < 0, 999, days_since)
    return pd.DataFrame(
        {
            "recipient_holiday_today": is_holiday,
            "days_to_recipient_holiday": np.minimum(days_to, 31),
            "days_since_recipient_holiday": np.minimum(days_since, 31),
            "recipient_holiday_next_7d": ((days_to >= 0) & (days_to <= 7)).astype(float),
            "recipient_holiday_previous_3d": ((days_since >= 0) & (days_since <= 3)).astype(float),
        },
        index=dates,
    )


def build_features(prices: pd.DataFrame, brent_path: Path | None = None) -> pd.DataFrame:
    """Build causal continuous features for every published target-currency date."""
    pivot = prices.pivot(index="date", columns="currency", values="rub_per_unit").sort_index()
    log_price, returns = np.log(pivot), np.log(pivot).diff()
    context = [name for name in ("USD", "EUR", "CNY") if name in pivot]
    parts: list[pd.DataFrame] = []

    for currency in TARGET_CURRENCIES:
        price, ret = pivot[currency], returns[currency]
        frame = pd.DataFrame(index=pivot.index)

        for lag in (0, 1, 2, 3, 5, 10, 20):
            frame[f"return_lag_{lag}"] = ret.shift(lag)

        for window in (3, 5, 10, 20, 60, 120):
            rolling_price, rolling_return = price.rolling(window), ret.rolling(window)
            frame[f"return_mean_{window}"] = rolling_return.mean()
            frame[f"return_std_{window}"] = rolling_return.std()
            frame[f"price_to_mean_{window}"] = price / rolling_price.mean() - 1
            frame[f"distance_to_min_{window}_bps"] = (price / rolling_price.min() - 1) * 10_000
            frame[f"drawdown_{window}_bps"] = (price / rolling_price.max() - 1) * 10_000

        for window in (3, 5, 10):
            frame[f"slope_{window}"] = rolling_slope(log_price[currency], window)

        frame["slope_change_3_10"] = frame["slope_3"] - frame["slope_10"]
        frame["acceleration_3"] = ret.rolling(3).mean() - ret.shift(3).rolling(3).mean()
        frame["streak_length"] = decline_streak(ret)
        frame["last_bounce_bps"] = ret.clip(lower=0) * 10_000
        frame["reversal_after_fall"] = (ret.gt(0) & ret.shift(1).lt(0)).astype(float)
        frame["vol_ratio_5_60"] = frame["return_std_5"] / frame["return_std_60"].replace(0, np.nan)
        frame["vol_ratio_20_120"] = frame["return_std_20"] / frame["return_std_120"].replace(0, np.nan)
        frame["time_since_min_20"] = periods_since_rolling_min(price, 20)
        frame["time_since_min_60"] = periods_since_rolling_min(price, 60)

        # Compare T with exactly the PREVIOUS 'window' observations.
        # rolling(window + 1) contains those previous observations plus T.
        for window in (20, 60, 120):
            frame[f"level_percentile_{window}"] = price.rolling(window + 1).apply(
                lambda values: float(np.mean(values[:-1] >= values[-1])),
                raw=True,
            )

        for other in context:
            for lag in (0, 1, 2, 5):
                frame[f"{other.lower()}_return_lag_{lag}"] = returns[other].shift(lag)

        if context:
            frame["context_return_mean_0"] = returns[context].mean(axis=1)
            frame["own_minus_context_0"] = ret - frame["context_return_mean_0"]

        if brent_path is not None:
            frame = frame.join(load_brent_features(brent_path, pivot.index))

        frame = frame.join(recipient_holiday_features(pivot.index, COUNTRY_BY_CURRENCY[currency]))
        # CBR records include effective dates on Saturdays and occasionally
        # Sundays. A seven-day cycle avoids aliasing Saturday with Monday.
        frame["dow_sin"] = np.sin(2 * np.pi * frame.index.dayofweek / 7)
        frame["dow_cos"] = np.cos(2 * np.pi * frame.index.dayofweek / 7)
        frame["month_sin"] = np.sin(2 * np.pi * frame.index.month / 12)
        frame["month_cos"] = np.cos(2 * np.pi * frame.index.month / 12)
        frame["currency"], frame["date"] = currency, frame.index

        # Technical field only: explicitly excluded from X.
        frame["observation_index"] = np.arange(len(frame))
        parts.append(frame.reset_index(drop=True))

    features = pd.concat(parts, ignore_index=True)
    return pd.concat(
        [features, pd.get_dummies(features["currency"], prefix="currency", dtype=float)],
        axis=1,
    )


def build_labels(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for currency, group in prices.loc[
        prices["currency"].isin(TARGET_CURRENCIES)
    ].groupby("currency", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        outcome = outcome_frame(
            group["rub_per_unit"], pd.DatetimeIndex(group["date"]), horizon
        ).reset_index()
        outcome["currency"] = currency
        outcome = outcome.rename(columns={"hit": "truth_now", "local_min_hit": "local_min"})
        rows.append(
            outcome[
                [
                    "date",
                    "currency",
                    "truth_now",
                    "local_min",
                    "benefit_bps",
                    "future_regret_bps",
                    "label_available_date",
                    "valid",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def feature_columns(data: pd.DataFrame) -> list[str]:
    excluded = {
        "date",
        "currency",
        "observation_index",
        "truth_now",
        "local_min",
        "target_hit",
        "benefit_bps",
        "future_regret_bps",
        "good_push",
        "label_available_date",
        "valid",
    }
    return [
        column
        for column in data.columns
        if column not in excluded and data[column].nunique(dropna=True) > 1
    ]


def model_train_start(config: str, cutoff: pd.Timestamp) -> pd.Timestamp:
    if config == "long_history":
        return pd.Timestamp("2014-01-01")
    if config.startswith("rolling_") and config.endswith("y"):
        years = int(config.removeprefix("rolling_").removesuffix("y"))
        return cutoff - pd.DateOffset(years=years)
    raise ValueError(f"Unknown model config: {config}")


def make_models() -> tuple[
    HistGradientBoostingClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
]:
    common = dict(
        learning_rate=0.10,
        max_iter=35,
        max_leaf_nodes=9,
        min_samples_leaf=25,
        l2_regularization=3,
        random_state=42,
    )
    return (
        HistGradientBoostingClassifier(class_weight="balanced", **common),
        HistGradientBoostingClassifier(
            class_weight="balanced", **{**common, "random_state": 43}
        ),
        HistGradientBoostingRegressor(
            loss="absolute_error", **{**common, "random_state": 44}
        ),
    )


def make_truth_model() -> HistGradientBoostingClassifier:
    """Truth-only model used by the controlled candidate-population experiment."""
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        learning_rate=0.10,
        max_iter=35,
        max_leaf_nodes=9,
        min_samples_leaf=25,
        l2_regularization=3,
        random_state=42,
    )


def candidate_gate_mask(data: pd.DataFrame, gate: str) -> pd.Series:
    """Causal stage-one eligibility based only on the trailing price level."""
    if gate == "all_days":
        return pd.Series(True, index=data.index)
    thresholds = {
        "level60_q70": 0.70,
        "level60_q80": 0.80,
        "level60_q90": 0.90,
    }
    if gate not in thresholds:
        raise ValueError(f"Unknown candidate gate: {gate}")
    return data["level_percentile_60"].ge(thresholds[gate]).fillna(False)


def fit_truth_model(
    train: pd.DataFrame, columns: list[str]
) -> HistGradientBoostingClassifier:
    model = make_truth_model()
    model.fit(train[columns], train["truth_now"])
    return model


def score_truth_model(
    model: HistGradientBoostingClassifier,
    score: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    result = score.copy()
    result["p_truth_now"] = model.predict_proba(score[columns])[:, 1]
    return result


def fit_local_model(
    train: pd.DataFrame, columns: list[str]
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        class_weight="balanced",
        learning_rate=0.10,
        max_iter=35,
        max_leaf_nodes=9,
        min_samples_leaf=25,
        l2_regularization=3,
        random_state=43,
    )
    model.fit(train[columns], train["local_min"])
    return model


def score_local_model(
    model: HistGradientBoostingClassifier,
    score: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    result = score.copy()
    result["p_local_min"] = model.predict_proba(score[columns])[:, 1]
    return result


def fit_regret_model(
    train: pd.DataFrame, columns: list[str]
) -> HistGradientBoostingRegressor:
    """Robust regression of the magnitude of waiting regret in basis points."""
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.10,
        max_iter=35,
        max_leaf_nodes=9,
        min_samples_leaf=25,
        l2_regularization=3,
        random_state=45,
    )
    model.fit(train[columns], train["future_regret_bps"])
    return model


def score_regret_model(
    model: HistGradientBoostingRegressor,
    score: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    result = score.copy()
    # Regret is non-negative by definition. Clipping affects interpretation,
    # while preserving the order among all non-negative predictions.
    result["predicted_future_regret_bps"] = np.maximum(
        model.predict(score[columns]), 0.0
    )
    return result


def make_benefit_quantile_model() -> HistGradientBoostingRegressor:
    """Conservative conditional 25th-percentile model for case benefit."""
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.25,
        learning_rate=0.10,
        max_iter=35,
        max_leaf_nodes=9,
        min_samples_leaf=25,
        l2_regularization=3,
        random_state=46,
    )


def make_good_push_model() -> HistGradientBoostingClassifier:
    """Classifier of the joint truth_now AND positive-benefit training label."""
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        learning_rate=0.10,
        max_iter=35,
        max_leaf_nodes=9,
        min_samples_leaf=25,
        l2_regularization=3,
        random_state=47,
    )


def fit_joint_architecture(
    train: pd.DataFrame,
    columns: list[str],
    architecture: str,
) -> tuple[object, ...]:
    """Fit one of the two controlled joint-selection architectures."""
    if architecture == "two_head":
        truth_model = fit_truth_model(train, columns)
        benefit_model = make_benefit_quantile_model()
        benefit_model.fit(train[columns], train["benefit_bps"])
        return truth_model, benefit_model
    if architecture == "joint_classifier":
        model = make_good_push_model()
        model.fit(train[columns], train["good_push"])
        return (model,)
    raise ValueError(f"Unknown joint architecture: {architecture}")


def score_joint_architecture(
    models: tuple[object, ...],
    score: pd.DataFrame,
    columns: list[str],
    architecture: str,
) -> pd.DataFrame:
    result = score.copy()
    if architecture == "two_head":
        truth_model, benefit_model = models
        result["p_truth_now"] = truth_model.predict_proba(score[columns])[:, 1]
        result["predicted_benefit_q25"] = benefit_model.predict(score[columns])
        return result
    if architecture == "joint_classifier":
        (model,) = models
        result["p_good_push"] = model.predict_proba(score[columns])[:, 1]
        return result
    raise ValueError(f"Unknown joint architecture: {architecture}")


def causal_score_percentiles(
    reference_scored: pd.DataFrame,
    target_scored: pd.DataFrame,
    score_columns: tuple[str, ...],
    lookback: int = SCORE_PERCENTILE_LOOKBACK,
) -> pd.DataFrame:
    """Convert model scores to causal trailing percentiles per currency.

    Every target row is compared only with earlier scores from the same
    currency. Target scores are appended to the reference after their date,
    so the normalization adapts to score drift without observing outcomes.
    """
    result = target_scored.copy()
    for column in score_columns:
        result[f"{column}_rank"] = np.nan

    for currency in TARGET_CURRENCIES:
        reference = reference_scored.loc[
            reference_scored["currency"].eq(currency)
        ].sort_values("date")
        target = result.loc[result["currency"].eq(currency)].sort_values("date")
        histories = {
            column: reference[column].dropna().astype(float).tolist()[-lookback:]
            for column in score_columns
        }
        for index, row in target.iterrows():
            for column in score_columns:
                value = float(row[column])
                history = histories[column]
                if history:
                    ordered = np.sort(np.asarray(history, dtype=float))
                    rank = np.searchsorted(ordered, value, side="right") / len(ordered)
                    result.at[index, f"{column}_rank"] = float(rank)
                history.append(value)
                if len(history) > lookback:
                    del history[0]
    return result


def fit_models(
    train: pd.DataFrame,
    columns: list[str],
) -> tuple[
    HistGradientBoostingClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
]:
    truth_model, local_model, benefit_model = make_models()
    truth_model.fit(train[columns], train["truth_now"])
    local_model.fit(train[columns], train["local_min"])
    benefit_model.fit(train[columns], train["benefit_bps"])
    return truth_model, local_model, benefit_model


def score_models(
    models: tuple[
        HistGradientBoostingClassifier,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    ],
    score: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    truth_model, local_model, benefit_model = models
    result = score.copy()
    result["p_truth_now"] = truth_model.predict_proba(score[columns])[:, 1]
    result["p_local_min"] = local_model.predict_proba(score[columns])[:, 1]
    result["predicted_benefit_bps"] = benefit_model.predict(score[columns])
    return result


def fit_and_score(train: pd.DataFrame, score: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compatibility helper for tests and small diagnostics."""
    return score_models(fit_models(train, columns), score, columns)


def apply_observation_cooldown(frame: pd.DataFrame, cooldown: int) -> pd.DataFrame:
    """Keep the first eligible signal, then suppress the next `cooldown` observations."""
    kept: list[pd.DataFrame] = []
    for _, group in frame.sort_values(["currency", "date"]).groupby("currency", sort=True):
        last, positions = -10**9, []
        for position, observation_index in enumerate(group["observation_index"].astype(int)):
            if observation_index - last > cooldown:
                positions.append(position)
                last = observation_index
        kept.append(group.iloc[positions])
    return pd.concat(kept, ignore_index=True) if kept else frame.iloc[0:0].copy()


def recency_weights(dates: pd.Series, cutoff: pd.Timestamp, half_life_days: int | None) -> np.ndarray:
    """Retained for backwards-compatible diagnostics; current nested grid uses windows."""
    if half_life_days is None:
        return np.ones(len(dates), dtype=float)
    age = (cutoff - dates).dt.days.clip(lower=0).to_numpy(float)
    return np.exp(-np.log(2.0) * age / half_life_days)


def weeks_in_period(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Elapsed calendar weeks in the half-open evaluation interval."""
    return max(
        1.0,
        (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (7 * 86400),
    )


def mature_training_rows(
    data: pd.DataFrame, train_start: pd.Timestamp, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Rows whose outcomes were already known strictly before `cutoff`."""
    return data.loc[
        data["date"].ge(train_start)
        & data["date"].lt(cutoff)
        & data["label_available_date"].lt(cutoff)
        & data["valid"]
    ]


def mature_validation_rows(
    data: pd.DataFrame,
    validation_start: pd.Timestamp,
    decision_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Inner-validation rows whose labels were actually available at decision time."""
    return data.loc[
        data["date"].ge(validation_start)
        & data["date"].lt(decision_cutoff)
        & data["label_available_date"].lt(decision_cutoff)
        & data["valid"]
    ]


def raw_metrics(selected: pd.DataFrame, pool: pd.DataFrame, weeks: int) -> dict[str, float]:
    if selected.empty:
        return {
            "count": 0,
            "frequency": 0.0,
            "truth_lift": np.nan,
            "local_lift": np.nan,
            "benefit": np.nan,
        }
    truth_base, local_base = pool["truth_now"].mean(), pool["local_min"].mean()
    return {
        "count": len(selected),
        "frequency": len(selected) / weeks,
        "truth_lift": selected["truth_now"].mean() / truth_base if truth_base else np.nan,
        "local_lift": selected["local_min"].mean() / local_base if local_base else np.nan,
        "benefit": selected["benefit_bps"].mean(),
    }


def choose_currency_thresholds(
    validation: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Choose per-currency thresholds on inner historical validation only."""
    weeks = weeks_in_period(start, end)
    choices: dict[str, dict[str, float]] = {}
    diagnostics: list[pd.Series] = []

    for currency in TARGET_CURRENCIES:
        data = validation.loc[validation["currency"].eq(currency)].sort_values("date")
        if data.empty:
            continue

        def quantile_candidates(column: str, quantiles: tuple[float, ...]) -> list[tuple[float, float]]:
            candidates: list[tuple[float, float]] = []
            seen: set[float] = set()
            for quantile in quantiles:
                value = float(data[column].quantile(quantile))
                if value not in seen:
                    candidates.append((float(quantile), value))
                    seen.add(value)
            return candidates

        truth_thresholds = quantile_candidates("p_truth_now", PROBABILITY_QUANTILES)
        local_thresholds = quantile_candidates("p_local_min", LOCAL_QUANTILES)
        benefit_thresholds = quantile_candidates(
            "predicted_benefit_bps", BENEFIT_QUANTILES
        )

        candidates: list[dict[str, float | int | str]] = []
        for truth_quantile, truth_threshold in truth_thresholds:
            for local_quantile, local_threshold in local_thresholds:
                for benefit_quantile, benefit_threshold in benefit_thresholds:
                    raw_selected = data.loc[
                        data["p_truth_now"].ge(truth_threshold)
                        & data["p_local_min"].ge(local_threshold)
                        & data["predicted_benefit_bps"].ge(benefit_threshold)
                    ]
                    selected = apply_observation_cooldown(
                        raw_selected, SELECTION_COOLDOWN
                    )
                    metrics = raw_metrics(selected, data, weeks)
                    candidates.append(
                        {
                            "currency": currency,
                            "truth_quantile": truth_quantile,
                            "truth_threshold": float(truth_threshold),
                            "local_quantile": local_quantile,
                            "local_threshold": float(local_threshold),
                            "benefit_quantile": benefit_quantile,
                            "benefit_threshold": float(benefit_threshold),
                            **metrics,
                        }
                    )

        table = pd.DataFrame(candidates)
        table = table.assign(
            enough=table["count"].ge(MIN_VALIDATION_SIGNALS),
            benefit_ok=table["benefit"].ge(0),
            truth_ok=table["truth_lift"].ge(TRUTH_LIFT_TARGET),
            local_ok=table["local_lift"].ge(LOCAL_LIFT_TARGET),
            lift_floor=np.minimum(
                table["truth_lift"] / TRUTH_LIFT_TARGET,
                table["local_lift"] / LOCAL_LIFT_TARGET,
            ),
        )
        table["joint_ok"] = (
            table["enough"]
            & table["benefit_ok"]
            & table["truth_ok"]
            & table["local_ok"]
        )

        # Prefer a policy satisfying all constraints. If none exists, return the
        # best honest compromise rather than silently relaxing a condition.
        table = table.sort_values(
            [
                "joint_ok",
                "enough",
                "truth_ok",
                "local_ok",
                "benefit_ok",
                "lift_floor",
                "benefit",
                "count",
            ],
            ascending=[False, False, False, False, False, False, False, False],
        )

        best = table.iloc[0]
        choices[currency] = {
            "truth_quantile": float(best["truth_quantile"]),
            "truth_threshold": float(best["truth_threshold"]),
            "local_quantile": float(best["local_quantile"]),
            "local_threshold": float(best["local_threshold"]),
            "benefit_quantile": float(best["benefit_quantile"]),
            "benefit_threshold": float(best["benefit_threshold"]),
        }
        diagnostics.append(best)

    return choices, pd.DataFrame(diagnostics)


def recalibrate_thresholds(
    policy: dict[str, dict[str, float]],
    reference_scored: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Map selected quantile policies to the score scale of a refitted model.

    The reference period and every chosen quantile are known before the outer
    fold. Outcomes are not used during this score-scale mapping.
    """
    result: dict[str, dict[str, float]] = {}
    for currency, choice in policy.items():
        reference = reference_scored.loc[reference_scored["currency"].eq(currency)]
        if reference.empty:
            continue
        result[currency] = {
            "truth_threshold": float(
                reference["p_truth_now"].quantile(choice["truth_quantile"])
            ),
            "local_threshold": float(
                reference["p_local_min"].quantile(choice["local_quantile"])
            ),
            "benefit_threshold": float(
                reference["predicted_benefit_bps"].quantile(
                    choice["benefit_quantile"]
                )
            ),
        }
    return result


def choose_truth_only_thresholds(
    validation_candidates: pd.DataFrame,
    validation_pool: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Select p_truth quantiles while keeping the official all-days baseline."""
    weeks = weeks_in_period(start, end)
    choices: dict[str, dict[str, float]] = {}
    diagnostics: list[pd.Series] = []

    for currency in TARGET_CURRENCIES:
        candidates = validation_candidates.loc[
            validation_candidates["currency"].eq(currency)
        ].sort_values("date")
        pool = validation_pool.loc[validation_pool["currency"].eq(currency)]
        if candidates.empty or pool.empty:
            continue

        rows: list[dict[str, float | int | str]] = []
        seen: set[float] = set()
        for quantile in PROBABILITY_QUANTILES:
            threshold = float(candidates["p_truth_now"].quantile(quantile))
            if threshold in seen:
                continue
            seen.add(threshold)
            selected = candidates.loc[candidates["p_truth_now"].ge(threshold)]
            metrics = raw_metrics(selected, pool, weeks)
            rows.append(
                {
                    "currency": currency,
                    "truth_quantile": float(quantile),
                    "truth_threshold": threshold,
                    **metrics,
                }
            )

        table = pd.DataFrame(rows).assign(
            enough=lambda frame: frame["count"].ge(MIN_VALIDATION_SIGNALS),
            truth_ok=lambda frame: frame["truth_lift"].ge(TRUTH_LIFT_TARGET),
            benefit_ok=lambda frame: frame["benefit"].ge(0),
        )
        table["joint_ok"] = table["enough"] & table["truth_ok"] & table["benefit_ok"]
        table = table.sort_values(
            ["joint_ok", "enough", "truth_ok", "benefit_ok", "truth_lift", "benefit", "count"],
            ascending=[False, False, False, False, False, False, False],
        )
        best = table.iloc[0]
        choices[currency] = {
            "truth_quantile": float(best["truth_quantile"]),
            "truth_threshold": float(best["truth_threshold"]),
        }
        diagnostics.append(best)

    return choices, pd.DataFrame(diagnostics)


def choose_local_only_thresholds(
    validation_candidates: pd.DataFrame,
    validation_pool: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Select p_local quantiles independently from the truth_now stream."""
    weeks = weeks_in_period(start, end)
    choices: dict[str, dict[str, float]] = {}
    diagnostics: list[pd.Series] = []
    for currency in TARGET_CURRENCIES:
        candidates = validation_candidates.loc[
            validation_candidates["currency"].eq(currency)
        ].sort_values("date")
        pool = validation_pool.loc[validation_pool["currency"].eq(currency)]
        if candidates.empty or pool.empty:
            continue
        rows: list[dict[str, float | int | str]] = []
        seen: set[float] = set()
        for quantile in LOCAL_QUANTILES:
            threshold = float(candidates["p_local_min"].quantile(quantile))
            if threshold in seen:
                continue
            seen.add(threshold)
            selected = candidates.loc[candidates["p_local_min"].ge(threshold)]
            metrics = raw_metrics(selected, pool, weeks)
            rows.append(
                {
                    "currency": currency,
                    "local_quantile": float(quantile),
                    "local_threshold": threshold,
                    **metrics,
                }
            )
        table = pd.DataFrame(rows).assign(
            enough=lambda frame: frame["count"].ge(MIN_VALIDATION_SIGNALS),
            local_ok=lambda frame: frame["local_lift"].ge(LOCAL_LIFT_TARGET),
            benefit_ok=lambda frame: frame["benefit"].ge(0),
        )
        table["joint_ok"] = table["enough"] & table["local_ok"] & table["benefit_ok"]
        table = table.sort_values(
            ["joint_ok", "enough", "local_ok", "benefit_ok", "local_lift", "benefit", "count"],
            ascending=[False, False, False, False, False, False, False],
        )
        best = table.iloc[0]
        choices[currency] = {
            "local_quantile": float(best["local_quantile"]),
            "local_threshold": float(best["local_threshold"]),
        }
        diagnostics.append(best)
    return choices, pd.DataFrame(diagnostics)


def choose_regret_thresholds(
    validation_scored: pd.DataFrame,
    validation_pool: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Select a low predicted-regret quantile using exact truth and case benefit."""
    weeks = weeks_in_period(start, end)
    choices: dict[str, dict[str, float]] = {}
    diagnostics: list[pd.Series] = []
    for currency in TARGET_CURRENCIES:
        scored = validation_scored.loc[
            validation_scored["currency"].eq(currency)
        ].sort_values("date")
        pool = validation_pool.loc[validation_pool["currency"].eq(currency)]
        if scored.empty or pool.empty:
            continue
        rows: list[dict[str, float | int | str]] = []
        seen: set[float] = set()
        for quantile in REGRET_QUANTILES:
            threshold = float(
                scored["predicted_future_regret_bps"].quantile(quantile)
            )
            if threshold in seen:
                continue
            seen.add(threshold)
            selected = scored.loc[
                scored["predicted_future_regret_bps"].le(threshold)
            ]
            metrics = raw_metrics(selected, pool, weeks)
            rows.append(
                {
                    "currency": currency,
                    "regret_quantile": float(quantile),
                    "regret_threshold_bps": threshold,
                    **metrics,
                }
            )
        table = pd.DataFrame(rows).assign(
            enough=lambda frame: frame["count"].ge(MIN_VALIDATION_SIGNALS),
            truth_ok=lambda frame: frame["truth_lift"].ge(TRUTH_LIFT_TARGET),
            benefit_ok=lambda frame: frame["benefit"].ge(0),
        )
        table["joint_ok"] = table["enough"] & table["truth_ok"] & table["benefit_ok"]
        table = table.sort_values(
            ["joint_ok", "enough", "truth_ok", "benefit_ok", "truth_lift", "benefit", "count"],
            ascending=[False, False, False, False, False, False, False],
        )
        best = table.iloc[0]
        choices[currency] = {
            "regret_quantile": float(best["regret_quantile"]),
            "regret_threshold_bps": float(best["regret_threshold_bps"]),
        }
        diagnostics.append(best)
    return choices, pd.DataFrame(diagnostics)


def remap_truth_thresholds(
    quantile_policy: dict[str, dict[str, float]],
    reference_scored: pd.DataFrame,
) -> dict[str, float]:
    """Map truth-score quantiles onto a refitted classifier's score scale."""
    thresholds: dict[str, float] = {}
    for currency, policy in quantile_policy.items():
        reference = reference_scored.loc[
            reference_scored["currency"].eq(currency), "p_truth_now"
        ]
        if not reference.empty:
            thresholds[currency] = float(reference.quantile(policy["truth_quantile"]))
    return thresholds


def remap_local_thresholds(
    quantile_policy: dict[str, dict[str, float]],
    reference_scored: pd.DataFrame,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for currency, policy in quantile_policy.items():
        reference = reference_scored.loc[
            reference_scored["currency"].eq(currency), "p_local_min"
        ]
        if not reference.empty:
            thresholds[currency] = float(reference.quantile(policy["local_quantile"]))
    return thresholds


def remap_regret_thresholds(
    quantile_policy: dict[str, dict[str, float]],
    reference_scored: pd.DataFrame,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for currency, policy in quantile_policy.items():
        reference = reference_scored.loc[
            reference_scored["currency"].eq(currency),
            "predicted_future_regret_bps",
        ]
        if not reference.empty:
            thresholds[currency] = float(
                reference.quantile(policy["regret_quantile"])
            )
    return thresholds


def apply_truth_thresholds(
    scored_candidates: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    pieces = [
        scored_candidates.loc[
            scored_candidates["currency"].eq(currency)
            & scored_candidates["p_truth_now"].ge(threshold)
        ]
        for currency, threshold in thresholds.items()
    ]
    return (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else scored_candidates.iloc[0:0].copy()
    )


def apply_local_thresholds(
    scored_candidates: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    pieces = [
        scored_candidates.loc[
            scored_candidates["currency"].eq(currency)
            & scored_candidates["p_local_min"].ge(threshold)
        ]
        for currency, threshold in thresholds.items()
    ]
    return (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else scored_candidates.iloc[0:0].copy()
    )


def apply_regret_thresholds(
    scored: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    pieces = [
        scored.loc[
            scored["currency"].eq(currency)
            & scored["predicted_future_regret_bps"].le(threshold)
        ]
        for currency, threshold in thresholds.items()
    ]
    return pd.concat(pieces, ignore_index=True) if pieces else scored.iloc[0:0].copy()


def union_signal_streams(*streams: pd.DataFrame) -> pd.DataFrame:
    """OR-combine streams once per currency/date without applying cooldown."""
    nonempty = [stream for stream in streams if not stream.empty]
    if not nonempty:
        return streams[0].iloc[0:0].copy() if streams else pd.DataFrame()
    return (
        pd.concat(nonempty, ignore_index=True)
        .sort_values(["currency", "date"])
        .drop_duplicates(["currency", "date"], keep="first")
        .reset_index(drop=True)
    )


def attach_target_stream_frequencies(
    metrics: pd.DataFrame,
    truth_stream: pd.DataFrame,
    local_stream: pd.DataFrame,
    joint_stream: pd.DataFrame,
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
) -> pd.DataFrame:
    """Report target-specific frequencies without using them for selection."""
    result = metrics.copy()
    weeks = weeks_in_period(fold_start, fold_end)
    for currency in TARGET_CURRENCIES:
        mask = result["currency"].eq(currency)
        truth_count = int(truth_stream["currency"].eq(currency).sum())
        local_count = int(local_stream["currency"].eq(currency).sum())
        joint_count = int(joint_stream["currency"].eq(currency).sum())
        result.loc[mask, "truth_stream_count"] = truth_count
        result.loc[mask, "truth_stream_per_week"] = truth_count / weeks
        result.loc[mask, "local_stream_count"] = local_count
        result.loc[mask, "local_stream_per_week"] = local_count / weeks
        result.loc[mask, "target_intersection_count"] = joint_count
        result.loc[mask, "target_intersection_per_week"] = joint_count / weeks
    return result


def apply_thresholds(
    scored: pd.DataFrame,
    thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for currency, threshold in thresholds.items():
        data = scored.loc[scored["currency"].eq(currency)]
        selected = data.loc[
            data["p_truth_now"].ge(threshold["truth_threshold"])
            & data["p_local_min"].ge(threshold["local_threshold"])
            & data["predicted_benefit_bps"].ge(threshold["benefit_threshold"])
        ]
        pieces.append(apply_observation_cooldown(selected, SELECTION_COOLDOWN))
    return pd.concat(pieces, ignore_index=True) if pieces else scored.iloc[0:0].copy()


def independent_target_streams(
    scored: pd.DataFrame,
    thresholds: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Truth-only, local-only and their intersection; benefit is not applied."""
    truth_parts: list[pd.DataFrame] = []
    local_parts: list[pd.DataFrame] = []
    joint_parts: list[pd.DataFrame] = []
    for currency, threshold in thresholds.items():
        data = scored.loc[scored["currency"].eq(currency)]
        truth_mask = data["p_truth_now"].ge(threshold["truth_threshold"])
        local_mask = data["p_local_min"].ge(threshold["local_threshold"])
        truth_parts.append(data.loc[truth_mask])
        local_parts.append(data.loc[local_mask])
        joint_parts.append(data.loc[truth_mask & local_mask])
    empty = scored.iloc[0:0].copy()
    return (
        pd.concat(truth_parts, ignore_index=True) if truth_parts else empty.copy(),
        pd.concat(local_parts, ignore_index=True) if local_parts else empty.copy(),
        pd.concat(joint_parts, ignore_index=True) if joint_parts else empty.copy(),
    )


def choose_model(
    data: pd.DataFrame,
    columns: list[str],
    outer_start: pd.Timestamp,
) -> tuple[
    str,
    dict[str, dict[str, float]],
    pd.DataFrame,
]:
    """Nested model/threshold selection using only information available before outer_start."""
    rows: list[pd.DataFrame] = []
    thresholds_by_config: dict[str, dict[str, dict[str, float]]] = {}

    for config in MODEL_CONFIGS:
        result = select_policy_for_config(data, columns, outer_start, config)
        if result is None:
            continue
        thresholds, diagnostics = result
        diagnostics["model_config"] = config
        rows.append(diagnostics)
        thresholds_by_config[config] = thresholds

    if not rows:
        raise RuntimeError(f"No valid inner configuration for outer_start={outer_start.date()}")

    table = pd.concat(rows, ignore_index=True)
    summary = table.groupby("model_config", as_index=False).agg(
        currency_coverage=("currency", "nunique"),
        joint_passing=("joint_ok", "sum"),
        truth_passing=("truth_ok", "sum"),
        local_passing=("local_ok", "sum"),
        positive_benefit=("benefit_ok", "sum"),
        min_lift_floor=("lift_floor", "min"),
        min_truth_lift=("truth_lift", "min"),
        min_local_lift=("local_lift", "min"),
        min_benefit=("benefit", "min"),
    )
    summary = summary.sort_values(
        [
            "currency_coverage",
            "joint_passing",
            "truth_passing",
            "local_passing",
            "positive_benefit",
            "min_lift_floor",
            "min_benefit",
        ],
        ascending=[False, False, False, False, False, False, False],
    )

    chosen = str(summary.iloc[0]["model_config"])
    table["chosen_model"] = table["model_config"].eq(chosen)
    return chosen, thresholds_by_config[chosen], table


def select_policy_for_config(
    data: pd.DataFrame,
    columns: list[str],
    outer_start: pd.Timestamp,
    config: str,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame] | None:
    """Select threshold quantiles for one fixed history-window policy."""
    validation_start = outer_start - pd.DateOffset(years=1)
    train = mature_training_rows(
        data, model_train_start(config, validation_start), validation_start
    )
    validation = mature_validation_rows(data, validation_start, outer_start)
    if len(train) < 500 or validation.empty:
        return None
    if train["truth_now"].nunique() < 2 or train["local_min"].nunique() < 2:
        return None

    models = fit_models(train, columns)
    scored = score_models(models, validation, columns)
    policy, diagnostics = choose_currency_thresholds(
        scored, validation_start, outer_start
    )
    if diagnostics.empty:
        return None
    return policy, diagnostics


def refit_selected_policy(
    data: pd.DataFrame,
    columns: list[str],
    outer_start: pd.Timestamp,
    model_config: str,
    quantile_policy: dict[str, dict[str, float]],
) -> tuple[
    tuple[
        HistGradientBoostingClassifier,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    ],
    dict[str, dict[str, float]],
    pd.DataFrame,
]:
    """Refit through outer_start and remap quantile policy to its score scale."""
    train = mature_training_rows(
        data, model_train_start(model_config, outer_start), outer_start
    )
    if len(train) < 500:
        raise RuntimeError(
            f"Too few refit rows for {model_config} at {outer_start.date()}: {len(train)}"
        )
    models = fit_models(train, columns)

    reference = mature_validation_rows(
        data, outer_start - pd.DateOffset(years=1), outer_start
    )
    reference_scored = score_models(models, reference, columns)
    thresholds = recalibrate_thresholds(quantile_policy, reference_scored)
    return models, thresholds, reference_scored


def message_fact(row: pd.Series) -> str:
    """A true, client-readable fact. This is NOT claimed to be a model explanation."""
    if row["streak_length"] >= 2:
        return f"курс снижается {int(row['streak_length'])} публикации подряд"
    if row["reversal_after_fall"] > 0:
        return "после снижения появился первый дневной отскок"
    if row["distance_to_min_20_bps"] <= 25:
        return "курс находится не далее 25 bp от минимума последних 20 публикаций"
    if row["level_percentile_60"] >= 0.80:
        return "курс выгоднее как минимум 80% предыдущих 60 публикаций"
    if row["slope_change_3_10"] > 0:
        return "короткий наклон курса улучшился относительно 10-дневного"
    return "сочетание низкого уровня, динамики и волатильности"


def add_message_facts(selected: pd.DataFrame) -> pd.DataFrame:
    result = selected.copy()
    if result.empty:
        result["message_fact"] = pd.Series(index=result.index, dtype="object")
    else:
        result["message_fact"] = result.apply(message_fact, axis=1)
    return result


def truth_threshold_curve(
    scored: pd.DataFrame,
    reference_scored: pd.DataFrame,
    horizon: int,
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
) -> pd.DataFrame:
    """Apply inner-validation score quantiles as fixed thresholds to outer OOT data."""
    rows: list[dict[str, object]] = []
    weeks = weeks_in_period(fold_start, fold_end)
    quantiles = (0.0, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.975)
    for currency in TARGET_CURRENCIES:
        data = scored.loc[scored["currency"].eq(currency)].sort_values("date")
        reference = reference_scored.loc[
            reference_scored["currency"].eq(currency), "p_truth_now"
        ]
        if data.empty or reference.empty:
            continue
        for quantile in quantiles:
            # This value is known at fold_start; the outer score distribution is
            # never used to decide what "high score" means.
            threshold = float(reference.quantile(quantile))
            selected = data.loc[data["p_truth_now"].ge(threshold)]
            metric = raw_metrics(selected, data, weeks)
            rows.append(
                {
                    "fold_start": fold_start,
                    "fold_end": fold_end,
                    "currency": currency,
                    "horizon": horizon,
                    "score_quantile": quantile,
                    "threshold_source": "inner_validation",
                    "p_truth_threshold": threshold,
                    "mean_predicted_truth": selected["p_truth_now"].mean(),
                    "truth_hit_rate": selected["truth_now"].mean(),
                    **metric,
                }
            )
    return pd.DataFrame(rows)


def evaluate_fold(
    selected: pd.DataFrame,
    pool: pd.DataFrame,
    horizon: int,
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
) -> pd.DataFrame:
    weeks = weeks_in_period(fold_start, fold_end)
    rows: list[dict[str, object]] = []

    for currency in TARGET_CURRENCIES:
        currency_pool = pool.loc[pool["currency"].eq(currency) & pool["valid"]]
        signals = selected.loc[selected["currency"].eq(currency)]
        count = len(signals)

        if count:
            # The expectation of a uniform sample without replacement is the
            # full-period mean. Use it directly so lift has no Monte Carlo noise.
            truth_random = float(currency_pool["truth_now"].mean())
            local_random = float(currency_pool["local_min"].mean())
            benefit = monthly_block_bootstrap_mean(
                signals.set_index("date")["benefit_bps"],
                RANDOM_REPEATS,
                deterministic_seed(43, fold_start, currency, horizon),
            )
            clustered = (
                signals.sort_values("observation_index")["observation_index"]
                .diff()
                .le(5)
                .mean()
            )
            # The case asks whether mean benefit is significantly greater than
            # zero, so the decision rule is the one-sided test H1: mean > 0.
            # The two-sided 95% interval remains a separate diagnostic and is
            # intentionally not combined with the p-value.
            benefit_significant = bool(benefit["p_value"] < 0.05)
        else:
            truth_random = local_random = np.nan
            benefit = {"ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
            clustered = np.nan
            benefit_significant = False

        truth_rate = signals["truth_now"].mean() if count else np.nan
        local_rate = signals["local_min"].mean() if count else np.nan
        rows.append(
            {
                "fold_start": fold_start,
                "fold_end": fold_end,
                "currency": currency,
                "horizon": horizon,
                "signal_count": count,
                "signals_per_week": count / weeks,
                "truth_now_hit_rate": truth_rate,
                "truth_now_random": truth_random,
                "truth_now_lift": truth_rate / truth_random if count and truth_random else np.nan,
                "local_min_hit_rate": local_rate,
                "local_min_random": local_random,
                "local_min_lift": local_rate / local_random if count and local_random else np.nan,
                "benefit_bps": signals["benefit_bps"].mean() if count else np.nan,
                "benefit_ci_low_95": benefit["ci_low"],
                "benefit_ci_high_95": benefit["ci_high"],
                "benefit_p_value_vs_zero": benefit["p_value"],
                "benefit_significant_5pct": benefit_significant,
                "clustered_share_within_5_observations": clustered,
            }
        )

    return pd.DataFrame(rows)


def outer_folds(data_as_of: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    boundaries = [
        pd.Timestamp(f"{year}-01-01") for year in range(2020, data_as_of.year + 1)
    ]
    boundaries.append(data_as_of + timedelta(days=1))
    return [
        (start, end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if start < end
    ]


def run_outer_task(
    horizon: int,
    data: pd.DataFrame,
    columns: list[str],
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chosen, quantile_policy, inner = choose_model(data, columns, fold_start)
    inner.insert(0, "horizon", horizon)
    inner.insert(0, "outer_start", fold_start)

    # Outer test is allowed to contain labels that became known later because this
    # is retrospective EVALUATION. They never participate in model/threshold choice.
    test = data.loc[
        data["date"].ge(fold_start)
        & data["date"].lt(fold_end)
        & data["valid"]
    ]
    # Refit on every matured row available by fold_start. The selected policy is
    # represented by quantiles, then remapped to the refitted model's score scale.
    models, thresholds, inner_reference_scored = refit_selected_policy(
        data, columns, fold_start, chosen, quantile_policy
    )
    scored = score_models(models, test, columns)
    threshold_curve = truth_threshold_curve(
        scored, inner_reference_scored, horizon, fold_start, fold_end
    )
    truth_stream, local_stream, target_intersection = independent_target_streams(
        scored, thresholds
    )
    selected = add_message_facts(apply_thresholds(scored, thresholds))
    selected["horizon"] = horizon
    selected["outer_start"] = fold_start
    selected["model_config"] = chosen

    # This is a BACKTEST file, so it intentionally contains future outcomes.
    backtest_signal_columns = [
        "date",
        "currency",
        "horizon",
        "outer_start",
        "model_config",
        "p_truth_now",
        "p_local_min",
        "predicted_benefit_bps",
        "truth_now",
        "local_min",
        "benefit_bps",
        "message_fact",
        "observation_index",
    ]

    fold_metrics = evaluate_fold(selected, test, horizon, fold_start, fold_end)
    fold_metrics = attach_target_stream_frequencies(
        fold_metrics,
        truth_stream,
        local_stream,
        target_intersection,
        fold_start,
        fold_end,
    )
    fold_metrics["model_config"] = chosen
    return fold_metrics, selected[backtest_signal_columns], inner, threshold_curve


def build_dataset(prices: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    features = build_features(prices)
    data = features.merge(build_labels(prices, horizon), on=["date", "currency"], how="inner")
    return data, feature_columns(data)


def signals_as_of_date(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    """Production-style signal calculation using ONLY prices available by `as_of`.

    The returned table contains no future outcomes. It is intended to satisfy the
    case requirement: "what signals would we have produced as of date T?"
    """
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")

    as_of = pd.Timestamp(as_of).normalize()
    causal_prices = prices.loc[prices["date"].le(as_of)].copy()
    if causal_prices.empty:
        return pd.DataFrame()

    data, columns = build_dataset(causal_prices, horizon)
    target_dates = causal_prices.loc[
        causal_prices["currency"].isin(TARGET_CURRENCIES), "date"
    ]
    if target_dates.empty:
        return pd.DataFrame()
    score_date = pd.Timestamp(target_dates.max())

    chosen, quantile_policy, _ = choose_model(data, columns, score_date)
    models, thresholds, _ = refit_selected_policy(
        data, columns, score_date, chosen, quantile_policy
    )
    score = data.loc[data["date"].eq(score_date)].copy()
    if score.empty:
        return pd.DataFrame()

    scored = score_models(models, score, columns)
    selected = apply_thresholds(scored, thresholds)
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "currency",
                "horizon",
                "scenario",
                "direction",
                "strength",
                "speed",
                "p_truth_now",
                "p_local_min",
                "predicted_benefit_bps",
                "message_fact",
                "model_config",
            ]
        )

    selected = add_message_facts(selected)
    selected["horizon"] = horizon
    selected["scenario"] = "favourable_now"
    selected["direction"] = "lower_rub_per_unit_is_better"
    selected["strength"] = selected["predicted_benefit_bps"]
    selected["speed"] = selected["return_mean_3"] * 10_000
    selected["model_config"] = chosen

    production_columns = [
        "date",
        "currency",
        "horizon",
        "scenario",
        "direction",
        "strength",
        "speed",
        "p_truth_now",
        "p_local_min",
        "predicted_benefit_bps",
        "message_fact",
        "model_config",
    ]
    return selected[production_columns].sort_values(["date", "currency"])


def run_backtest(root: Path) -> None:
    output_dir = root / "data/processed/ml_signals"
    output_dir.mkdir(parents=True, exist_ok=True)

    prices = load_ml_prices(latest_default_input())
    features = build_features(prices)
    data_as_of = pd.Timestamp(prices["date"].max())
    folds = outer_folds(data_as_of)

    metrics_parts: list[pd.DataFrame] = []
    signal_parts: list[pd.DataFrame] = []
    threshold_parts: list[pd.DataFrame] = []
    threshold_curve_parts: list[pd.DataFrame] = []
    datasets: dict[int, tuple[pd.DataFrame, list[str]]] = {}

    for horizon in HORIZONS:
        print(f"Preparing h={horizon}", flush=True)
        data = features.merge(build_labels(prices, horizon), on=["date", "currency"], how="inner")
        datasets[horizon] = (data, feature_columns(data))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                run_outer_task,
                horizon,
                datasets[horizon][0],
                datasets[horizon][1],
                fold_start,
                fold_end,
            ): (horizon, fold_start, fold_end)
            for horizon in HORIZONS
            for fold_start, fold_end in folds
        }
        for future in as_completed(futures):
            horizon, fold_start, fold_end = futures[future]
            fold_metrics, selected, inner, threshold_curve = future.result()
            metrics_parts.append(fold_metrics)
            signal_parts.append(selected)
            threshold_parts.append(inner)
            threshold_curve_parts.append(threshold_curve)
            print(
                f"Finished h={horizon}, outer={fold_start.date()}..{(fold_end - timedelta(days=1)).date()}",
                flush=True,
            )

    metrics = pd.concat(metrics_parts, ignore_index=True).sort_values(
        ["horizon", "fold_start", "currency"]
    )
    signals = pd.concat(signal_parts, ignore_index=True).sort_values(
        ["horizon", "date", "currency"]
    )
    thresholds = pd.concat(threshold_parts, ignore_index=True)
    threshold_curves = pd.concat(threshold_curve_parts, ignore_index=True).sort_values(
        ["horizon", "fold_start", "currency", "score_quantile"]
    )

    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    signals.to_csv(output_dir / "signals_backtest.csv", index=False)
    thresholds.to_csv(output_dir / "nested_selection.csv", index=False)
    threshold_curves.to_csv(output_dir / "truth_threshold_curves.csv", index=False)

    stability = metrics.groupby(["horizon", "currency"], as_index=False).agg(
        folds=("fold_start", "nunique"),
        signals=("signal_count", "sum"),
        median_truth_stream_frequency=("truth_stream_per_week", "median"),
        median_local_stream_frequency=("local_stream_per_week", "median"),
        median_target_intersection_frequency=("target_intersection_per_week", "median"),
        median_frequency=("signals_per_week", "median"),
        min_frequency=("signals_per_week", "min"),
        median_local_lift=("local_min_lift", "median"),
        min_local_lift=("local_min_lift", "min"),
        median_truth_lift=("truth_now_lift", "median"),
        min_truth_lift=("truth_now_lift", "min"),
        median_benefit_bps=("benefit_bps", "median"),
        min_benefit_bps=("benefit_bps", "min"),
        min_benefit_ci_low_95=("benefit_ci_low_95", "min"),
        significant_benefit_folds=("benefit_significant_5pct", "sum"),
        median_clustered_share=("clustered_share_within_5_observations", "median"),
        max_clustered_share=("clustered_share_within_5_observations", "max"),
    )
    stability["significant_benefit_share"] = (
        stability["significant_benefit_folds"] / stability["folds"]
    )
    stability.to_csv(output_dir / "stability.csv", index=False)

    metadata = {
        "model": "pooled HistGradientBoosting: truth_now + local_min classifiers and benefit regressor",
        "scoring_population": "every published target-currency date",
        "horizons": list(HORIZONS),
        "outer_folds": [[str(a.date()), str(b.date())] for a, b in folds],
        "inner_validation": "one year before every outer fold; only labels matured before outer_start are eligible",
        "thresholds": (
            "quantiles selected separately by currency on inner validation; "
            "after full pre-outer refit they are remapped on the last historical year"
        ),
        "cooldown": "disabled during model/threshold selection",
        "frequency": "reported only; not used as a selection constraint",
        "observation_index": "excluded from model features; used only for cooldown/clustering",
        "holiday_source": f"python-holidays {holidays.__version__}; recipient-country calendars",
        "data_as_of": str(data_as_of.date()),
        "confirmatory_status": "nested development walk-forward; a future untouched period is still required",
        "backtest_signal_file": "signals_backtest.csv contains outcomes and must not be used as production output",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Saved nested ML backtest to {output_dir}")


def run_training_window_task(
    horizon: int,
    data: pd.DataFrame,
    columns: list[str],
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate every fixed history window on the same untouched outer fold."""
    metric_parts: list[pd.DataFrame] = []
    signal_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    test = data.loc[
        data["date"].ge(fold_start)
        & data["date"].lt(fold_end)
        & data["valid"]
    ]

    for config in TRAINING_WINDOW_CONFIGS:
        inner_result = select_policy_for_config(
            data, columns, fold_start, config
        )
        if inner_result is None:
            continue
        quantile_policy, diagnostics = inner_result
        models, thresholds, _ = refit_selected_policy(
            data, columns, fold_start, config, quantile_policy
        )
        scored = score_models(models, test, columns)
        truth_stream, local_stream, target_intersection = independent_target_streams(
            scored, thresholds
        )
        selected = add_message_facts(apply_thresholds(scored, thresholds))
        selected["horizon"] = horizon
        selected["outer_start"] = fold_start
        selected["model_config"] = config

        for currency, values in thresholds.items():
            mask = diagnostics["currency"].eq(currency)
            diagnostics.loc[mask, "refit_truth_threshold"] = values["truth_threshold"]
            diagnostics.loc[mask, "refit_local_threshold"] = values["local_threshold"]
            diagnostics.loc[mask, "refit_benefit_threshold"] = values["benefit_threshold"]
        diagnostics.insert(0, "model_config", config)
        diagnostics.insert(0, "horizon", horizon)
        diagnostics.insert(0, "outer_start", fold_start)
        selection_parts.append(diagnostics)

        fold_metrics = evaluate_fold(
            selected, test, horizon, fold_start, fold_end
        )
        fold_metrics = attach_target_stream_frequencies(
            fold_metrics,
            truth_stream,
            local_stream,
            target_intersection,
            fold_start,
            fold_end,
        )
        fold_metrics["model_config"] = config
        metric_parts.append(fold_metrics)
        signal_parts.append(
            selected[
                [
                    "date",
                    "currency",
                    "horizon",
                    "outer_start",
                    "model_config",
                    "p_truth_now",
                    "p_local_min",
                    "predicted_benefit_bps",
                    "truth_now",
                    "local_min",
                    "benefit_bps",
                    "message_fact",
                    "observation_index",
                ]
            ]
        )

    if not metric_parts:
        raise RuntimeError(
            f"No valid window configuration for h={horizon}, outer={fold_start.date()}"
        )
    return (
        pd.concat(metric_parts, ignore_index=True),
        pd.concat(signal_parts, ignore_index=True),
        pd.concat(selection_parts, ignore_index=True),
    )


def run_training_window_experiment(
    root: Path,
    horizons: tuple[int, ...],
    outer_start_year: int,
    workers: int,
) -> None:
    """Compare recent and long training histories without selecting on outer data."""
    output_dir = root / "data/processed/ml_training_windows"
    output_dir.mkdir(parents=True, exist_ok=True)

    prices = load_ml_prices(latest_default_input())
    features = build_features(prices)
    data_as_of = pd.Timestamp(prices["date"].max())
    folds = [
        fold
        for fold in outer_folds(data_as_of)
        if fold[0].year >= outer_start_year
    ]
    if not folds:
        raise ValueError(
            f"No outer folds from {outer_start_year}; data_as_of={data_as_of.date()}"
        )

    datasets: dict[int, tuple[pd.DataFrame, list[str]]] = {}
    for horizon in horizons:
        print(f"Preparing h={horizon}", flush=True)
        data = features.merge(
            build_labels(prices, horizon), on=["date", "currency"], how="inner"
        )
        datasets[horizon] = (data, feature_columns(data))

    metric_parts: list[pd.DataFrame] = []
    signal_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_training_window_task,
                horizon,
                datasets[horizon][0],
                datasets[horizon][1],
                fold_start,
                fold_end,
            ): (horizon, fold_start, fold_end)
            for horizon in horizons
            for fold_start, fold_end in folds
        }
        for future in as_completed(futures):
            horizon, fold_start, fold_end = futures[future]
            metrics, signals, selections = future.result()
            metric_parts.append(metrics)
            signal_parts.append(signals)
            selection_parts.append(selections)
            print(
                f"Finished h={horizon}, outer={fold_start.date()}.."
                f"{(fold_end - timedelta(days=1)).date()}",
                flush=True,
            )

    metrics = pd.concat(metric_parts, ignore_index=True).sort_values(
        ["model_config", "horizon", "fold_start", "currency"]
    )
    signals = pd.concat(signal_parts, ignore_index=True).sort_values(
        ["model_config", "horizon", "date", "currency"]
    )
    selections = pd.concat(selection_parts, ignore_index=True).sort_values(
        ["model_config", "horizon", "outer_start", "currency"]
    )

    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    signals.to_csv(output_dir / "signals_backtest.csv.gz", index=False)
    selections.to_csv(output_dir / "inner_selections.csv", index=False)

    stability = metrics.groupby(
        ["model_config", "horizon", "currency"], as_index=False
    ).agg(
        folds=("fold_start", "nunique"),
        signals=("signal_count", "sum"),
        median_truth_lift=("truth_now_lift", "median"),
        min_truth_lift=("truth_now_lift", "min"),
        median_local_lift=("local_min_lift", "median"),
        min_local_lift=("local_min_lift", "min"),
        median_benefit_bps=("benefit_bps", "median"),
        min_benefit_bps=("benefit_bps", "min"),
        significant_benefit_share=("benefit_significant_5pct", "mean"),
        median_truth_stream_frequency=("truth_stream_per_week", "median"),
        median_local_stream_frequency=("local_stream_per_week", "median"),
        median_target_intersection_frequency=("target_intersection_per_week", "median"),
        median_frequency=("signals_per_week", "median"),
    )
    stability.to_csv(output_dir / "stability.csv", index=False)

    summary = metrics.groupby(["model_config", "horizon"], as_index=False).agg(
        fold_currency_cells=("currency", "size"),
        signals=("signal_count", "sum"),
        median_truth_lift=("truth_now_lift", "median"),
        q25_truth_lift=("truth_now_lift", lambda values: values.quantile(0.25)),
        min_truth_lift=("truth_now_lift", "min"),
        truth_lift_115_share=("truth_now_lift", lambda values: values.ge(1.15).mean()),
        truth_lift_130_share=("truth_now_lift", lambda values: values.ge(1.30).mean()),
        median_local_lift=("local_min_lift", "median"),
        median_benefit_bps=("benefit_bps", "median"),
        nonnegative_benefit_share=("benefit_bps", lambda values: values.ge(0).mean()),
        significant_benefit_share=("benefit_significant_5pct", "mean"),
        median_truth_stream_frequency=("truth_stream_per_week", "median"),
        median_local_stream_frequency=("local_stream_per_week", "median"),
        median_target_intersection_frequency=("target_intersection_per_week", "median"),
        median_frequency=("signals_per_week", "median"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    metadata = {
        "experiment": "fixed training-window ablation with full pre-outer refit",
        "training_windows": list(TRAINING_WINDOW_CONFIGS),
        "horizons": list(horizons),
        "outer_start_year": outer_start_year,
        "outer_folds": [[str(a.date()), str(b.date())] for a, b in folds],
        "threshold_policy": (
            "quantiles selected on previous inner-validation year; after full refit, "
            "quantiles remapped using that same historical year's refitted-model scores"
        ),
        "frequency": "reported only; never used for selection or filtering",
        "data_as_of": str(data_as_of.date()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Saved training-window experiment to {output_dir}")


def run_candidate_gate_task(
    horizon: int,
    gate: str,
    data: pd.DataFrame,
    columns: list[str],
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train truth_now only inside one causal candidate gate and test one OOT fold."""
    validation_start = fold_start - pd.DateOffset(years=1)
    inner_train_pool = mature_training_rows(
        data,
        model_train_start(CANDIDATE_HISTORY_CONFIG, validation_start),
        validation_start,
    )
    validation_pool = mature_validation_rows(data, validation_start, fold_start)
    inner_train = inner_train_pool.loc[candidate_gate_mask(inner_train_pool, gate)]
    validation_candidates = validation_pool.loc[
        candidate_gate_mask(validation_pool, gate)
    ]
    if len(inner_train) < 500 or validation_candidates.empty:
        raise RuntimeError(
            f"Too few candidate rows for gate={gate}, h={horizon}, "
            f"outer={fold_start.date()}"
        )
    if inner_train["truth_now"].nunique() < 2:
        raise RuntimeError(f"One-class inner train for gate={gate}, h={horizon}")

    inner_model = fit_truth_model(inner_train, columns)
    inner_local_model = fit_local_model(inner_train, columns)
    validation_scored = score_truth_model(
        inner_model, validation_candidates, columns
    )
    validation_scored = score_local_model(
        inner_local_model, validation_scored, columns
    )
    quantile_policy, truth_diagnostics = choose_truth_only_thresholds(
        validation_scored, validation_pool, validation_start, fold_start
    )
    local_quantile_policy, local_diagnostics = choose_local_only_thresholds(
        validation_scored, validation_pool, validation_start, fold_start
    )

    # Stage two is refitted on every matured candidate available at outer_start.
    refit_pool = mature_training_rows(
        data,
        model_train_start(CANDIDATE_HISTORY_CONFIG, fold_start),
        fold_start,
    )
    refit_train = refit_pool.loc[candidate_gate_mask(refit_pool, gate)]
    if len(refit_train) < 500 or refit_train["truth_now"].nunique() < 2:
        raise RuntimeError(f"Invalid refit population for gate={gate}, h={horizon}")
    model = fit_truth_model(refit_train, columns)
    local_model = fit_local_model(refit_train, columns)

    # Remap the selected quantiles on the same historical reference period,
    # now scored by the refitted model. No outer outcome or score is involved.
    reference_scored = score_truth_model(
        model, validation_candidates, columns
    )
    reference_scored = score_local_model(
        local_model, reference_scored, columns
    )
    thresholds = remap_truth_thresholds(quantile_policy, reference_scored)
    local_thresholds = remap_local_thresholds(
        local_quantile_policy, reference_scored
    )
    for currency, threshold in thresholds.items():
        truth_diagnostics.loc[
            truth_diagnostics["currency"].eq(currency), "refit_truth_threshold"
        ] = threshold
    for currency, threshold in local_thresholds.items():
        local_diagnostics.loc[
            local_diagnostics["currency"].eq(currency), "refit_local_threshold"
        ] = threshold
    truth_diagnostics["target_stream"] = "truth_now"
    local_diagnostics["target_stream"] = "local_min"
    diagnostics = pd.concat(
        [truth_diagnostics, local_diagnostics], ignore_index=True, sort=False
    )
    diagnostics.insert(0, "gate", gate)
    diagnostics.insert(0, "horizon", horizon)
    diagnostics.insert(0, "outer_start", fold_start)

    test_pool = data.loc[
        data["date"].ge(fold_start)
        & data["date"].lt(fold_end)
        & data["valid"]
    ]
    test_candidates = test_pool.loc[candidate_gate_mask(test_pool, gate)]
    scored_candidates = score_truth_model(model, test_candidates, columns)
    scored_candidates = score_local_model(
        local_model, scored_candidates, columns
    )
    truth_stream = apply_truth_thresholds(scored_candidates, thresholds)
    local_stream = apply_local_thresholds(scored_candidates, local_thresholds)
    joint_keys = truth_stream[["date", "currency"]].merge(
        local_stream[["date", "currency"]], on=["date", "currency"], how="inner"
    )
    joint_stream = scored_candidates.merge(
        joint_keys, on=["date", "currency"], how="inner"
    )
    selected = add_message_facts(truth_stream)
    selected["horizon"] = horizon
    selected["outer_start"] = fold_start
    selected["gate"] = gate

    metrics = evaluate_fold(selected, test_pool, horizon, fold_start, fold_end)
    metrics = attach_target_stream_frequencies(
        metrics,
        truth_stream,
        local_stream,
        joint_stream,
        fold_start,
        fold_end,
    )
    metrics["gate"] = gate
    for currency in TARGET_CURRENCIES:
        pool_count = int(test_pool["currency"].eq(currency).sum())
        candidate_count = int(test_candidates["currency"].eq(currency).sum())
        mask = metrics["currency"].eq(currency)
        metrics.loc[mask, "candidate_count"] = candidate_count
        metrics.loc[mask, "candidate_share"] = (
            candidate_count / pool_count if pool_count else np.nan
        )

    signal_columns = [
        "date",
        "currency",
        "horizon",
        "outer_start",
        "gate",
        "p_truth_now",
        "p_local_min",
        "truth_now",
        "local_min",
        "benefit_bps",
        "message_fact",
        "observation_index",
    ]
    return metrics, selected[signal_columns], diagnostics


def run_candidate_gate_experiment(
    root: Path,
    horizons: tuple[int, ...],
    outer_start_year: int,
    workers: int,
    gates: tuple[str, ...] = CANDIDATE_GATES,
    output_subdir: str = "ml_candidate_gate",
    experiment_label: str = "candidate-gate ablation",
) -> None:
    """Controlled all-days vs favourable-level candidate-population ablation."""
    unknown_gates = set(gates) - set(CANDIDATE_GATES)
    if unknown_gates:
        raise ValueError(f"Unknown candidate gates: {sorted(unknown_gates)}")
    output_dir = root / "data/processed" / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    prices = load_ml_prices(latest_default_input())
    features = build_features(prices)
    data_as_of = pd.Timestamp(prices["date"].max())
    folds = [
        fold for fold in outer_folds(data_as_of) if fold[0].year >= outer_start_year
    ]
    if not folds:
        raise ValueError(
            f"No outer folds from {outer_start_year}; data_as_of={data_as_of.date()}"
        )

    datasets: dict[int, tuple[pd.DataFrame, list[str]]] = {}
    for horizon in horizons:
        print(f"Preparing h={horizon}", flush=True)
        data = features.merge(
            build_labels(prices, horizon), on=["date", "currency"], how="inner"
        )
        datasets[horizon] = (data, feature_columns(data))

    metric_parts: list[pd.DataFrame] = []
    signal_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_candidate_gate_task,
                horizon,
                gate,
                datasets[horizon][0],
                datasets[horizon][1],
                fold_start,
                fold_end,
            ): (horizon, gate, fold_start, fold_end)
            for horizon in horizons
            for gate in gates
            for fold_start, fold_end in folds
        }
        for future in as_completed(futures):
            horizon, gate, fold_start, fold_end = futures[future]
            metrics, signals, selections = future.result()
            metric_parts.append(metrics)
            signal_parts.append(signals)
            selection_parts.append(selections)
            print(
                f"Finished gate={gate}, h={horizon}, outer={fold_start.date()}.."
                f"{(fold_end - timedelta(days=1)).date()}",
                flush=True,
            )

    metrics = pd.concat(metric_parts, ignore_index=True).sort_values(
        ["gate", "horizon", "fold_start", "currency"]
    )
    signals = pd.concat(signal_parts, ignore_index=True).sort_values(
        ["gate", "horizon", "date", "currency"]
    )
    selections = pd.concat(selection_parts, ignore_index=True).sort_values(
        ["gate", "horizon", "outer_start", "currency"]
    )
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    signals.to_csv(output_dir / "signals_backtest.csv.gz", index=False)
    selections.to_csv(output_dir / "inner_selections.csv", index=False)

    stability = metrics.groupby(["gate", "horizon", "currency"], as_index=False).agg(
        folds=("fold_start", "nunique"),
        candidates=("candidate_count", "sum"),
        signals=("signal_count", "sum"),
        median_candidate_share=("candidate_share", "median"),
        median_truth_hit_rate=("truth_now_hit_rate", "median"),
        median_truth_random=("truth_now_random", "median"),
        median_truth_lift=("truth_now_lift", "median"),
        min_truth_lift=("truth_now_lift", "min"),
        median_local_lift=("local_min_lift", "median"),
        median_benefit_bps=("benefit_bps", "median"),
        median_benefit_ci_low_95=("benefit_ci_low_95", "median"),
        median_benefit_ci_high_95=("benefit_ci_high_95", "median"),
        min_benefit_bps=("benefit_bps", "min"),
        significant_benefit_share=("benefit_significant_5pct", "mean"),
        median_truth_stream_frequency=("truth_stream_per_week", "median"),
        median_local_stream_frequency=("local_stream_per_week", "median"),
        median_target_intersection_frequency=("target_intersection_per_week", "median"),
        median_frequency=("signals_per_week", "median"),
    )
    stability.to_csv(output_dir / "stability.csv", index=False)

    summary = metrics.groupby(["gate", "horizon"], as_index=False).agg(
        fold_currency_cells=("currency", "size"),
        candidates=("candidate_count", "sum"),
        signals=("signal_count", "sum"),
        median_candidate_share=("candidate_share", "median"),
        median_truth_hit_rate=("truth_now_hit_rate", "median"),
        median_truth_random=("truth_now_random", "median"),
        median_truth_lift=("truth_now_lift", "median"),
        q25_truth_lift=("truth_now_lift", lambda values: values.quantile(0.25)),
        min_truth_lift=("truth_now_lift", "min"),
        truth_lift_115_share=("truth_now_lift", lambda values: values.ge(1.15).mean()),
        truth_lift_130_share=("truth_now_lift", lambda values: values.ge(1.30).mean()),
        median_local_lift=("local_min_lift", "median"),
        median_benefit_bps=("benefit_bps", "median"),
        median_benefit_ci_low_95=("benefit_ci_low_95", "median"),
        median_benefit_ci_high_95=("benefit_ci_high_95", "median"),
        nonnegative_benefit_share=("benefit_bps", lambda values: values.ge(0).mean()),
        significant_benefit_share=("benefit_significant_5pct", "mean"),
        median_truth_stream_frequency=("truth_stream_per_week", "median"),
        median_local_stream_frequency=("local_stream_per_week", "median"),
        median_target_intersection_frequency=("target_intersection_per_week", "median"),
        median_frequency=("signals_per_week", "median"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    metadata = {
        "experiment": experiment_label,
        "gates": {key: value for key, value in {
            "all_days": "control: every published date",
            "level60_q70": "P[T] is no higher than at least 70% of previous 60 publications",
            "level60_q80": "P[T] is no higher than at least 80% of previous 60 publications",
            "level60_q90": "P[T] is no higher than at least 90% of previous 60 publications",
        }.items() if key in gates},
        "history_config": CANDIDATE_HISTORY_CONFIG,
        "horizons": list(horizons),
        "outer_start_year": outer_start_year,
        "outer_folds": [[str(a.date()), str(b.date())] for a, b in folds],
        "official_random_baseline": "all published dates in the same currency and outer fold",
        "selection": (
            "truth and local score thresholds are selected independently; "
            "primary candidate-gate metrics use the truth stream, local is diagnostic"
        ),
        "frequency": "reported only; never used for selection or filtering",
        "data_as_of": str(data_as_of.date()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Saved {experiment_label} to {output_dir}")


def run_regret_regression_task(
    horizon: int,
    data: pd.DataFrame,
    columns: list[str],
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate regret, local-min and OR-union streams on one outer fold."""
    validation_start = fold_start - pd.DateOffset(years=1)
    inner_train = mature_training_rows(
        data,
        model_train_start("long_history", validation_start),
        validation_start,
    )
    validation = mature_validation_rows(data, validation_start, fold_start)
    if len(inner_train) < 500 or validation.empty:
        raise RuntimeError(f"Insufficient inner data for h={horizon}, {fold_start.date()}")

    inner_regret_model = fit_regret_model(inner_train, columns)
    inner_local_model = fit_local_model(inner_train, columns)
    validation_scored = score_regret_model(
        inner_regret_model, validation, columns
    )
    validation_scored = score_local_model(
        inner_local_model, validation_scored, columns
    )
    regret_policy, regret_diagnostics = choose_regret_thresholds(
        validation_scored, validation, validation_start, fold_start
    )
    local_policy, local_diagnostics = choose_local_only_thresholds(
        validation_scored, validation, validation_start, fold_start
    )

    refit_train = mature_training_rows(
        data, model_train_start("long_history", fold_start), fold_start
    )
    regret_model = fit_regret_model(refit_train, columns)
    local_model = fit_local_model(refit_train, columns)
    reference_scored = score_regret_model(regret_model, validation, columns)
    reference_scored = score_local_model(local_model, reference_scored, columns)
    regret_thresholds = remap_regret_thresholds(regret_policy, reference_scored)
    local_thresholds = remap_local_thresholds(local_policy, reference_scored)

    for currency, threshold in regret_thresholds.items():
        regret_diagnostics.loc[
            regret_diagnostics["currency"].eq(currency), "refit_regret_threshold_bps"
        ] = threshold
    for currency, threshold in local_thresholds.items():
        local_diagnostics.loc[
            local_diagnostics["currency"].eq(currency), "refit_local_threshold"
        ] = threshold
    regret_diagnostics["target_stream"] = "regret_truth"
    local_diagnostics["target_stream"] = "local_min"
    diagnostics = pd.concat(
        [regret_diagnostics, local_diagnostics], ignore_index=True, sort=False
    )
    diagnostics.insert(0, "horizon", horizon)
    diagnostics.insert(0, "outer_start", fold_start)

    test = data.loc[
        data["date"].ge(fold_start)
        & data["date"].lt(fold_end)
        & data["valid"]
    ]
    scored = score_regret_model(regret_model, test, columns)
    scored = score_local_model(local_model, scored, columns)
    regret_stream = apply_regret_thresholds(scored, regret_thresholds)
    local_stream = apply_local_thresholds(scored, local_thresholds)
    union_stream = union_signal_streams(regret_stream, local_stream)

    metric_parts: list[pd.DataFrame] = []
    signal_parts: list[pd.DataFrame] = []
    for stream_name, stream in (
        ("regret_truth", regret_stream),
        ("local_min", local_stream),
        ("union", union_stream),
    ):
        stream = add_message_facts(stream)
        stream["stream"] = stream_name
        stream["horizon"] = horizon
        stream["outer_start"] = fold_start
        metrics = evaluate_fold(stream, test, horizon, fold_start, fold_end)
        metrics["stream"] = stream_name
        metric_parts.append(metrics)
        signal_parts.append(
            stream[
                [
                    "date",
                    "currency",
                    "horizon",
                    "outer_start",
                    "stream",
                    "predicted_future_regret_bps",
                    "p_local_min",
                    "truth_now",
                    "local_min",
                    "future_regret_bps",
                    "benefit_bps",
                    "message_fact",
                    "observation_index",
                ]
            ]
        )

    metrics = pd.concat(metric_parts, ignore_index=True)
    for currency in TARGET_CURRENCIES:
        currency_scored = scored.loc[scored["currency"].eq(currency)]
        mae = (
            currency_scored["predicted_future_regret_bps"]
            .sub(currency_scored["future_regret_bps"])
            .abs()
            .mean()
        )
        rank_corr = currency_scored[
            ["predicted_future_regret_bps", "future_regret_bps"]
        ].corr(method="spearman").iloc[0, 1]
        mask = metrics["currency"].eq(currency)
        metrics.loc[mask, "regret_mae_bps"] = mae
        metrics.loc[mask, "regret_spearman"] = rank_corr

    overlap_rows: list[dict[str, object]] = []
    for currency in TARGET_CURRENCIES:
        regret_keys = set(
            regret_stream.loc[regret_stream["currency"].eq(currency), "date"]
        )
        local_keys = set(
            local_stream.loc[local_stream["currency"].eq(currency), "date"]
        )
        intersection = len(regret_keys & local_keys)
        union_count = len(regret_keys | local_keys)
        overlap_rows.append(
            {
                "fold_start": fold_start,
                "fold_end": fold_end,
                "currency": currency,
                "horizon": horizon,
                "regret_count": len(regret_keys),
                "local_count": len(local_keys),
                "intersection_count": intersection,
                "union_count": union_count,
                "jaccard": intersection / union_count if union_count else np.nan,
            }
        )
    return (
        metrics,
        pd.concat(signal_parts, ignore_index=True),
        diagnostics,
        pd.DataFrame(overlap_rows),
    )


def run_regret_regression_experiment(
    root: Path,
    horizons: tuple[int, ...],
    outer_start_year: int,
    workers: int,
) -> None:
    """All-days regret regression plus independent local-min stream and OR union."""
    output_dir = root / "data/processed/ml_regret_regression"
    output_dir.mkdir(parents=True, exist_ok=True)
    prices = load_ml_prices(latest_default_input())
    features = build_features(prices)
    data_as_of = pd.Timestamp(prices["date"].max())
    folds = [
        fold for fold in outer_folds(data_as_of) if fold[0].year >= outer_start_year
    ]
    if not folds:
        raise ValueError(
            f"No outer folds from {outer_start_year}; data_as_of={data_as_of.date()}"
        )

    datasets: dict[int, tuple[pd.DataFrame, list[str]]] = {}
    for horizon in horizons:
        print(f"Preparing h={horizon}", flush=True)
        data = features.merge(
            build_labels(prices, horizon), on=["date", "currency"], how="inner"
        )
        datasets[horizon] = (data, feature_columns(data))

    metric_parts: list[pd.DataFrame] = []
    signal_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    overlap_parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_regret_regression_task,
                horizon,
                datasets[horizon][0],
                datasets[horizon][1],
                fold_start,
                fold_end,
            ): (horizon, fold_start, fold_end)
            for horizon in horizons
            for fold_start, fold_end in folds
        }
        for future in as_completed(futures):
            horizon, fold_start, fold_end = futures[future]
            metrics, signals, selections, overlap = future.result()
            metric_parts.append(metrics)
            signal_parts.append(signals)
            selection_parts.append(selections)
            overlap_parts.append(overlap)
            print(
                f"Finished h={horizon}, outer={fold_start.date()}.."
                f"{(fold_end - timedelta(days=1)).date()}",
                flush=True,
            )

    metrics = pd.concat(metric_parts, ignore_index=True).sort_values(
        ["stream", "horizon", "fold_start", "currency"]
    )
    signals = pd.concat(signal_parts, ignore_index=True).sort_values(
        ["stream", "horizon", "date", "currency"]
    )
    selections = pd.concat(selection_parts, ignore_index=True).sort_values(
        ["target_stream", "horizon", "outer_start", "currency"]
    )
    overlap = pd.concat(overlap_parts, ignore_index=True).sort_values(
        ["horizon", "fold_start", "currency"]
    )
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    signals.to_csv(output_dir / "signals_backtest.csv.gz", index=False)
    selections.to_csv(output_dir / "inner_selections.csv", index=False)
    overlap.to_csv(output_dir / "stream_overlap.csv", index=False)

    stability = metrics.groupby(["stream", "horizon", "currency"], as_index=False).agg(
        folds=("fold_start", "nunique"),
        signals=("signal_count", "sum"),
        median_truth_lift=("truth_now_lift", "median"),
        min_truth_lift=("truth_now_lift", "min"),
        median_local_lift=("local_min_lift", "median"),
        min_local_lift=("local_min_lift", "min"),
        median_benefit_bps=("benefit_bps", "median"),
        min_benefit_bps=("benefit_bps", "min"),
        significant_benefit_share=("benefit_significant_5pct", "mean"),
        median_frequency=("signals_per_week", "median"),
        median_regret_mae_bps=("regret_mae_bps", "median"),
        median_regret_spearman=("regret_spearman", "median"),
    )
    stability.to_csv(output_dir / "stability.csv", index=False)

    summary = metrics.groupby(["stream", "horizon"], as_index=False).agg(
        fold_currency_cells=("currency", "size"),
        signals=("signal_count", "sum"),
        median_truth_lift=("truth_now_lift", "median"),
        q25_truth_lift=("truth_now_lift", lambda values: values.quantile(0.25)),
        truth_lift_115_share=("truth_now_lift", lambda values: values.ge(1.15).mean()),
        truth_lift_130_share=("truth_now_lift", lambda values: values.ge(1.30).mean()),
        median_local_lift=("local_min_lift", "median"),
        local_lift_130_share=("local_min_lift", lambda values: values.ge(1.30).mean()),
        median_benefit_bps=("benefit_bps", "median"),
        nonnegative_benefit_share=("benefit_bps", lambda values: values.ge(0).mean()),
        significant_benefit_share=("benefit_significant_5pct", "mean"),
        median_frequency=("signals_per_week", "median"),
        median_regret_mae_bps=("regret_mae_bps", "median"),
        median_regret_spearman=("regret_spearman", "median"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    metadata = {
        "experiment": "all-days future-regret regression plus local-min classifier",
        "future_regret_bps": "max(P[T]/min(P[T+1:T+h])-1, 0)*10000",
        "regression": "HistGradientBoostingRegressor absolute_error; lower prediction is better",
        "streams": {
            "regret_truth": "low predicted regret, evaluated with exact truth_now",
            "local_min": "independent high p_local_min stream",
            "union": "OR of regret_truth and local_min; duplicate currency/date emitted once",
        },
        "history_config": "long_history from 2014",
        "horizons": list(horizons),
        "outer_start_year": outer_start_year,
        "outer_folds": [[str(a.date()), str(b.date())] for a, b in folds],
        "frequency": "reported only; never used for selection or filtering",
        "data_as_of": str(data_as_of.date()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Saved regret-regression experiment to {output_dir}")


def joint_score_columns(architecture: str) -> tuple[str, ...]:
    if architecture == "two_head":
        return ("p_truth_now", "predicted_benefit_q25")
    if architecture == "joint_classifier":
        return ("p_good_push",)
    raise ValueError(f"Unknown joint architecture: {architecture}")


def build_joint_inner_oof(
    data: pd.DataFrame,
    columns: list[str],
    architecture: str,
    outer_start: pd.Timestamp,
) -> pd.DataFrame:
    """Build three strictly forward inner OOF years for policy selection."""
    parts: list[pd.DataFrame] = []
    first_year = outer_start.year - INNER_OOF_YEARS
    for year in range(first_year, outer_start.year):
        validation_start = pd.Timestamp(f"{year}-01-01")
        validation_end = pd.Timestamp(f"{year + 1}-01-01")
        train = mature_training_rows(
            data,
            model_train_start("long_history", validation_start),
            validation_start,
        )
        validation = data.loc[
            data["date"].ge(validation_start)
            & data["date"].lt(validation_end)
            & data["label_available_date"].lt(outer_start)
            & data["valid"]
        ]
        reference = data.loc[
            data["date"].ge(validation_start - pd.DateOffset(years=1))
            & data["date"].lt(validation_start)
        ]
        if len(train) < 500 or validation.empty or reference.empty:
            continue
        models = fit_joint_architecture(train, columns, architecture)
        reference_scored = score_joint_architecture(
            models, reference, columns, architecture
        )
        validation_scored = score_joint_architecture(
            models, validation, columns, architecture
        )
        ranked = causal_score_percentiles(
            reference_scored,
            validation_scored,
            joint_score_columns(architecture),
        )
        ranked["inner_fold_start"] = validation_start
        parts.append(ranked)
    if not parts:
        raise RuntimeError(
            f"No inner OOF scores for {architecture}, outer={outer_start.date()}"
        )
    return pd.concat(parts, ignore_index=True)


def joint_policy_mask(
    scored: pd.DataFrame,
    architecture: str,
    truth_quantile: float | None = None,
    benefit_quantile: float | None = None,
    joint_quantile: float | None = None,
) -> pd.Series:
    if architecture == "two_head":
        if truth_quantile is None or benefit_quantile is None:
            raise ValueError("two_head requires truth and benefit quantiles")
        return (
            scored["p_truth_now_rank"].ge(truth_quantile)
            & scored["predicted_benefit_q25_rank"].ge(benefit_quantile)
            & scored["predicted_benefit_q25"].ge(0.0)
        )
    if architecture == "joint_classifier":
        if joint_quantile is None:
            raise ValueError("joint_classifier requires joint_quantile")
        return scored["p_good_push_rank"].ge(joint_quantile)
    raise ValueError(f"Unknown joint architecture: {architecture}")


def select_joint_global_policy(
    inner_oof: pd.DataFrame,
    architecture: str,
    policy_quantiles: tuple[float, ...] = JOINT_POLICY_QUANTILES,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Select one rank policy across currencies and multiple inner OOF years."""
    if architecture == "two_head":
        policies = [
            {
                "truth_quantile": float(truth_q),
                "benefit_quantile": float(benefit_q),
                "joint_quantile": np.nan,
            }
            for truth_q in policy_quantiles
            for benefit_q in policy_quantiles
        ]
    elif architecture == "joint_classifier":
        policies = [
            {
                "truth_quantile": np.nan,
                "benefit_quantile": np.nan,
                "joint_quantile": float(joint_q),
            }
            for joint_q in policy_quantiles
        ]
    else:
        raise ValueError(f"Unknown joint architecture: {architecture}")

    expected_cells = (
        inner_oof["inner_fold_start"].nunique() * len(TARGET_CURRENCIES)
    )
    rows: list[dict[str, object]] = []
    for policy in policies:
        selected_mask = joint_policy_mask(inner_oof, architecture, **policy)
        cell_rows: list[dict[str, object]] = []
        for (fold_start, currency), pool in inner_oof.groupby(
            ["inner_fold_start", "currency"], sort=True
        ):
            selected = inner_oof.loc[
                selected_mask
                & inner_oof["inner_fold_start"].eq(fold_start)
                & inner_oof["currency"].eq(currency)
            ]
            metric = raw_metrics(
                selected,
                pool,
                weeks_in_period(fold_start, fold_start + pd.DateOffset(years=1)),
            )
            cell_rows.append(
                {"inner_fold_start": fold_start, "currency": currency, **metric}
            )

        cells = pd.DataFrame(cell_rows)
        reliable = cells.loc[
            cells["count"].ge(MIN_VALIDATION_SIGNALS)
            & cells["truth_lift"].notna()
        ]
        by_currency = reliable.groupby("currency", as_index=False).agg(
            lift=("truth_lift", "median"),
            benefit=("benefit", "median"),
        )
        passing_currencies = int(
            (by_currency["lift"].ge(TRUTH_LIFT_TARGET)
             & by_currency["benefit"].ge(0)).sum()
        )
        both = (
            reliable["truth_lift"].ge(TRUTH_LIFT_TARGET)
            & reliable["benefit"].ge(0)
        )
        enough_coverage = bool(
            reliable["currency"].nunique() == len(TARGET_CURRENCIES)
            and len(reliable) >= max(len(TARGET_CURRENCIES), int(np.ceil(0.60 * expected_cells)))
        )
        rows.append(
            {
                **policy,
                "expected_cells": expected_cells,
                "reliable_cells": len(reliable),
                "reliable_currency_coverage": reliable["currency"].nunique(),
                "enough_coverage": enough_coverage,
                "passing_currencies": passing_currencies,
                "both_cell_share": float(both.mean()) if len(reliable) else 0.0,
                "q25_truth_lift": reliable["truth_lift"].quantile(0.25),
                "median_truth_lift": reliable["truth_lift"].median(),
                "q25_benefit_bps": reliable["benefit"].quantile(0.25),
                "median_benefit_bps": reliable["benefit"].median(),
                "signals": int(cells["count"].sum()),
            }
        )

    grid = pd.DataFrame(rows).sort_values(
        [
            "enough_coverage",
            "passing_currencies",
            "both_cell_share",
            "q25_truth_lift",
            "q25_benefit_bps",
            "reliable_cells",
            "signals",
        ],
        ascending=[False, False, False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    grid["chosen"] = False
    grid.loc[0, "chosen"] = True
    best = grid.iloc[0]
    policy = {
        key: float(best[key])
        for key in ("truth_quantile", "benefit_quantile", "joint_quantile")
        if pd.notna(best[key])
    }
    return policy, grid


def run_joint_policy_task(
    horizon: int,
    architecture: str,
    data: pd.DataFrame,
    columns: list[str],
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inner_oof = build_joint_inner_oof(data, columns, architecture, fold_start)
    policy, grid = select_joint_global_policy(inner_oof, architecture)

    refit_train = mature_training_rows(
        data, model_train_start("long_history", fold_start), fold_start
    )
    models = fit_joint_architecture(refit_train, columns, architecture)
    reference = data.loc[
        data["date"].ge(fold_start - pd.DateOffset(years=1))
        & data["date"].lt(fold_start)
    ]
    test = data.loc[
        data["date"].ge(fold_start)
        & data["date"].lt(fold_end)
        & data["valid"]
    ]
    reference_scored = score_joint_architecture(
        models, reference, columns, architecture
    )
    test_scored = score_joint_architecture(models, test, columns, architecture)
    ranked = causal_score_percentiles(
        reference_scored,
        test_scored,
        joint_score_columns(architecture),
    )
    selected = ranked.loc[joint_policy_mask(ranked, architecture, **policy)].copy()
    selected = add_message_facts(selected)
    selected["architecture"] = architecture
    selected["horizon"] = horizon
    selected["outer_start"] = fold_start

    metrics = evaluate_fold(selected, test, horizon, fold_start, fold_end)
    metrics["architecture"] = architecture
    metrics["policy_truth_quantile"] = policy.get("truth_quantile", np.nan)
    metrics["policy_benefit_quantile"] = policy.get("benefit_quantile", np.nan)
    metrics["policy_joint_quantile"] = policy.get("joint_quantile", np.nan)

    grid.insert(0, "architecture", architecture)
    grid.insert(0, "horizon", horizon)
    grid.insert(0, "outer_start", fold_start)
    signal_columns = [
        "date", "currency", "horizon", "outer_start", "architecture",
        "truth_now", "local_min", "benefit_bps", "good_push",
        "p_truth_now", "predicted_benefit_q25", "p_good_push",
        "p_truth_now_rank", "predicted_benefit_q25_rank", "p_good_push_rank",
        "message_fact", "observation_index",
    ]
    for column in signal_columns:
        if column not in selected:
            selected[column] = np.nan
    return metrics, selected[signal_columns], grid


def run_joint_policy_experiment(
    root: Path,
    horizons: tuple[int, ...],
    outer_start_year: int,
    workers: int,
) -> None:
    """Compare two-head AND selection with a direct joint classifier."""
    output_dir = root / "data/processed/ml_joint_policy"
    output_dir.mkdir(parents=True, exist_ok=True)
    prices = load_ml_prices(latest_default_input())
    features = build_features(prices)
    data_as_of = pd.Timestamp(prices["date"].max())
    folds = [
        fold for fold in outer_folds(data_as_of) if fold[0].year >= outer_start_year
    ]
    if not folds:
        raise ValueError(f"No outer folds from {outer_start_year}")

    datasets: dict[int, tuple[pd.DataFrame, list[str]]] = {}
    for horizon in horizons:
        print(f"Preparing joint-policy h={horizon}", flush=True)
        data = features.merge(
            build_labels(prices, horizon), on=["date", "currency"], how="inner"
        )
        data["good_push"] = (
            data["truth_now"].eq(1) & data["benefit_bps"].gt(0)
        ).astype(float)
        datasets[horizon] = (data, feature_columns(data))

    metric_parts: list[pd.DataFrame] = []
    signal_parts: list[pd.DataFrame] = []
    grid_parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_joint_policy_task,
                horizon,
                architecture,
                datasets[horizon][0],
                datasets[horizon][1],
                fold_start,
                fold_end,
            ): (horizon, architecture, fold_start, fold_end)
            for horizon in horizons
            for architecture in JOINT_ARCHITECTURES
            for fold_start, fold_end in folds
        }
        for future in as_completed(futures):
            horizon, architecture, fold_start, fold_end = futures[future]
            metrics, signals, grid = future.result()
            metric_parts.append(metrics)
            signal_parts.append(signals)
            grid_parts.append(grid)
            print(
                f"Finished {architecture}, h={horizon}, outer={fold_start.date()}",
                flush=True,
            )

    metrics = pd.concat(metric_parts, ignore_index=True).sort_values(
        ["architecture", "horizon", "fold_start", "currency"]
    )
    signals = pd.concat(signal_parts, ignore_index=True).sort_values(
        ["architecture", "horizon", "date", "currency"]
    )
    policy_grid = pd.concat(grid_parts, ignore_index=True).sort_values(
        ["architecture", "horizon", "outer_start", "chosen"],
        ascending=[True, True, True, False],
    )
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    signals.to_csv(output_dir / "signals_backtest.csv.gz", index=False)
    policy_grid.to_csv(output_dir / "policy_grid.csv", index=False)

    stability = metrics.groupby(
        ["architecture", "horizon", "currency"], as_index=False
    ).agg(
        folds=("fold_start", "nunique"),
        signals=("signal_count", "sum"),
        empty_folds=("signal_count", lambda values: values.eq(0).sum()),
        median_truth_hit_rate=("truth_now_hit_rate", "median"),
        median_truth_random=("truth_now_random", "median"),
        median_truth_lift=("truth_now_lift", "median"),
        min_truth_lift=("truth_now_lift", "min"),
        median_benefit_bps=("benefit_bps", "median"),
        min_benefit_bps=("benefit_bps", "min"),
        significant_benefit_share=("benefit_significant_5pct", "mean"),
        median_frequency=("signals_per_week", "median"),
    )
    stability.to_csv(output_dir / "stability.csv", index=False)

    summary = metrics.groupby(["architecture", "horizon"], as_index=False).agg(
        fold_currency_cells=("currency", "size"),
        signals=("signal_count", "sum"),
        empty_cells=("signal_count", lambda values: values.eq(0).sum()),
        median_truth_hit_rate=("truth_now_hit_rate", "median"),
        median_truth_random=("truth_now_random", "median"),
        median_truth_lift=("truth_now_lift", "median"),
        q25_truth_lift=("truth_now_lift", lambda values: values.quantile(0.25)),
        truth_lift_115_share=("truth_now_lift", lambda values: values.ge(1.15).mean()),
        truth_lift_130_share=("truth_now_lift", lambda values: values.ge(1.30).mean()),
        median_benefit_bps=("benefit_bps", "median"),
        nonnegative_benefit_share=("benefit_bps", lambda values: values.ge(0).mean()),
        significant_benefit_share=("benefit_significant_5pct", "mean"),
        median_frequency=("signals_per_week", "median"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    metadata = {
        "experiment": "joint truth_now and economic-benefit policies",
        "architectures": {
            "two_head": "binary truth_now AND predicted 25th-quantile benefit",
            "joint_classifier": "binary truth_now AND benefit_bps>0 label",
        },
        "horizons": list(horizons),
        "outer_start_year": outer_start_year,
        "outer_folds": [[str(a.date()), str(b.date())] for a, b in folds],
        "inner_oof_years": INNER_OOF_YEARS,
        "score_normalization": (
            f"causal trailing percentile per currency, lookback={SCORE_PERCENTILE_LOOKBACK}"
        ),
        "policy": "one shared percentile policy across all currencies",
        "frequency": "reported only; never used as a business constraint",
        "benefit_inference": "calendar-month percentile block bootstrap",
        "data_as_of": str(data_as_of.date()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Saved joint-policy experiment to {output_dir}")


def make_tuned_good_push_model(
    config_name: str,
    model_configs: dict[str, dict[str, object]] = TUNING_MODEL_CONFIGS,
) -> HistGradientBoostingClassifier:
    """Create one predeclared capacity/history candidate for nested tuning."""
    if config_name not in model_configs:
        raise ValueError(f"Unknown tuning model config: {config_name}")
    config = model_configs[config_name]
    return HistGradientBoostingClassifier(
        class_weight=config.get("class_weight", "balanced"),
        learning_rate=float(config["learning_rate"]),
        max_iter=int(config["max_iter"]),
        max_leaf_nodes=int(config["max_leaf_nodes"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        l2_regularization=float(config["l2_regularization"]),
        max_features=float(config.get("max_features", 1.0)),
        random_state=47,
    )


def build_tuning_inner_oof(
    data: pd.DataFrame,
    columns: list[str],
    outer_start: pd.Timestamp,
    config_name: str,
    model_configs: dict[str, dict[str, object]] = TUNING_MODEL_CONFIGS,
    score_lookbacks: tuple[int, ...] = TUNING_SCORE_LOOKBACKS,
    cache_file: Path | None = None,
) -> pd.DataFrame:
    """Generate reusable inner OOF scores and causal ranks for each lookback."""
    if cache_file is not None and cache_file.exists():
        print(f"Loading checkpoint {cache_file.name}", flush=True)
        return pd.read_pickle(cache_file, compression="gzip")

    config = model_configs[config_name]
    parts: list[pd.DataFrame] = []
    for year in range(outer_start.year - INNER_OOF_YEARS, outer_start.year):
        validation_start = pd.Timestamp(f"{year}-01-01")
        validation_end = pd.Timestamp(f"{year + 1}-01-01")
        train = mature_training_rows(
            data,
            model_train_start(str(config["history"]), validation_start),
            validation_start,
        )
        validation = data.loc[
            data["date"].ge(validation_start)
            & data["date"].lt(validation_end)
            & data["label_available_date"].lt(outer_start)
            & data["valid"]
        ]
        reference = data.loc[
            data["date"].ge(validation_start - pd.DateOffset(years=1))
            & data["date"].lt(validation_start)
        ]
        if len(train) < 500 or validation.empty or reference.empty:
            continue
        model = make_tuned_good_push_model(config_name, model_configs)
        model.fit(train[columns], train["good_push"])
        reference_scored = reference.copy()
        validation_scored = validation.copy()
        reference_scored["p_good_push"] = model.predict_proba(reference[columns])[:, 1]
        validation_scored["p_good_push"] = model.predict_proba(validation[columns])[:, 1]
        ranked = validation_scored.copy()
        for lookback in score_lookbacks:
            normalized = causal_score_percentiles(
                reference_scored,
                validation_scored,
                ("p_good_push",),
                lookback=lookback,
            )
            ranked[f"p_good_push_rank_{lookback}"] = normalized["p_good_push_rank"]
        ranked["inner_fold_start"] = validation_start
        parts.append(ranked)
    if not parts:
        raise RuntimeError(
            f"No tuning OOF data for config={config_name}, outer={outer_start.date()}"
        )
    result = pd.concat(parts, ignore_index=True)
    keep = [
        "date", "currency", "truth_now", "local_min", "benefit_bps",
        "inner_fold_start", "p_good_push",
        *[f"p_good_push_rank_{lookback}" for lookback in score_lookbacks],
    ]
    result = result[[column for column in keep if column in result.columns]]
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_suffix(cache_file.suffix + ".tmp")
        result.to_pickle(temporary, compression="gzip")
        temporary.replace(cache_file)
    return result


def tune_horizon_config_on_inner_oof(
    inner_oof: pd.DataFrame,
    horizon: int,
    config_name: str,
    lookback: int,
    policy_quantiles: tuple[float, ...] = JOINT_POLICY_QUANTILES,
) -> pd.DataFrame:
    """Evaluate all shared joint-score quantiles for one fixed candidate."""
    ranked = inner_oof.copy()
    ranked["p_good_push_rank"] = ranked[f"p_good_push_rank_{lookback}"]
    _, grid = select_joint_global_policy(
        ranked, "joint_classifier", policy_quantiles=policy_quantiles
    )
    grid["horizon"] = horizon
    grid["model_config"] = config_name
    grid["score_lookback"] = lookback
    return grid


def tuning_sort_columns() -> tuple[list[str], list[bool]]:
    return (
        [
            "enough_coverage", "passing_currencies", "both_cell_share",
            "q25_truth_lift", "q25_benefit_bps", "reliable_cells", "signals",
        ],
        [False, False, False, False, False, False, False],
    )


def refit_tuned_horizon_policy(
    data: pd.DataFrame,
    columns: list[str],
    horizon: int,
    choice: pd.Series,
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
    model_configs: dict[str, dict[str, object]] = TUNING_MODEL_CONFIGS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_name = str(choice["model_config"])
    config = model_configs[config_name]
    lookback = int(choice["score_lookback"])
    threshold = float(choice["joint_quantile"])
    train = mature_training_rows(
        data,
        model_train_start(str(config["history"]), fold_start),
        fold_start,
    )
    model = make_tuned_good_push_model(config_name, model_configs)
    model.fit(train[columns], train["good_push"])
    reference = data.loc[
        data["date"].ge(fold_start - pd.DateOffset(years=1))
        & data["date"].lt(fold_start)
    ].copy()
    test = data.loc[
        data["date"].ge(fold_start)
        & data["date"].lt(fold_end)
        & data["valid"]
    ].copy()
    reference["p_good_push"] = model.predict_proba(reference[columns])[:, 1]
    test["p_good_push"] = model.predict_proba(test[columns])[:, 1]
    ranked = causal_score_percentiles(
        reference, test, ("p_good_push",), lookback=lookback
    )
    selected = ranked.loc[ranked["p_good_push_rank"].ge(threshold)].copy()
    selected = add_message_facts(selected)
    selected["horizon"] = horizon
    selected["outer_start"] = fold_start
    selected["model_config"] = config_name
    selected["score_lookback"] = lookback
    selected["joint_quantile"] = threshold

    metrics = evaluate_fold(selected, test, horizon, fold_start, fold_end)
    metrics["model_config"] = config_name
    metrics["score_lookback"] = lookback
    metrics["joint_quantile"] = threshold
    return metrics, selected


def run_horizon_tuning_outer_task(
    datasets: dict[int, tuple[pd.DataFrame, list[str]]],
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
    model_configs: dict[str, dict[str, object]] = TUNING_MODEL_CONFIGS,
    score_lookbacks: tuple[int, ...] = TUNING_SCORE_LOOKBACKS,
    policy_quantiles: tuple[float, ...] = JOINT_POLICY_QUANTILES,
    checkpoint_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nested selection of horizon, model configuration, rank window and cutoff."""
    grid_parts: list[pd.DataFrame] = []
    total_inner_blocks = len(datasets) * len(model_configs)
    completed_inner_blocks = 0
    for horizon, (data, columns) in datasets.items():
        for config_name in model_configs:
            block_started = time.monotonic()
            cache_file = None
            if checkpoint_dir is not None:
                cache_file = checkpoint_dir / (
                    f"inner_outer{fold_start.year}_h{horizon}_{config_name}.pkl.gz"
                )
            inner_oof = build_tuning_inner_oof(
                data,
                columns,
                fold_start,
                config_name,
                model_configs=model_configs,
                score_lookbacks=score_lookbacks,
                cache_file=cache_file,
            )
            for lookback in score_lookbacks:
                grid_parts.append(
                    tune_horizon_config_on_inner_oof(
                        inner_oof,
                        horizon,
                        config_name,
                        lookback,
                        policy_quantiles=policy_quantiles,
                    )
                )
            completed_inner_blocks += 1
            print(
                f"[outer={fold_start.year}] inner blocks "
                f"{completed_inner_blocks}/{total_inner_blocks}: "
                f"h={horizon}, config={config_name}, "
                f"elapsed={((time.monotonic() - block_started) / 60):.1f} min",
                flush=True,
            )
    full_grid = pd.concat(grid_parts, ignore_index=True)
    full_grid["outer_start"] = fold_start
    full_grid["chosen_within_candidate"] = full_grid["chosen"]

    sort_columns, ascending = tuning_sort_columns()
    candidate_winners = (
        full_grid.sort_values(sort_columns, ascending=ascending, na_position="last")
        .groupby(["horizon", "model_config", "score_lookback"], group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
    horizon_choices = (
        candidate_winners.sort_values(sort_columns, ascending=ascending, na_position="last")
        .groupby("horizon", group_keys=False)
        .head(1)
        .sort_values(sort_columns, ascending=ascending, na_position="last")
        .reset_index(drop=True)
    )
    horizon_choices["official_horizon"] = horizon_choices["horizon"].isin(HORIZONS)
    horizon_choices["exploratory_chosen"] = False
    horizon_choices.loc[0, "exploratory_chosen"] = True
    horizon_choices["overall_chosen"] = False
    official_rows = horizon_choices.index[horizon_choices["official_horizon"]]
    if len(official_rows):
        horizon_choices.loc[official_rows[0], "overall_chosen"] = True
    horizon_choices["outer_start"] = fold_start

    metric_parts: list[pd.DataFrame] = []
    signal_parts: list[pd.DataFrame] = []
    for _, choice in horizon_choices.iterrows():
        horizon = int(choice["horizon"])
        data, columns = datasets[horizon]
        metrics, signals = refit_tuned_horizon_policy(
            data,
            columns,
            horizon,
            choice,
            fold_start,
            fold_end,
            model_configs=model_configs,
        )
        is_overall = bool(choice["overall_chosen"])
        metrics["overall_chosen"] = is_overall
        signals["overall_chosen"] = is_overall
        metric_parts.append(metrics)
        signal_parts.append(signals)
        print(
            f"[outer={fold_start.year}] refit h={horizon}: "
            f"config={choice['model_config']}, lookback={int(choice['score_lookback'])}, "
            f"q={float(choice['joint_quantile']):.2f}",
            flush=True,
        )

    return (
        pd.concat(metric_parts, ignore_index=True),
        pd.concat(signal_parts, ignore_index=True),
        full_grid,
        horizon_choices,
    )


def run_horizon_tuning_experiment(
    root: Path,
    horizons: tuple[int, ...],
    outer_start_year: int,
    workers: int,
    *,
    model_configs: dict[str, dict[str, object]] = TUNING_MODEL_CONFIGS,
    score_lookbacks: tuple[int, ...] = TUNING_SCORE_LOOKBACKS,
    policy_quantiles: tuple[float, ...] = JOINT_POLICY_QUANTILES,
    output_subdir: str = "ml_horizon_tuning",
    resume: bool = False,
) -> None:
    """Nested refinement of h and training parameters for the joint classifier."""
    output_dir = root / "data/processed" / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = latest_default_input()
    prices = load_ml_prices(input_path)
    features = build_features(prices)
    data_as_of = pd.Timestamp(prices["date"].max())
    folds = [
        fold for fold in outer_folds(data_as_of) if fold[0].year >= outer_start_year
    ]
    if not folds:
        raise ValueError(f"No outer folds from {outer_start_year}")

    run_config = {
        "horizons": list(horizons),
        "model_configs": model_configs,
        "score_lookbacks": list(score_lookbacks),
        "score_quantiles": list(policy_quantiles),
        "inner_oof_years": INNER_OOF_YEARS,
        "outer_start_year": outer_start_year,
        "outer_folds": [[str(a.date()), str(b.date())] for a, b in folds],
        "data_as_of": str(data_as_of.date()),
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "code_sha256": file_sha256(Path(__file__).resolve()),
        "feature_columns": feature_columns(features),
    }
    run_config_path = output_dir / "run_config.json"
    if resume and run_config_path.exists():
        previous = json.loads(run_config_path.read_text())
        if previous != run_config:
            raise RuntimeError(
                f"Checkpoint configuration differs from this run: {run_config_path}"
            )
    run_config_path.write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n"
    )
    checkpoint_dir = output_dir / "checkpoints" if resume else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[int, tuple[pd.DataFrame, list[str]]] = {}
    for horizon in horizons:
        print(f"Preparing horizon tuning h={horizon}", flush=True)
        data = features.merge(
            build_labels(prices, horizon), on=["date", "currency"], how="inner"
        )
        data["good_push"] = (
            data["truth_now"].eq(1) & data["benefit_bps"].gt(0)
        ).astype(float)
        datasets[horizon] = (data, feature_columns(data))

    metric_parts: list[pd.DataFrame] = []
    signal_parts: list[pd.DataFrame] = []
    grid_parts: list[pd.DataFrame] = []
    choice_parts: list[pd.DataFrame] = []
    pending_folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for fold_start, fold_end in folds:
        prefix = checkpoint_dir / f"outer_{fold_start.year}" if checkpoint_dir else None
        paths = [] if prefix is None else [
            prefix.with_name(prefix.name + "_metrics.csv"),
            prefix.with_name(prefix.name + "_signals.csv.gz"),
            prefix.with_name(prefix.name + "_grid.csv.gz"),
            prefix.with_name(prefix.name + "_choices.csv"),
        ]
        if resume and all(path.exists() for path in paths):
            metric_parts.append(pd.read_csv(paths[0], parse_dates=["fold_start", "fold_end"]))
            signal_parts.append(pd.read_csv(paths[1], parse_dates=["date", "outer_start"]))
            grid_parts.append(pd.read_csv(paths[2], parse_dates=["outer_start"]))
            choice_parts.append(pd.read_csv(paths[3], parse_dates=["outer_start"]))
            print(f"Resumed completed outer fold {fold_start.year}", flush=True)
        else:
            pending_folds.append((fold_start, fold_end))

    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending_folds)))) as executor:
        futures = {
            executor.submit(
                run_horizon_tuning_outer_task,
                datasets,
                fold_start,
                fold_end,
                model_configs,
                score_lookbacks,
                policy_quantiles,
                checkpoint_dir,
            ): (fold_start, fold_end)
            for fold_start, fold_end in pending_folds
        }
        for future in as_completed(futures):
            fold_start, _ = futures[future]
            metrics, signals, grid, choices = future.result()
            metric_parts.append(metrics)
            signal_parts.append(signals)
            grid_parts.append(grid)
            choice_parts.append(choices)
            if checkpoint_dir is not None:
                prefix = checkpoint_dir / f"outer_{fold_start.year}"
                metrics.to_csv(prefix.with_name(prefix.name + "_metrics.csv"), index=False)
                signals.to_csv(prefix.with_name(prefix.name + "_signals.csv.gz"), index=False)
                grid.to_csv(prefix.with_name(prefix.name + "_grid.csv.gz"), index=False)
                choices.to_csv(prefix.with_name(prefix.name + "_choices.csv"), index=False)
            print(f"Finished horizon tuning outer={fold_start.date()}", flush=True)

    metrics = pd.concat(metric_parts, ignore_index=True).sort_values(
        ["horizon", "fold_start", "currency"]
    )
    signals = pd.concat(signal_parts, ignore_index=True).sort_values(
        ["horizon", "date", "currency"]
    )
    metrics["official_horizon"] = metrics["horizon"].isin(HORIZONS)
    signals["official_horizon"] = signals["horizon"].isin(HORIZONS)
    grid = pd.concat(grid_parts, ignore_index=True).sort_values(
        ["outer_start", "horizon", "model_config", "score_lookback", "joint_quantile"]
    )
    choices = pd.concat(choice_parts, ignore_index=True).sort_values(
        ["outer_start", "overall_chosen", "horizon"], ascending=[True, False, True]
    )
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    signals.to_csv(output_dir / "signals_backtest.csv.gz", index=False)
    grid.to_csv(output_dir / "tuning_grid.csv.gz", index=False)
    choices.to_csv(output_dir / "outer_choices.csv", index=False)

    summary = metrics.groupby("horizon", as_index=False).agg(
        folds=("fold_start", "nunique"),
        signals=("signal_count", "sum"),
        empty_cells=("signal_count", lambda values: values.eq(0).sum()),
        median_truth_hit_rate=("truth_now_hit_rate", "median"),
        median_truth_random=("truth_now_random", "median"),
        median_truth_lift=("truth_now_lift", "median"),
        q25_truth_lift=("truth_now_lift", lambda values: values.quantile(0.25)),
        truth_lift_115_share=("truth_now_lift", lambda values: values.ge(1.15).mean()),
        truth_lift_130_share=("truth_now_lift", lambda values: values.ge(1.30).mean()),
        median_benefit_bps=("benefit_bps", "median"),
        nonnegative_benefit_share=("benefit_bps", lambda values: values.ge(0).mean()),
        significant_benefit_share=("benefit_significant_5pct", "mean"),
        median_frequency=("signals_per_week", "median"),
        chosen_outer_folds=(
            "overall_chosen",
            lambda values: values.sum() / len(TARGET_CURRENCIES),
        ),
    )
    summary["official_horizon"] = summary["horizon"].isin(HORIZONS)
    summary.to_csv(output_dir / "summary_by_horizon.csv", index=False)
    final_metrics = metrics.loc[metrics["overall_chosen"]].copy()
    final_metrics.to_csv(output_dir / "final_selected_metrics.csv", index=False)

    # Operational estimates on nested trailing OOT windows.  Each constituent
    # date remains a genuine yearly outer prediction; only evaluation is pooled.
    rolling_parts: list[pd.DataFrame] = []
    last_outer_year = max(start.year for start, _ in folds)
    for window_start_year in range(outer_start_year, last_outer_year + 1):
        window_start = pd.Timestamp(f"{window_start_year}-01-01")
        window_end = data_as_of + timedelta(days=1)
        for horizon, (data, _) in datasets.items():
            window_signals = signals.loc[
                signals["horizon"].eq(horizon)
                & signals["date"].ge(window_start)
                & signals["date"].lt(window_end)
            ]
            window_pool = data.loc[
                data["date"].ge(window_start) & data["date"].lt(window_end)
            ]
            evaluated = evaluate_fold(
                window_signals, window_pool, horizon, window_start, window_end
            )
            evaluated.insert(0, "window", f"{window_start_year}-{data_as_of.year}")
            rolling_parts.append(evaluated)
    rolling_metrics = pd.concat(rolling_parts, ignore_index=True)
    rolling_metrics.to_csv(output_dir / "rolling_window_metrics.csv", index=False)
    rolling_summary = rolling_metrics.groupby(["window", "horizon"], as_index=False).agg(
        signals=("signal_count", "sum"),
        empty_currencies=("signal_count", lambda values: values.eq(0).sum()),
        median_truth_hit_rate=("truth_now_hit_rate", "median"),
        median_truth_random=("truth_now_random", "median"),
        median_truth_lift=("truth_now_lift", "median"),
        min_truth_lift=("truth_now_lift", "min"),
        median_benefit_bps=("benefit_bps", "median"),
        min_benefit_bps=("benefit_bps", "min"),
        significant_benefit_share=("benefit_significant_5pct", "mean"),
    )
    rolling_summary["official_horizon"] = rolling_summary["horizon"].isin(HORIZONS)
    rolling_summary.to_csv(output_dir / "rolling_window_summary.csv", index=False)

    metadata = {
        "experiment": "nested horizon and training-parameter tuning for joint_classifier",
        "horizons": list(horizons),
        "model_configs": model_configs,
        "score_lookbacks": list(score_lookbacks),
        "score_quantiles": list(policy_quantiles),
        "inner_oof_years": INNER_OOF_YEARS,
        "outer_start_year": outer_start_year,
        "outer_folds": [[str(a.date()), str(b.date())] for a, b in folds],
        "selection": "h, model config, score lookback and quantile selected only on inner OOF",
        "rolling_windows": (
            "pooled yearly outer predictions ending at data_as_of; random-day baseline "
            "is recomputed within the same currency and pooled window"
        ),
        "frequency": "reported only; not used for selection",
        "data_as_of": str(data_as_of.date()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Saved horizon-tuning experiment to {output_dir}")


def score_cross_h_fold(
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    columns: list[str],
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
    choice: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Refit the frozen h=11 policy and evaluate its dates with official h=10.

    Policy/model parameters were selected by the earlier h=11 experiment, as
    requested. The h=10 labels are never used to select or fit the policy.
    """
    config_name = str(choice["model_config"])
    config = NIGHTLY_MODEL_CONFIGS[config_name]
    lookback = int(choice["score_lookback"])
    quantile = float(choice["joint_quantile"])

    # A causal OOF reference year avoids normalising OOS outer scores against
    # scores from rows on which the same model was fitted.
    reference_start = fold_start - pd.DateOffset(years=1)
    reference_train = mature_training_rows(
        train_data,
        model_train_start(str(config["history"]), reference_start),
        reference_start,
    )
    reference = train_data.loc[
        train_data["date"].ge(reference_start)
        & train_data["date"].lt(fold_start)
        & train_data["valid"]
    ].copy()
    if len(reference_train) < 500 or reference.empty:
        raise RuntimeError(f"Insufficient causal reference data for outer {fold_start.year}")
    reference_model = make_tuned_good_push_model(config_name, NIGHTLY_MODEL_CONFIGS)
    reference_model.fit(reference_train[columns], reference_train["good_push"])
    reference["p_good_push"] = reference_model.predict_proba(reference[columns])[:, 1]

    train = mature_training_rows(
        train_data,
        model_train_start(str(config["history"]), fold_start),
        fold_start,
    )
    model = make_tuned_good_push_model(config_name, NIGHTLY_MODEL_CONFIGS)
    model.fit(train[columns], train["good_push"])

    # The train target needs eleven future observations. On a partial final
    # fold this is the strictest maturity boundary and therefore the common
    # evaluation boundary for the signal stream and random-day pool.
    valid_fold = train_data.loc[
        train_data["date"].ge(fold_start)
        & train_data["date"].lt(fold_end)
        & train_data["valid"]
    ]
    last_by_currency = valid_fold.groupby("currency")["date"].max()
    if len(last_by_currency) != len(TARGET_CURRENCIES):
        raise RuntimeError(f"Incomplete target-currency fold {fold_start.year}")
    effective_end = min(fold_end, pd.Timestamp(last_by_currency.min()) + timedelta(days=1))

    test = train_data.loc[
        train_data["date"].ge(fold_start)
        & train_data["date"].lt(effective_end)
        & train_data["valid"]
    ].copy()
    test["p_good_push"] = model.predict_proba(test[columns])[:, 1]
    ranked = causal_score_percentiles(
        reference, test, ("p_good_push",), lookback=lookback
    )
    selected = ranked.loc[ranked["p_good_push_rank"].ge(quantile)].copy()

    # Replace all h=11 outcomes by independent h=10 evaluation outcomes.
    outcome_columns = [
        "truth_now", "local_min", "benefit_bps", "future_regret_bps",
        "label_available_date", "valid",
    ]
    selected = selected.drop(columns=outcome_columns, errors="ignore").merge(
        eval_data[["date", "currency", *outcome_columns]],
        on=["date", "currency"],
        how="inner",
        validate="one_to_one",
    )
    selected = selected.loc[selected["valid"]].copy()
    selected = add_message_facts(selected)
    selected["train_horizon"] = CROSS_H_TRAIN_HORIZON
    selected["evaluation_horizon"] = CROSS_H_EVAL_HORIZON
    selected["outer_start"] = fold_start
    selected["model_config"] = config_name
    selected["score_lookback"] = lookback
    selected["joint_quantile"] = quantile

    pool = eval_data.loc[
        eval_data["date"].ge(fold_start)
        & eval_data["date"].lt(effective_end)
        & eval_data["valid"]
    ]
    metrics = evaluate_fold(
        selected, pool, CROSS_H_EVAL_HORIZON, fold_start, effective_end
    )
    metrics["train_horizon"] = CROSS_H_TRAIN_HORIZON
    metrics["evaluation_horizon"] = CROSS_H_EVAL_HORIZON
    metrics["model_config"] = config_name
    metrics["score_lookback"] = lookback
    metrics["joint_quantile"] = quantile
    metrics["effective_evaluation_end"] = effective_end
    return metrics, selected, effective_end


def run_cross_h_reproduction(root: Path, outer_start_year: int = 2023) -> None:
    """Retrain frozen best h=11 policies and score their dates at official h=10."""
    output_dir = root / "data/processed/ml_cross_h_11_to_10"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = latest_default_input()
    prices = load_ml_prices(input_path)
    features = build_features(prices)
    train_data = features.merge(
        build_labels(prices, CROSS_H_TRAIN_HORIZON),
        on=["date", "currency"],
        how="inner",
    )
    train_data["good_push"] = (
        train_data["truth_now"].eq(1) & train_data["benefit_bps"].gt(0)
    ).astype(float)
    eval_data = features[["date", "currency"]].merge(
        build_labels(prices, CROSS_H_EVAL_HORIZON),
        on=["date", "currency"],
        how="inner",
    )
    columns = feature_columns(train_data)
    context_columns = [
        column for column in columns
        if column.startswith(("usd_", "eur_", "cny_", "context_", "own_minus_"))
    ]
    if not context_columns:
        raise RuntimeError("FX context features are absent from the ML matrix")

    data_as_of = pd.Timestamp(prices["date"].max())
    folds = [fold for fold in outer_folds(data_as_of) if fold[0].year >= outer_start_year]
    metric_parts: list[pd.DataFrame] = []
    signal_parts: list[pd.DataFrame] = []
    choice_rows: list[dict[str, object]] = []
    for fold_start, fold_end in folds:
        if fold_start.year not in CROSS_H_LEGACY_CHOICES:
            raise ValueError(f"No frozen h=11 policy for outer {fold_start.year}")
        choice = CROSS_H_LEGACY_CHOICES[fold_start.year]
        print(
            f"Cross-h outer={fold_start.year}: {choice['model_config']}, "
            f"lookback={choice['score_lookback']}, q={choice['joint_quantile']}",
            flush=True,
        )
        metrics, signals, effective_end = score_cross_h_fold(
            train_data, eval_data, columns, fold_start, fold_end, choice
        )
        metric_parts.append(metrics)
        signal_parts.append(signals)
        choice_rows.append(
            {
                "outer_start": fold_start,
                "effective_evaluation_end": effective_end,
                "train_horizon": CROSS_H_TRAIN_HORIZON,
                "evaluation_horizon": CROSS_H_EVAL_HORIZON,
                **choice,
            }
        )

    metrics = pd.concat(metric_parts, ignore_index=True).sort_values(
        ["fold_start", "currency"]
    )
    signals = pd.concat(signal_parts, ignore_index=True).sort_values(
        ["date", "currency"]
    )
    choices = pd.DataFrame(choice_rows)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    signals.to_csv(output_dir / "signals_backtest.csv.gz", index=False)
    choices.to_csv(output_dir / "outer_choices.csv", index=False)

    rolling_parts: list[pd.DataFrame] = []
    last_effective_end = pd.Timestamp(choices["effective_evaluation_end"].max())
    for start_year in range(outer_start_year, data_as_of.year + 1):
        window_start = pd.Timestamp(f"{start_year}-01-01")
        window_signals = signals.loc[
            signals["date"].ge(window_start) & signals["date"].lt(last_effective_end)
        ]
        window_pool = eval_data.loc[
            eval_data["date"].ge(window_start)
            & eval_data["date"].lt(last_effective_end)
            & eval_data["valid"]
        ]
        evaluated = evaluate_fold(
            window_signals,
            window_pool,
            CROSS_H_EVAL_HORIZON,
            window_start,
            last_effective_end,
        )
        evaluated.insert(0, "window", f"{start_year}-{data_as_of.year}")
        rolling_parts.append(evaluated)
    rolling = pd.concat(rolling_parts, ignore_index=True)
    rolling.to_csv(output_dir / "rolling_window_metrics.csv", index=False)

    final_metrics, final_summary, common_end = evaluate_signal_stream_across_horizons(
        signals,
        prices,
        pd.Timestamp("2025-01-01"),
        HORIZONS,
    )
    final_metrics.to_csv(output_dir / "final_2025_2026_by_horizon.csv", index=False)
    final_summary.to_csv(
        output_dir / "final_2025_2026_summary_by_horizon.csv", index=False
    )

    metadata = {
        "experiment": "frozen h=11 policy retrained and independently evaluated at official h=10",
        "selection_rule": "model/lookback/quantile are frozen from h=11 inner-OOF selection; h=10 is evaluation only",
        "train_horizon": CROSS_H_TRAIN_HORIZON,
        "evaluation_horizon": CROSS_H_EVAL_HORIZON,
        "legacy_choices": CROSS_H_LEGACY_CHOICES,
        "reference_scores": "causal OOF previous-year scores; reference rows are not used to fit their scoring model",
        "random_baseline": "exact truth mean over same currency and evaluation period",
        "benefit_inference": "calendar-month basic block-bootstrap CI and null-centred one-sided test",
        "frequency": "reported only; not used for selection",
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "code_sha256": file_sha256(Path(__file__).resolve()),
        "feature_columns": columns,
        "context_feature_columns": context_columns,
        "data_as_of": str(data_as_of.date()),
        "final_reporting_window": ["2025-01-01", str((common_end - timedelta(days=1)).date())],
        "final_horizon_profile": (
            "the same frozen OOT signal dates are evaluated at h=1/3/5/10/20; "
            "all horizons share the h=20 maturity cutoff"
        ),
        "development_warning": "2023-2026 was inspected previously and is not a pristine final holdout",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Saved cross-h reproduction to {output_dir}", flush=True)


def evaluate_signal_stream_across_horizons(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    window_start: pd.Timestamp,
    horizons: tuple[int, ...] = HORIZONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Evaluate one frozen OOT signal stream on several outcome horizons.

    The latest evaluation date is shared by every horizon and currency.  It is
    determined by the longest requested label, so differences across h cannot
    be caused by different signal dates or by an immature right edge.
    """
    if not horizons:
        raise ValueError("At least one evaluation horizon is required")
    window_start = pd.Timestamp(window_start)
    labels_by_horizon = {h: build_labels(prices, h) for h in horizons}
    maturity_limits: list[pd.Timestamp] = []
    for labels in labels_by_horizon.values():
        valid = labels.loc[labels["valid"]]
        last_by_currency = valid.groupby("currency")["date"].max()
        if len(last_by_currency) != len(TARGET_CURRENCIES):
            raise RuntimeError("Incomplete target-currency labels for horizon profile")
        maturity_limits.append(pd.Timestamp(last_by_currency.min()))
    common_end = min(maturity_limits) + timedelta(days=1)
    if common_end <= window_start:
        raise RuntimeError("No mature observations in the requested final window")

    outcome_columns = [
        "truth_now", "local_min", "benefit_bps", "future_regret_bps",
        "label_available_date", "valid",
    ]
    signal_keys = signals.loc[
        signals["date"].ge(window_start) & signals["date"].lt(common_end)
    ].drop(columns=outcome_columns, errors="ignore")
    if signal_keys.duplicated(["date", "currency"]).any():
        raise ValueError("Frozen signal stream contains duplicate currency/date rows")

    metric_parts: list[pd.DataFrame] = []
    for horizon, labels in labels_by_horizon.items():
        selected = signal_keys.merge(
            labels[["date", "currency", *outcome_columns]],
            on=["date", "currency"],
            how="inner",
            validate="one_to_one",
        )
        selected = selected.loc[selected["valid"]].copy()
        pool = labels.loc[
            labels["date"].ge(window_start)
            & labels["date"].lt(common_end)
            & labels["valid"]
        ]
        evaluated = evaluate_fold(selected, pool, horizon, window_start, common_end)
        evaluated.insert(0, "window", f"{window_start.year}-{common_end.year}")
        metric_parts.append(evaluated)

    metrics = pd.concat(metric_parts, ignore_index=True).sort_values(
        ["horizon", "currency"]
    )
    counts_per_horizon = metrics.groupby("horizon")["signal_count"].sum()
    if counts_per_horizon.nunique() != 1:
        raise RuntimeError("Horizon profile did not preserve the same signal dates")
    summary = metrics.groupby("horizon", as_index=False).agg(
        currencies=("currency", "nunique"),
        signals=("signal_count", "sum"),
        median_truth_hit_rate=("truth_now_hit_rate", "median"),
        median_truth_random=("truth_now_random", "median"),
        mean_truth_lift=("truth_now_lift", "mean"),
        median_truth_lift=("truth_now_lift", "median"),
        min_truth_lift=("truth_now_lift", "min"),
        median_local_min_lift=("local_min_lift", "median"),
        min_local_min_lift=("local_min_lift", "min"),
        mean_benefit_bps=("benefit_bps", "mean"),
        median_benefit_bps=("benefit_bps", "median"),
        min_benefit_bps=("benefit_bps", "min"),
        all_benefits_significant=("benefit_significant_5pct", "all"),
        significant_benefit_share=("benefit_significant_5pct", "mean"),
        median_signals_per_week=("signals_per_week", "median"),
    )
    summary.insert(0, "window", f"{window_start.year}-{common_end.year}")
    summary.insert(2, "evaluation_end_exclusive", common_end)
    return metrics, summary, common_end


def match_delayed_hard_confirmations(
    ml_signals: pd.DataFrame,
    hard_signals: pd.DataFrame,
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_wait_observations: int = 5,
) -> pd.DataFrame:
    """Pair each sent ML signal with the first unused later hard confirmation.

    Matching is chronological within a corridor and uses publication indices,
    not calendar days. A hard event can confirm at most one ML episode. Future
    rows are used only for retrospective evaluation of the waiting policy.
    """
    if max_wait_observations < 0:
        raise ValueError("max_wait_observations must be non-negative")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    market = prices.loc[prices["currency"].isin(TARGET_CURRENCIES), [
        "date", "currency", "rub_per_unit"
    ]].sort_values(["currency", "date"]).copy()
    market["market_observation_index"] = market.groupby("currency").cumcount()
    lookup = market.set_index(["currency", "date"])[
        ["rub_per_unit", "market_observation_index"]
    ]

    ml = ml_signals.loc[
        ml_signals["date"].ge(start) & ml_signals["date"].lt(end)
    ].drop_duplicates(["currency", "date"]).sort_values(["currency", "date"])
    hard = hard_signals.loc[
        hard_signals["date"].ge(start) & hard_signals["date"].lt(end)
    ].copy()
    hard = (
        hard.groupby(["currency", "date"], as_index=False)
        .agg(confirmation_families=("signal_family", lambda x: "+".join(sorted(set(x)))))
        .sort_values(["currency", "date"])
    )
    rows: list[dict[str, object]] = []
    for currency in TARGET_CURRENCIES:
        fast = ml.loc[ml["currency"].eq(currency)]
        confirmations = hard.loc[hard["currency"].eq(currency)].copy()
        if fast.empty:
            continue
        confirmations["slow_observation_index"] = [
            int(lookup.loc[(currency, pd.Timestamp(date)), "market_observation_index"])
            for date in confirmations["date"]
        ]
        used_confirmation_rows: set[int] = set()
        for _, signal in fast.iterrows():
            fast_date = pd.Timestamp(signal["date"])
            fast_price = float(lookup.loc[(currency, fast_date), "rub_per_unit"])
            fast_index = int(
                lookup.loc[(currency, fast_date), "market_observation_index"]
            )
            candidates = confirmations.loc[
                confirmations.index.map(lambda value: value not in used_confirmation_rows)
                & confirmations["slow_observation_index"].ge(fast_index)
                & confirmations["slow_observation_index"].le(
                    fast_index + max_wait_observations
                )
            ]
            if candidates.empty:
                rows.append(
                    {
                        "currency": currency,
                        "fast_date": fast_date,
                        "fast_price": fast_price,
                        "confirmed": False,
                        "slow_date": pd.NaT,
                        "slow_price": np.nan,
                        "slow_observation_index": np.nan,
                        "delay_observations": np.nan,
                        "delay_calendar_days": np.nan,
                        "waiting_cost_bps": np.nan,
                        "confirmation_families": "",
                    }
                )
                continue
            confirmation_index = int(candidates.index[0])
            confirmation = candidates.loc[confirmation_index]
            used_confirmation_rows.add(confirmation_index)
            slow_date = pd.Timestamp(confirmation["date"])
            slow_price = float(lookup.loc[(currency, slow_date), "rub_per_unit"])
            rows.append(
                {
                    "currency": currency,
                    "fast_date": fast_date,
                    "fast_price": fast_price,
                    "confirmed": True,
                    "slow_date": slow_date,
                    "slow_price": slow_price,
                    "slow_observation_index": int(
                        confirmation["slow_observation_index"]
                    ),
                    "delay_observations": int(
                        confirmation["slow_observation_index"] - fast_index
                    ),
                    "delay_calendar_days": int((slow_date - fast_date).days),
                    "waiting_cost_bps": (slow_price / fast_price - 1.0) * 10_000,
                    "confirmation_families": confirmation["confirmation_families"],
                }
            )
    return pd.DataFrame(rows).sort_values(["currency", "fast_date"]).reset_index(drop=True)


def waiting_cost_summary(
    pairs: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Summarise confirmation coverage, delay, and paired waiting cost."""
    rows: list[dict[str, object]] = []
    for currency in TARGET_CURRENCIES:
        group = pairs.loc[pairs["currency"].eq(currency)]
        confirmed = group.loc[group["confirmed"]].copy()
        inference = (
            monthly_block_bootstrap_mean(
                confirmed.set_index("fast_date")["waiting_cost_bps"],
                RANDOM_REPEATS,
                deterministic_seed(43, "waiting-cost", currency),
            )
            if not confirmed.empty
            else {"ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
        )
        rows.append(
            {
                "currency": currency,
                "ml_signals": len(group),
                "confirmed_signals": len(confirmed),
                "confirmation_rate": len(confirmed) / len(group) if len(group) else np.nan,
                "median_delay_observations": confirmed["delay_observations"].median(),
                "p90_delay_observations": confirmed["delay_observations"].quantile(0.90),
                "mean_waiting_cost_bps": confirmed["waiting_cost_bps"].mean(),
                "median_waiting_cost_bps": confirmed["waiting_cost_bps"].median(),
                "p90_adverse_waiting_cost_bps": confirmed["waiting_cost_bps"].quantile(0.90),
                "waiting_cost_ci_low_95": inference["ci_low"],
                "waiting_cost_ci_high_95": inference["ci_high"],
                "waiting_cost_p_value_gt_zero": inference["p_value"],
                "immediate_signals_per_week": len(group) / weeks_in_period(start, end),
                "confirmed_signals_per_week": len(confirmed) / weeks_in_period(start, end),
            }
        )
    return pd.DataFrame(rows)


def run_waiting_cost_analysis(root: Path) -> None:
    """Compare immediate ML delivery with waiting for a causal hard fact."""
    ml_dir = root / "data/processed/ml_cross_h_11_to_10"
    hard_dir = root / "data/processed/hard_signals"
    required = [
        ml_dir / "ml_plus_reminders.csv.gz",
        hard_dir / "walk_forward_signals.csv.gz",
        hard_dir / "alternative_walk_forward_signals.csv.gz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run cross-h reproduction, calendar-reminders, and hard backtest first; "
            f"missing: {missing}"
        )
    prices = load_ml_prices(latest_default_input())
    combined = pd.read_csv(ml_dir / "ml_plus_reminders.csv.gz", parse_dates=["date"])
    immediate = combined.loc[combined["signal_source"].eq("ml")].copy()
    spike = pd.read_csv(hard_dir / "walk_forward_signals.csv.gz", parse_dates=["date"])
    spike = spike.assign(signal_family="momentum")
    alternative = pd.read_csv(
        hard_dir / "alternative_walk_forward_signals.csv.gz", parse_dates=["date"]
    )
    alternative = alternative.loc[
        alternative["signal_family"].isin(["level", "corridor_exit_down"])
    ]
    hard = pd.concat(
        [spike[["date", "currency", "signal_family"]],
         alternative[["date", "currency", "signal_family"]]],
        ignore_index=True,
    ).drop_duplicates()
    start = CALENDAR_TEST_START
    _, _, end = evaluate_signal_stream_across_horizons(
        immediate, prices, start, HORIZONS
    )
    pairs = match_delayed_hard_confirmations(
        immediate, hard, prices, start, end, max_wait_observations=5
    )
    confirmed_stream = pairs.loc[
        pairs["confirmed"], ["slow_date", "currency", "slow_observation_index"]
    ].rename(
        columns={"slow_date": "date", "slow_observation_index": "observation_index"}
    )
    immediate_metrics, immediate_summary, _ = evaluate_signal_stream_across_horizons(
        immediate, prices, start, HORIZONS
    )
    confirmed_metrics, confirmed_summary, _ = evaluate_signal_stream_across_horizons(
        confirmed_stream, prices, start, HORIZONS
    )
    comparison = immediate_metrics.merge(
        confirmed_metrics,
        on=["currency", "horizon"],
        suffixes=("_immediate", "_confirmed"),
        validate="one_to_one",
    )
    comparison["lift_change"] = (
        comparison["truth_now_lift_confirmed"]
        - comparison["truth_now_lift_immediate"]
    )
    comparison["benefit_change_bps"] = (
        comparison["benefit_bps_confirmed"] - comparison["benefit_bps_immediate"]
    )
    comparison["frequency_change"] = (
        comparison["signals_per_week_confirmed"]
        - comparison["signals_per_week_immediate"]
    )
    summary = waiting_cost_summary(pairs, start, end)
    pairs.to_csv(ml_dir / "waiting_cost_pairs.csv", index=False)
    summary.to_csv(ml_dir / "waiting_cost_summary.csv", index=False)
    comparison.to_csv(ml_dir / "waiting_policy_comparison.csv", index=False)
    pd.concat(
        [immediate_summary.assign(policy="send_immediately"),
         confirmed_summary.assign(policy="wait_for_hard_confirmation")],
        ignore_index=True,
    ).to_csv(ml_dir / "waiting_policy_horizon_summary.csv", index=False)
    metadata = {
        "fast_policy": "sent ML signal after communication limits",
        "slow_policy": "first unused causal OOT momentum/level/corridor-down confirmation",
        "max_wait_observations": 5,
        "matching": "chronological, within corridor, one hard confirmation per ML episode",
        "waiting_cost_bps": "(slow rub_per_unit / fast rub_per_unit - 1) * 10000; positive is client cost",
        "reporting_window": [str(start.date()), str((end - timedelta(days=1)).date())],
        "no_test_set_tuning": True,
    }
    (ml_dir / "waiting_cost_policy.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Saved waiting-cost analysis to {ml_dir}", flush=True)


def run_cross_h_multi_evaluation(root: Path) -> None:
    """Re-evaluate saved frozen OOT signal dates without fitting any model."""
    output_dir = root / "data/processed/ml_cross_h_11_to_10"
    signal_path = output_dir / "signals_backtest.csv.gz"
    if not signal_path.exists():
        raise FileNotFoundError(
            f"Run --experiment cross-h-reproduction first: missing {signal_path}"
        )
    signals = pd.read_csv(
        signal_path,
        parse_dates=["date", "label_available_date", "outer_start"],
    )
    prices = load_ml_prices(latest_default_input())
    metrics, summary, common_end = evaluate_signal_stream_across_horizons(
        signals, prices, pd.Timestamp("2025-01-01"), HORIZONS
    )
    metrics.to_csv(output_dir / "final_2025_2026_by_horizon.csv", index=False)
    summary.to_csv(output_dir / "final_2025_2026_summary_by_horizon.csv", index=False)
    print(
        f"Saved 2025-2026 horizon profile through "
        f"{(common_end - timedelta(days=1)).date()} to {output_dir}",
        flush=True,
    )


CALENDAR_DEVELOPMENT_START = pd.Timestamp("2023-01-01")
CALENDAR_TEST_START = pd.Timestamp("2025-01-01")
CALENDAR_ANCHOR_DAYS = (5, 20)


def expected_calendar_anchors(
    start: pd.Timestamp, end: pd.Timestamp, radius_days: int
) -> pd.DataFrame:
    """All deterministic 5/20 anchors whose full +/- window is observable."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for month in pd.period_range(start.to_period("M"), (end - timedelta(days=1)).to_period("M"), freq="M"):
        for day in CALENDAR_ANCHOR_DAYS:
            anchor = pd.Timestamp(year=month.year, month=month.month, day=day)
            if (
                anchor - timedelta(days=int(radius_days)) >= start
                and anchor + timedelta(days=int(radius_days)) < end
            ):
                rows.append({"calendar_anchor": anchor, "anchor_day": day})
    return pd.DataFrame(rows)


def calendar_anchor_candidates(
    features: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    radius_days: int,
) -> pd.DataFrame:
    """Attach dates in a completed +/- radius window to their 5/20 anchor.

    The returned rows contain only causal features.  Future outcomes are never
    used to decide which date is selected.  Anchors whose complete calendar
    window is outside [start, end) are deliberately excluded.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    complete = expected_calendar_anchors(start, end, radius_days)[
        "calendar_anchor"
    ].tolist()
    parts: list[pd.DataFrame] = []
    target = features.loc[features["currency"].isin(TARGET_CURRENCIES)]
    for anchor in complete:
        window = target.loc[
            target["date"].between(
                anchor - timedelta(days=int(radius_days)),
                anchor + timedelta(days=int(radius_days)),
            )
        ].copy()
        if not window.empty:
            window["calendar_anchor"] = anchor
            window["anchor_day"] = anchor.day
            window["days_from_anchor"] = (window["date"] - anchor).dt.days
            parts.append(window)
    return pd.concat(parts, ignore_index=True) if parts else target.iloc[0:0].copy()


def calendar_attractive_mask(
    frame: pd.DataFrame,
    lookback: int,
    quantile: float,
    confirmation: str,
) -> pd.Series:
    """Causal market-attractiveness rule used inside calendar windows."""
    level = frame[f"level_percentile_{lookback}"].ge(quantile)
    if confirmation == "level_only":
        return level.fillna(False)
    if confirmation == "falling":
        return (level & frame["return_lag_0"].le(0)).fillna(False)
    if confirmation == "near_min":
        return (level & frame["distance_to_min_20_bps"].le(25)).fillna(False)
    if confirmation == "falling_or_turn":
        return (
            level
            & (frame["return_lag_0"].le(0) | frame["reversal_after_fall"].gt(0))
        ).fillna(False)
    raise ValueError(f"Unknown calendar confirmation: {confirmation}")


def select_calendar_signals(
    candidates: pd.DataFrame,
    lookback: int,
    quantile: float,
    confirmation: str,
    mandatory: bool,
) -> pd.DataFrame:
    """Sequentially choose at most one date per currency/5-or-20 window.

    In quality-only mode the first qualifying published date in the complete
    +/- window is emitted.  In mandatory mode we search from anchor-radius up
    to the anchor; if nothing qualifies, the first publication on/after the
    anchor is emitted as an explicit fallback.  That fallback is causal: on
    its date all failed earlier checks and the calendar deadline are known.
    """
    frame = candidates.copy()
    frame["attractive"] = calendar_attractive_mask(
        frame, lookback, quantile, confirmation
    )
    selected: list[pd.DataFrame] = []
    keys = ["currency", "calendar_anchor"]
    for _, group in frame.sort_values([*keys, "date"]).groupby(keys, sort=True):
        if mandatory:
            eligible = group.loc[group["date"].le(group["calendar_anchor"].iloc[0])]
        else:
            eligible = group
        qualified = eligible.loc[eligible["attractive"]]
        if not qualified.empty:
            row = qualified.iloc[[0]].copy()
            row["calendar_fallback"] = False
        elif mandatory:
            after = group.loc[group["date"].ge(group["calendar_anchor"].iloc[0])]
            if after.empty:
                continue
            row = after.iloc[[0]].copy()
            row["calendar_fallback"] = True
        else:
            continue
        selected.append(row)
    if not selected:
        empty = frame.iloc[0:0].copy()
        empty["calendar_fallback"] = pd.Series(dtype=bool)
        return empty
    result = pd.concat(selected, ignore_index=True)
    result["signal_source"] = np.where(
        result["calendar_fallback"], "calendar_fallback", "calendar_attractive"
    )
    result["message_fact"] = result.apply(
        lambda row: (
            f"курс выгоднее {int(round(row[f'level_percentile_{lookback}'] * 100))}% "
            f"предыдущих {lookback} публикаций"
            if not row["calendar_fallback"]
            else f"плановая дата перевода около {int(row['anchor_day'])}-го числа"
        ),
        axis=1,
    )
    return result.sort_values(["date", "currency"])


def combine_calendar_with_ml(
    ml_signals: pd.DataFrame,
    calendar_signals: pd.DataFrame,
    cooldown: int,
    rolling_days: int = 7,
    rolling_cap: int = 2,
) -> pd.DataFrame:
    """Union streams, then causally thin them to control clustering.

    ML wins ties on the same currency/date.  Afterwards the first eligible
    event is kept, the next `cooldown` market observations are suppressed, and
    at most `rolling_cap` pushes may occur in the preceding `rolling_days`.
    """
    ml = ml_signals.copy()
    ml["signal_source"] = "ml"
    ml["calendar_fallback"] = False
    combined = pd.concat([ml, calendar_signals], ignore_index=True, sort=False)
    priority = {"ml": 0, "calendar_attractive": 1, "calendar_fallback": 2}
    combined["_priority"] = combined["signal_source"].map(priority).fillna(9)
    combined = (
        combined.sort_values(["currency", "date", "_priority"])
        .drop_duplicates(["currency", "date"], keep="first")
        .drop(columns="_priority")
    )
    kept: list[pd.DataFrame] = []
    for _, group in combined.groupby("currency", sort=True):
        last_observation = -10**9
        recent_dates: list[pd.Timestamp] = []
        positions: list[int] = []
        group = group.sort_values("date").reset_index(drop=True)
        for position, row in group.iterrows():
            date = pd.Timestamp(row["date"])
            recent_dates = [
                prior for prior in recent_dates
                if date - prior < timedelta(days=int(rolling_days))
            ]
            if int(row["observation_index"]) - last_observation <= cooldown:
                continue
            if len(recent_dates) >= rolling_cap:
                continue
            positions.append(position)
            last_observation = int(row["observation_index"])
            recent_dates.append(date)
        kept.append(group.iloc[positions])
    return pd.concat(kept, ignore_index=True).sort_values(["date", "currency"])


def signal_distribution_metrics(
    signals: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    stream: str,
) -> pd.DataFrame:
    """Frequency plus clustering and silence diagnostics per corridor."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    total_weeks = max(1, len(pd.period_range(start, end - timedelta(days=1), freq="W")))
    end_month_start = end.to_period("M").start_time
    complete_month_end = end if end == end_month_start else end_month_start
    complete_months = pd.period_range(
        start.to_period("M"), complete_month_end - timedelta(days=1), freq="M"
    )
    total_months = max(1, len(complete_months))
    rows: list[dict[str, object]] = []
    for currency in TARGET_CURRENCIES:
        group = signals.loc[
            signals["currency"].eq(currency)
            & signals["date"].ge(start)
            & signals["date"].lt(end)
        ].sort_values("date")
        gaps_obs = group["observation_index"].diff()
        gaps_days = group["date"].diff().dt.days
        weekly = group.assign(week=group["date"].dt.to_period("W")).groupby("week").size()
        monthly = (
            group.loc[group["date"].lt(complete_month_end)]
            .assign(month=lambda data: data["date"].dt.to_period("M"))
            .groupby("month").size()
        )
        rows.append(
            {
                "stream": stream,
                "currency": currency,
                "signal_count": len(group),
                "signals_per_week": len(group) / weeks_in_period(start, end),
                "active_week_count": weekly.size,
                "total_week_count": total_weeks,
                "active_week_share": weekly.size / total_weeks,
                "empty_week_share": 1 - weekly.size / total_weeks,
                "active_month_share": monthly.size / total_months,
                "empty_month_share": 1 - monthly.size / total_months,
                "share_gap_le_2_observations": gaps_obs.le(2).mean() if len(group) else np.nan,
                "share_gap_le_5_observations": gaps_obs.le(5).mean() if len(group) else np.nan,
                "share_gap_le_1_calendar_day": gaps_days.le(1).mean() if len(group) else np.nan,
                "max_gap_calendar_days": gaps_days.max() if len(group) > 1 else np.nan,
                "gap_calendar_days_q90": (
                    gaps_days.dropna().quantile(0.90, interpolation="higher")
                    if len(group) > 1 else np.nan
                ),
                "gap_calendar_days_q95": (
                    gaps_days.dropna().quantile(0.95, interpolation="higher")
                    if len(group) > 1 else np.nan
                ),
                "gap_calendar_days_q99": (
                    gaps_days.dropna().quantile(0.99, interpolation="higher")
                    if len(group) > 1 else np.nan
                ),
                "max_pushes_in_week": weekly.max() if len(weekly) else 0,
                "max_pushes_in_month": monthly.max() if len(monthly) else 0,
                "weekly_count_cv": (
                    weekly.reindex(
                        pd.period_range(start, end - timedelta(days=1), freq="W"),
                        fill_value=0,
                    ).std()
                    / max(weekly.reindex(
                        pd.period_range(start, end - timedelta(days=1), freq="W"),
                        fill_value=0,
                    ).mean(), 1e-12)
                ),
                "calendar_share": group["signal_source"].ne("ml").mean() if len(group) else 0.0,
                "fallback_share": group["calendar_fallback"].fillna(False).mean() if len(group) else 0.0,
            }
        )
    result = pd.DataFrame(rows)
    # The case does not prescribe numeric clustering cutoffs. These explicit
    # operational thresholds encode its qualitative requirement: no bursts
    # followed by silent months. They are reporting rules, never model-selection
    # targets and never affect lift/benefit.
    result["clustering_pass_operational"] = (
        result["active_week_share"].ge(0.50)
        & result["empty_month_share"].eq(0)
        & result["max_gap_calendar_days"].le(31)
        & result["share_gap_le_2_observations"].le(0.50)
        & result["weekly_count_cv"].le(1.10)
    )
    return result


def _fast_calendar_metrics(
    signals: pd.DataFrame,
    labels_by_horizon: dict[int, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, float]:
    """Cheap development metrics; final inference is calculated separately."""
    result: dict[str, float] = {}
    for horizon, labels in labels_by_horizon.items():
        sample = signals.merge(
            labels[["date", "currency", "truth_now", "benefit_bps", "valid"]],
            on=["date", "currency"], how="inner", validate="one_to_one",
        )
        sample = sample.loc[
            sample["valid"] & sample["date"].ge(start) & sample["date"].lt(end)
        ]
        pool = labels.loc[
            labels["valid"] & labels["date"].ge(start) & labels["date"].lt(end)
        ]
        cells = []
        for currency in TARGET_CURRENCIES:
            chosen = sample.loc[sample["currency"].eq(currency)]
            random_rate = pool.loc[pool["currency"].eq(currency), "truth_now"].mean()
            cells.append(
                (
                    chosen["truth_now"].mean() / random_rate if len(chosen) and random_rate else np.nan,
                    chosen["benefit_bps"].mean() if len(chosen) else np.nan,
                )
            )
        result[f"min_lift_h{horizon}"] = float(np.nanmin([cell[0] for cell in cells]))
        result[f"min_benefit_h{horizon}"] = float(np.nanmin([cell[1] for cell in cells]))
    distribution = signal_distribution_metrics(signals, start, end, "development")
    result["median_signals_per_week"] = float(distribution["signals_per_week"].median())
    result["median_empty_week_share"] = float(distribution["empty_week_share"].median())
    result["max_cluster_share_2"] = float(distribution["share_gap_le_2_observations"].max())
    return result


def run_calendar_augmentation(root: Path) -> None:
    """Select calendar policies pre-2025 and evaluate them on 2025-2026."""
    output_dir = root / "data/processed/ml_cross_h_11_to_10"
    signal_path = output_dir / "signals_backtest.csv.gz"
    if not signal_path.exists():
        raise FileNotFoundError(
            f"Run --experiment cross-h-reproduction first: missing {signal_path}"
        )
    print("Calendar augmentation: loading frozen OOT ML stream", flush=True)
    ml = pd.read_csv(signal_path, parse_dates=["date"])
    prices = load_ml_prices(latest_default_input())
    features = build_features(prices)
    labels_by_horizon = {h: build_labels(prices, h) for h in HORIZONS}
    maturity = [
        labels.loc[labels["valid"]].groupby("currency")["date"].max().min()
        for labels in labels_by_horizon.values()
    ]
    common_end = pd.Timestamp(min(maturity)) + timedelta(days=1)
    ml_keys = ml.drop(
        columns=[
            "truth_now", "local_min", "benefit_bps", "future_regret_bps",
            "label_available_date", "valid", "message_fact",
        ],
        errors="ignore",
    ).merge(
        features[["date", "currency", "observation_index"]],
        on=["date", "currency"], how="left", suffixes=("", "_feature"),
    )
    if "observation_index_feature" in ml_keys:
        ml_keys["observation_index"] = ml_keys["observation_index"].fillna(
            ml_keys.pop("observation_index_feature")
        )

    print("Calendar augmentation: pre-2025 policy grid", flush=True)
    development_labels = {h: labels_by_horizon[h] for h in (5, 10)}
    grid_rows: list[dict[str, object]] = []
    for radius in (2, 3):
        candidates = calendar_anchor_candidates(
            features, CALENDAR_DEVELOPMENT_START, CALENDAR_TEST_START, radius
        )
        for lookback in (20, 60, 120):
            for quantile in (0.60, 0.70, 0.80, 0.90):
                for confirmation in ("level_only", "falling", "near_min", "falling_or_turn"):
                    for mandatory in (False, True):
                        calendar = select_calendar_signals(
                            candidates, lookback, quantile, confirmation, mandatory
                        )
                        for cooldown in (0, 1, 2, 3):
                            combined = combine_calendar_with_ml(
                                ml_keys.loc[
                                    ml_keys["date"].ge(CALENDAR_DEVELOPMENT_START)
                                    & ml_keys["date"].lt(CALENDAR_TEST_START)
                                ],
                                calendar,
                                cooldown,
                            )
                            metrics = _fast_calendar_metrics(
                                combined,
                                development_labels,
                                CALENDAR_DEVELOPMENT_START,
                                CALENDAR_TEST_START,
                            )
                            grid_rows.append(
                                {
                                    "radius_days": radius,
                                    "lookback": lookback,
                                    "level_quantile": quantile,
                                    "confirmation": confirmation,
                                    "mandatory": mandatory,
                                    "cooldown": cooldown,
                                    **metrics,
                                }
                            )
    grid = pd.DataFrame(grid_rows)
    # Freeze one global rule for each product mode.  Stability takes precedence:
    # maximise the worse h=5/h=10 lift, then economic floor and coverage.
    grid["selection_lift"] = grid[["min_lift_h5", "min_lift_h10"]].min(axis=1)
    grid["selection_benefit"] = grid[["min_benefit_h5", "min_benefit_h10"]].min(axis=1)
    feasible = grid.loc[
        grid["median_signals_per_week"].between(1.0, 2.0)
        & grid["selection_benefit"].ge(0)
    ].copy()
    selections: list[pd.Series] = []
    for mandatory in (False, True):
        mode = feasible.loc[feasible["mandatory"].eq(mandatory)]
        if mode.empty:
            mode = grid.loc[grid["mandatory"].eq(mandatory)]
        selections.append(
            mode.sort_values(
                ["selection_lift", "selection_benefit", "median_empty_week_share", "max_cluster_share_2"],
                ascending=[False, False, True, True],
            ).iloc[0]
        )
    coverage_pool = grid.loc[
        grid["mandatory"] & grid["selection_benefit"].ge(0)
    ]
    selections.append(
        coverage_pool.sort_values(
            ["median_signals_per_week", "selection_lift", "selection_benefit"],
            ascending=[False, False, False],
        ).iloc[0]
    )
    selected_policies = pd.DataFrame(selections).reset_index(drop=True)
    selected_policies["stream"] = [
        "ML + attractive calendar",
        "ML + mandatory calendar",
        "ML + calendar coverage",
    ]
    selected_policies["selection_role"] = [
        "quality_first", "quality_first", "coverage_first"
    ]
    selected_policies["frequency_constraint_met"] = selected_policies[
        "median_signals_per_week"
    ].between(1.0, 2.0)
    grid.to_csv(output_dir / "calendar_policy_grid.csv", index=False)
    selected_policies.to_csv(output_dir / "calendar_selected_policies.csv", index=False)

    print("Calendar augmentation: frozen 2025-2026 evaluation", flush=True)
    final_candidates = {
        radius: calendar_anchor_candidates(
            features, CALENDAR_TEST_START, common_end, radius
        )
        for radius in selected_policies["radius_days"].astype(int).unique()
    }
    final_streams: dict[str, pd.DataFrame] = {
        "ML raw": ml_keys.loc[
            ml_keys["date"].ge(CALENDAR_TEST_START) & ml_keys["date"].lt(common_end)
        ].copy()
    }
    final_streams["ML raw"]["signal_source"] = "ml"
    final_streams["ML raw"]["calendar_fallback"] = False
    # A fixed anti-clustering baseline makes the impact of calendar additions clear.
    final_streams["ML thinned"] = combine_calendar_with_ml(
        final_streams["ML raw"], features.iloc[0:0].copy(), cooldown=2
    )
    for _, policy in selected_policies.iterrows():
        calendar = select_calendar_signals(
            final_candidates[int(policy["radius_days"])],
            int(policy["lookback"]),
            float(policy["level_quantile"]),
            str(policy["confirmation"]),
            bool(policy["mandatory"]),
        )
        final_streams[str(policy["stream"])] = combine_calendar_with_ml(
            final_streams["ML raw"], calendar, int(policy["cooldown"])
        )

    metric_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    distribution_parts: list[pd.DataFrame] = []
    signal_parts: list[pd.DataFrame] = []
    for name, stream in final_streams.items():
        metrics, summary, _ = evaluate_signal_stream_across_horizons(
            stream, prices, CALENDAR_TEST_START, HORIZONS
        )
        metrics.insert(0, "stream", name)
        summary.insert(0, "stream", name)
        metric_parts.append(metrics)
        summary_parts.append(summary)
        distribution_parts.append(
            signal_distribution_metrics(stream, CALENDAR_TEST_START, common_end, name)
        )
        exported = stream.copy()
        exported.insert(0, "stream", name)
        signal_parts.append(exported)
    pd.concat(metric_parts, ignore_index=True).to_csv(
        output_dir / "calendar_final_by_horizon.csv", index=False
    )
    pd.concat(summary_parts, ignore_index=True).to_csv(
        output_dir / "calendar_final_summary.csv", index=False
    )
    pd.concat(distribution_parts, ignore_index=True).to_csv(
        output_dir / "calendar_distribution.csv", index=False
    )
    pd.concat(signal_parts, ignore_index=True).to_csv(
        output_dir / "calendar_signals_backtest.csv.gz", index=False
    )
    print(
        f"Saved causal calendar analysis through {(common_end - timedelta(days=1)).date()} "
        f"to {output_dir}",
        flush=True,
    )


def mark_recent_ml_signal(
    candidates: pd.DataFrame,
    ml_signals: pd.DataFrame,
    observations: int = 2,
) -> pd.DataFrame:
    """Mark whether an ML push is already known on T or the prior observations."""
    result = candidates.copy()
    result["recent_ml_signal"] = False
    for currency, group in result.groupby("currency", sort=True):
        ml_indices = np.sort(
            ml_signals.loc[
                ml_signals["currency"].eq(currency), "observation_index"
            ].dropna().astype(int).unique()
        )
        if not len(ml_indices):
            continue
        current = group["observation_index"].astype(int).to_numpy()
        right = np.searchsorted(ml_indices, current, side="right") - 1
        has_previous = right >= 0
        previous = np.where(has_previous, ml_indices[np.maximum(right, 0)], -10**9)
        result.loc[group.index, "recent_ml_signal"] = (
            has_previous & ((current - previous) <= observations)
        )
    return result


def select_mandatory_reminders(
    candidates: pd.DataFrame,
    ml_signals: pd.DataFrame,
    lookback: int = 20,
    quantile: float = 0.80,
    avoid_recent_ml_observations: int = 2,
) -> pd.DataFrame:
    """Choose one causal CRM reminder for every completed 5/20 window.

    Before the anchor, the first attractive date without a recent ML message is
    preferred.  If none exists, the first publication on/after the anchor is a
    mandatory reminder.  The reminder never claims that the rate will improve.
    """
    frame = mark_recent_ml_signal(
        candidates, ml_signals, avoid_recent_ml_observations
    )
    frame["attractive"] = calendar_attractive_mask(
        frame, lookback, quantile, "level_only"
    )
    rows: list[pd.DataFrame] = []
    for _, group in frame.sort_values(
        ["currency", "calendar_anchor", "date"]
    ).groupby(["currency", "calendar_anchor"], sort=True):
        anchor = pd.Timestamp(group["calendar_anchor"].iloc[0])
        preferred = group.loc[
            group["date"].le(anchor)
            & group["attractive"]
            & ~group["recent_ml_signal"]
        ]
        if not preferred.empty:
            row = preferred.iloc[[0]].copy()
            row["reminder_fallback"] = False
        else:
            fallback = group.loc[group["date"].ge(anchor)]
            if fallback.empty:
                # A completed +/-3 window normally contains a CBR publication;
                # keeping this explicit protects against unexpected data holes.
                continue
            row = fallback.iloc[[0]].copy()
            row["reminder_fallback"] = True
        rows.append(row)
    if not rows:
        empty = frame.iloc[0:0].copy()
        empty["reminder_fallback"] = pd.Series(dtype=bool)
        return empty
    reminders = pd.concat(rows, ignore_index=True)
    reminders["signal_source"] = "calendar_reminder"
    reminders["message_fact"] = reminders.apply(
        lambda row: (
            f"Напоминание о переводе: курс ниже, чем в "
            f"{int(round(row[f'level_percentile_{lookback}'] * 100))}% "
            f"предыдущих {lookback} публикаций; сумма и реквизиты заполнены"
            if not row["reminder_fallback"]
            else f"Плановый перевод около {int(row['anchor_day'])}-го числа: "
            "сумма и реквизиты уже заполнены"
        ),
        axis=1,
    )
    return reminders.sort_values(["date", "currency"])


def fill_missing_calendar_reminders(
    reminders: pd.DataFrame,
    features: pd.DataFrame,
    expected_anchors: pd.DataFrame,
) -> pd.DataFrame:
    """Emit a calendar-day fallback when no CBR publication exists in a window."""
    expected = pd.MultiIndex.from_product(
        [TARGET_CURRENCIES, expected_anchors["calendar_anchor"]],
        names=["currency", "calendar_anchor"],
    )
    actual = pd.MultiIndex.from_frame(
        reminders[["currency", "calendar_anchor"]]
    )
    missing = expected.difference(actual)
    rows: list[pd.DataFrame] = []
    for currency, anchor in missing:
        history = features.loc[
            features["currency"].eq(currency) & features["date"].le(anchor)
        ].sort_values("date")
        if history.empty:
            raise RuntimeError(f"No known rate for {currency} by {anchor.date()}")
        row = history.iloc[[-1]].copy()
        row["date"] = anchor
        row["calendar_anchor"] = anchor
        row["anchor_day"] = anchor.day
        row["days_from_anchor"] = 0
        row["attractive"] = False
        row["recent_ml_signal"] = False
        row["reminder_fallback"] = True
        row["signal_source"] = "calendar_reminder"
        row["message_fact"] = (
            f"Плановый перевод около {anchor.day}-го числа: "
            "сумма и реквизиты уже заполнены"
        )
        rows.append(row)
    if rows:
        reminders = pd.concat([reminders, *rows], ignore_index=True, sort=False)
    return reminders.sort_values(["date", "currency"]).reset_index(drop=True)


def union_ml_and_reminders(
    ml_signals: pd.DataFrame, reminders: pd.DataFrame
) -> pd.DataFrame:
    """Deduplicate same-day ML and CRM messages without suppressing either event."""
    ml = ml_signals.copy()
    ml["signal_source"] = "ml"
    reminder_columns = [
        "date", "currency", "observation_index", "calendar_anchor", "anchor_day", "days_from_anchor",
        "reminder_fallback", "attractive", "recent_ml_signal", "message_fact",
    ]
    reminder_info = reminders[reminder_columns].rename(
        columns={"message_fact": "reminder_message"}
    )
    combined = ml.merge(
        reminder_info, on=["date", "currency"], how="outer", indicator=True,
        suffixes=("", "_reminder"), validate="one_to_one",
    )
    combined["signal_source"] = combined["_merge"].map(
        {"left_only": "ml", "right_only": "calendar_reminder", "both": "ml+reminder"}
    ).astype("object")
    if "observation_index_reminder" in combined:
        combined["observation_index"] = combined["observation_index"].fillna(
            combined.pop("observation_index_reminder")
        )
    combined = combined.drop(columns="_merge")
    return combined.sort_values(["date", "currency"]).reset_index(drop=True)


def apply_weekly_push_cap(frame: pd.DataFrame, max_pushes: int = 2) -> pd.DataFrame:
    """Causally keep at most the first `max_pushes` communications Mon-Sun."""
    if max_pushes < 1:
        raise ValueError("max_pushes must be positive")
    result = frame.sort_values(["currency", "date"]).copy()
    result["_calendar_week"] = result["date"].dt.to_period("W-SUN")
    result["_position_in_week"] = result.groupby(
        ["currency", "_calendar_week"], sort=False
    ).cumcount()
    result = result.loc[result["_position_in_week"].lt(max_pushes)].drop(
        columns=["_calendar_week", "_position_in_week"]
    )
    return result.sort_values(["date", "currency"]).reset_index(drop=True)


def apply_ml_communication_limits(
    frame: pd.DataFrame,
    max_pushes_per_week: int = 2,
    min_gap_calendar_days: int = 1,
) -> pd.DataFrame:
    """Causally enforce both a weekly cap and no adjacent-day ML pushes.

    `min_gap_calendar_days=1` means that after a kept signal on T, another ML
    signal may be sent no earlier than T+2 calendar days. Rejected candidates
    do not consume the weekly quota, so a later valid signal can still be sent.
    """
    if max_pushes_per_week < 1 or min_gap_calendar_days < 0:
        raise ValueError("Communication limits must be non-negative and non-zero")
    kept: list[pd.DataFrame] = []
    for _, group in frame.sort_values(["currency", "date"]).groupby(
        "currency", sort=True
    ):
        weekly_counts: dict[pd.Period, int] = {}
        last_sent: pd.Timestamp | None = None
        positions: list[int] = []
        group = group.reset_index(drop=True)
        for position, row in group.iterrows():
            date = pd.Timestamp(row["date"])
            week = date.to_period("W-SUN")
            if (
                last_sent is not None
                and (date - last_sent).days <= min_gap_calendar_days
            ):
                continue
            if weekly_counts.get(week, 0) >= max_pushes_per_week:
                continue
            positions.append(position)
            weekly_counts[week] = weekly_counts.get(week, 0) + 1
            last_sent = date
        kept.append(group.iloc[positions])
    return (
        pd.concat(kept, ignore_index=True).sort_values(["date", "currency"])
        if kept else frame.iloc[0:0].copy()
    )


def reminders_for_uncovered_anchors(
    features: pd.DataFrame,
    sent_ml: pd.DataFrame,
    anchors: pd.DataFrame,
    radius_days: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Schedule anchor+radius reminders only when no sent ML exists in +/-radius.

    Waiting until the right edge is deliberate: only then can the decision use
    the fact that no ML signal occurred anywhere in the symmetric window
    without looking into the future.
    """
    reminder_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for currency in TARGET_CURRENCIES:
        currency_ml = sent_ml.loc[sent_ml["currency"].eq(currency)]
        currency_features = features.loc[features["currency"].eq(currency)].sort_values("date")
        for anchor in anchors["calendar_anchor"]:
            left = anchor - timedelta(days=radius_days)
            right = anchor + timedelta(days=radius_days)
            covered = currency_ml["date"].between(left, right).any()
            audit_rows.append(
                {
                    "currency": currency,
                    "calendar_anchor": anchor,
                    "covered_by_ml": bool(covered),
                    "reminder_candidate": not bool(covered),
                }
            )
            if covered:
                continue
            history = currency_features.loc[currency_features["date"].le(right)]
            if history.empty:
                raise RuntimeError(f"No known rate for {currency} by {right.date()}")
            row = history.iloc[[-1]].copy()
            # CRM messages may be sent on a non-publication day.  The copied
            # observation_index identifies the last rate known by that date.
            row["date"] = right
            row["calendar_anchor"] = anchor
            row["anchor_day"] = anchor.day
            row["days_from_anchor"] = radius_days
            row["attractive"] = row["level_percentile_20"].ge(0.80)
            row["recent_ml_signal"] = False
            row["signal_source"] = "calendar_reminder"
            row["reminder_fallback"] = ~row["attractive"]
            row["message_fact"] = row.apply(
                lambda item: (
                    f"Напоминание о переводе около {anchor.day}-го: курс ниже, "
                    f"чем в {int(round(item['level_percentile_20'] * 100))}% "
                    "предыдущих 20 публикаций; сумма и реквизиты заполнены"
                    if bool(item["attractive"])
                    else f"Плановый перевод около {anchor.day}-го числа: "
                    "сумма и реквизиты уже заполнены"
                ),
                axis=1,
            )
            reminder_rows.append(row)
    reminders = (
        pd.concat(reminder_rows, ignore_index=True, sort=False)
        if reminder_rows else features.iloc[0:0].copy()
    )
    return reminders.sort_values(["date", "currency"]), pd.DataFrame(audit_rows)


def unconditional_anchor_reminders(
    features: pd.DataFrame,
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    """Create one non-financial CRM reminder exactly on every 5/20 anchor."""
    rows: list[pd.DataFrame] = []
    for currency in TARGET_CURRENCIES:
        history = features.loc[features["currency"].eq(currency)].sort_values("date")
        for anchor in anchors["calendar_anchor"]:
            known = history.loc[history["date"].le(anchor)]
            if known.empty:
                raise RuntimeError(f"No known rate for {currency} by {anchor.date()}")
            row = known.iloc[[-1]].copy()
            row["date"] = anchor
            row["calendar_anchor"] = anchor
            row["anchor_day"] = anchor.day
            row["days_from_anchor"] = 0
            row["attractive"] = row["level_percentile_20"].ge(0.80)
            row["recent_ml_signal"] = False
            row["reminder_fallback"] = ~row["attractive"]
            row["signal_source"] = "calendar_reminder"
            row["message_fact"] = row.apply(
                lambda item: (
                    f"Напоминание о переводе на {anchor.day}-е число: курс ниже, "
                    f"чем в {int(round(item['level_percentile_20'] * 100))}% "
                    "предыдущих 20 публикаций; сумма и реквизиты заполнены"
                    if bool(item["attractive"])
                    else f"Плановый перевод на {anchor.day}-е число: "
                    "сумма и реквизиты уже заполнены"
                ),
                axis=1,
            )
            rows.append(row)
    return pd.concat(rows, ignore_index=True, sort=False).sort_values(
        ["date", "currency"]
    )


def run_calendar_reminders(root: Path) -> None:
    """Enforce max two weekly pushes and add non-financial 5/20 reminders."""
    output_dir = root / "data/processed/ml_cross_h_11_to_10"
    signal_path = output_dir / "signals_backtest.csv.gz"
    if not signal_path.exists():
        raise FileNotFoundError(
            f"Run --experiment cross-h-reproduction first: missing {signal_path}"
        )
    prices = load_ml_prices(latest_default_input())
    features = build_features(prices)
    ml = pd.read_csv(signal_path, parse_dates=["date"])
    ml_keys = ml.drop(
        columns=[
            "truth_now", "local_min", "benefit_bps", "future_regret_bps",
            "label_available_date", "valid", "message_fact",
        ], errors="ignore",
    )
    # Use the same h=20 maturity boundary as the financial ML report so all
    # displayed 2025-2026 periods are directly comparable.
    labels_h20 = build_labels(prices, max(HORIZONS))
    common_end = (
        labels_h20.loc[labels_h20["valid"]]
        .groupby("currency")["date"].max().min()
        + timedelta(days=1)
    )
    start = CALENDAR_TEST_START
    ml_window = ml_keys.loc[
        ml_keys["date"].ge(start) & ml_keys["date"].lt(common_end)
    ].copy()
    raw_ml = ml_window.assign(signal_source="ml")
    # First remove ML bursts. This preliminary stream is also the only stream
    # allowed to cover an anchor and suppress its reminder.
    weekly_capped_ml = apply_weekly_push_cap(raw_ml, max_pushes=2)
    capped_ml = apply_ml_communication_limits(
        raw_ml, max_pushes_per_week=2, min_gap_calendar_days=1
    )
    expected_anchor_dates = expected_calendar_anchors(start, common_end, radius_days=3)
    strict_reminders, anchor_audit = reminders_for_uncovered_anchors(
        features, capped_ml, expected_anchor_dates, radius_days=3
    )
    strict_candidates = union_ml_and_reminders(capped_ml, strict_reminders)
    strict_combined = apply_weekly_push_cap(strict_candidates, max_pushes=2)
    emitted_reminder_keys = strict_combined.loc[
        strict_combined["signal_source"].isin(["calendar_reminder", "ml+reminder"]),
        ["currency", "calendar_anchor"],
    ].dropna().drop_duplicates()
    anchor_audit = anchor_audit.merge(
        emitted_reminder_keys.assign(reminder_emitted=True),
        on=["currency", "calendar_anchor"], how="left",
    )
    anchor_audit["reminder_emitted"] = anchor_audit["reminder_emitted"].fillna(False)
    anchor_audit["covered_after_policy"] = (
        anchor_audit["covered_by_ml"] | anchor_audit["reminder_emitted"]
    )

    # Requested alternative: cap only ML, then ALWAYS add separate reminders
    # exactly on every 5th and 20th.  They are not cancelled by nearby/same-day
    # ML and there is deliberately no cap on the combined communication stream.
    exact_anchor_dates = expected_calendar_anchors(
        start, common_end, radius_days=0
    )
    reminders = unconditional_anchor_reminders(features, exact_anchor_dates)
    combined = pd.concat([capped_ml, reminders], ignore_index=True, sort=False).sort_values(
        ["date", "currency", "signal_source"]
    ).reset_index(drop=True)

    distribution = pd.concat(
        [
            signal_distribution_metrics(
                raw_ml.assign(calendar_fallback=False), start, common_end, "ML raw",
            ),
            signal_distribution_metrics(
                weekly_capped_ml.assign(calendar_fallback=False),
                start, common_end, "ML weekly cap",
            ),
            signal_distribution_metrics(
                capped_ml.assign(calendar_fallback=False),
                start, common_end, "ML weekly cap + day gap",
            ),
            signal_distribution_metrics(
                strict_combined.assign(
                    calendar_fallback=strict_combined["reminder_fallback"].fillna(False)
                ),
                start, common_end, "Strict total cap",
            ),
            signal_distribution_metrics(
                combined.assign(
                    calendar_fallback=combined["reminder_fallback"].fillna(False)
                ),
                start, common_end, "ML cap + mandatory 5/20",
            ),
        ],
        ignore_index=True,
    )
    strict_coverage_summary = anchor_audit.groupby("currency", as_index=False).agg(
        anchors=("calendar_anchor", "size"),
        covered_by_ml=("covered_by_ml", "sum"),
        reminder_candidates=("reminder_candidate", "sum"),
        reminders_emitted=("reminder_emitted", "sum"),
        anchors_covered_after_policy=("covered_after_policy", "sum"),
    )
    strict_coverage_summary["anchor_coverage"] = (
        strict_coverage_summary["anchors_covered_after_policy"]
        / strict_coverage_summary["anchors"]
    )
    coverage_summary = reminders.groupby("currency", as_index=False).agg(
        anchors=("calendar_anchor", "nunique"),
        reminders_emitted=("calendar_anchor", "size"),
        attractive_reminders=("reminder_fallback", lambda values: int((~values).sum())),
    )
    coverage_summary["anchor_coverage"] = 1.0

    financial_metrics, financial_summary, _ = evaluate_signal_stream_across_horizons(
        capped_ml, prices, start, HORIZONS
    )
    total_frequency = distribution.loc[
        distribution["stream"].eq("ML cap + mandatory 5/20"), "signals_per_week"
    ]
    financial_summary["median_ml_pushes_per_week"] = financial_summary[
        "median_signals_per_week"
    ]
    financial_summary["mean_ml_pushes_per_week"] = distribution.loc[
        distribution["stream"].eq("ML weekly cap + day gap"), "signals_per_week"
    ].mean()
    financial_summary["mean_total_pushes_per_week"] = total_frequency.mean()
    financial_summary["min_total_pushes_per_week"] = total_frequency.min()
    financial_summary["max_total_pushes_per_week"] = total_frequency.max()

    reminder_export_columns = [
        "date", "currency", "observation_index", "calendar_anchor", "anchor_day",
        "days_from_anchor", "level_percentile_20", "attractive",
        "reminder_fallback", "signal_source", "message_fact",
    ]
    reminders[reminder_export_columns].to_csv(
        output_dir / "calendar_reminders.csv", index=False
    )
    combined.to_csv(output_dir / "ml_plus_reminders.csv.gz", index=False)
    distribution.to_csv(output_dir / "reminder_distribution.csv", index=False)
    coverage_summary.to_csv(output_dir / "reminder_coverage.csv", index=False)
    anchor_audit.to_csv(output_dir / "reminder_anchor_audit.csv", index=False)
    strict_coverage_summary.to_csv(
        output_dir / "strict_reminder_coverage.csv", index=False
    )
    strict_combined.to_csv(output_dir / "strict_ml_plus_reminders.csv.gz", index=False)
    financial_metrics.to_csv(
        output_dir / "communication_ml_by_horizon.csv", index=False
    )
    financial_summary.to_csv(
        output_dir / "communication_summary_by_horizon.csv", index=False
    )
    policy = {
        "role": "non-financial CRM reminder; excluded from hit/lift/benefit",
        "anchors": [5, 20],
        "radius_calendar_days": 0,
        "ml_limits": "at most two chronological ML pushes per Monday-Sunday week and no ML pushes on adjacent calendar days",
        "reminder_rule": "one separate CRM reminder exactly on every 5th and 20th; never cancelled by nearby ML; no total-stream cap",
        "same_day_rule": "same-day ML and reminder remain two separate communications",
        "strict_control_same_day_rule": "the separate strict-control stream deduplicates same-day ML and reminder",
        "reporting_window": [str(start.date()), str((common_end - timedelta(days=1)).date())],
        "financial_metrics_scope": "only ML rows remaining after the ML-only weekly cap",
    }
    (output_dir / "reminder_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"Saved mandatory reminder analysis through "
        f"{(common_end - timedelta(days=1)).date()} to {output_dir}", flush=True
    )


def cross_h_ml_candidates_as_of_date(
    prices: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Return all causal ML candidates in the active frozen-policy fold."""
    as_of = pd.Timestamp(as_of).normalize()
    causal_prices = prices.loc[prices["date"].le(as_of)].copy()
    features = build_features(causal_prices)
    target_dates = causal_prices.loc[
        causal_prices["currency"].isin(TARGET_CURRENCIES), "date"
    ]
    if target_dates.empty:
        return pd.DataFrame()
    score_date = pd.Timestamp(target_dates.max())
    policy_years = [year for year in CROSS_H_LEGACY_CHOICES if year <= score_date.year]
    if not policy_years:
        raise ValueError(f"No frozen cross-h policy available by {score_date.date()}")
    policy_year = max(policy_years)
    fold_start = pd.Timestamp(f"{policy_year}-01-01")
    choice = CROSS_H_LEGACY_CHOICES[policy_year]
    config_name = str(choice["model_config"])
    config = NIGHTLY_MODEL_CONFIGS[config_name]
    lookback = int(choice["score_lookback"])
    quantile = float(choice["joint_quantile"])

    data = features.merge(
        build_labels(causal_prices, CROSS_H_TRAIN_HORIZON),
        on=["date", "currency"],
        how="inner",
    )
    data["good_push"] = (
        data["truth_now"].eq(1) & data["benefit_bps"].gt(0)
    ).astype(float)
    columns = feature_columns(data)
    reference_start = fold_start - pd.DateOffset(years=1)
    reference_train = mature_training_rows(
        data,
        model_train_start(str(config["history"]), reference_start),
        reference_start,
    )
    reference = data.loc[
        data["date"].ge(reference_start) & data["date"].lt(fold_start)
    ].copy()
    reference_model = make_tuned_good_push_model(config_name, NIGHTLY_MODEL_CONFIGS)
    reference_model.fit(reference_train[columns], reference_train["good_push"])
    reference["p_good_push"] = reference_model.predict_proba(reference[columns])[:, 1]

    train = mature_training_rows(
        data,
        model_train_start(str(config["history"]), fold_start),
        fold_start,
    )
    model = make_tuned_good_push_model(config_name, NIGHTLY_MODEL_CONFIGS)
    model.fit(train[columns], train["good_push"])
    score_stream = data.loc[
        data["date"].ge(fold_start) & data["date"].le(score_date)
    ].copy()
    score_stream["p_good_push"] = model.predict_proba(score_stream[columns])[:, 1]
    ranked = causal_score_percentiles(
        reference, score_stream, ("p_good_push",), lookback=lookback
    )
    selected = ranked.loc[ranked["p_good_push_rank"].ge(quantile)].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "date", "currency", "train_horizon", "evaluation_horizon",
                "scenario", "direction", "strength", "speed", "p_good_push",
                "p_good_push_rank", "message_fact", "model_config",
                "score_lookback", "joint_quantile",
            ]
        )
    selected = add_message_facts(selected)
    selected["train_horizon"] = CROSS_H_TRAIN_HORIZON
    selected["evaluation_horizon"] = CROSS_H_EVAL_HORIZON
    selected["scenario"] = "favourable_now"
    selected["direction"] = "lower_rub_per_unit_is_better"
    selected["strength"] = selected["p_good_push_rank"]
    selected["speed"] = selected["return_mean_3"] * 10_000
    selected["model_config"] = config_name
    selected["score_lookback"] = lookback
    selected["joint_quantile"] = quantile
    columns_out = [
        "date", "currency", "train_horizon", "evaluation_horizon", "scenario",
        "direction", "strength", "speed", "p_good_push", "p_good_push_rank",
        "message_fact", "model_config", "score_lookback", "joint_quantile",
        "observation_index", "return_mean_3", "streak_length",
        "level_percentile_60",
    ]
    return selected[columns_out].sort_values(["date", "currency"])


def cross_h_signals_as_of_date(
    prices: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Return current ML candidates using no observations after ``as_of``."""
    as_of = pd.Timestamp(as_of).normalize()
    candidates = cross_h_ml_candidates_as_of_date(prices, as_of)
    if candidates.empty:
        return candidates
    latest = (
        prices.loc[
            prices["date"].le(as_of)
            & prices["currency"].isin(TARGET_CURRENCIES)
        ]
        .groupby("currency")["date"]
        .max()
        .rename("latest_date")
    )
    current = candidates.join(latest, on="currency")
    return (
        current.loc[current["date"].eq(current["latest_date"])]
        .drop(columns="latest_date")
        .sort_values(["date", "currency"])
        .reset_index(drop=True)
    )


def _format_rate(value: float) -> str:
    """Format a normalized RUB-per-unit rate without false precision."""
    decimals = 4 if value < 1 else 2
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".").replace(".", ",")


def _production_copy_features(
    candidates: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Attach causal price and a fixed L20 stable-corridor description."""
    if candidates.empty:
        return candidates.copy()
    parts: list[pd.DataFrame] = []
    target = prices.loc[prices["currency"].isin(TARGET_CURRENCIES)].copy()
    for currency, group in target.groupby("currency", sort=True):
        group = group.sort_values("date")[["date", "currency", "rub_per_unit"]].copy()
        prior = group["rub_per_unit"].shift(1).rolling(20, min_periods=20)
        group["prior_low_20"] = prior.min()
        group["prior_high_20"] = prior.max()
        group["prior_range_20_bps"] = (
            group["prior_high_20"] / group["prior_low_20"] - 1
        ) * 10_000
        parts.append(group)
    facts = pd.concat(parts, ignore_index=True)
    return candidates.merge(facts, on=["date", "currency"], how="left", validate="many_to_one")


def render_ml_pushes(
    candidates: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Render approved factual copy for already selected ML candidates."""
    frame = _production_copy_features(candidates, prices)
    if frame.empty:
        return pd.DataFrame(columns=PRODUCTION_SIGNAL_COLUMNS)

    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        currency = str(row["currency"])
        copy = CURRENCY_COPY[currency]
        rate = float(row["rub_per_unit"])
        percentile = float(row.get("level_percentile_60", np.nan))
        streak_value = row.get("streak_length", 0)
        streak = int(streak_value) if pd.notna(streak_value) else 0
        prior_low = float(row.get("prior_low_20", np.nan))
        prior_range = float(row.get("prior_range_20_bps", np.nan))

        if np.isfinite(percentile) and percentile >= 0.80:
            template_id = "ml_low_level"
            explanation = "low_level"
            title = f"Курс {copy['genitive']} ниже большинства недавних значений"
            body = (
                f"Последний курс ЦБ — {_format_rate(rate)} ₽ за {copy['unit']}. "
                f"Это ниже, чем в {int(round(percentile * 100))}% из предыдущих "
                f"60 публикаций. Проверьте условия перевода в {copy['country']}."
            )
        elif (
            np.isfinite(prior_low)
            and np.isfinite(prior_range)
            and prior_range <= 400
            and rate < prior_low
        ):
            template_id = "ml_corridor_exit_down"
            explanation = "corridor_exit_down_L20_R400"
            exit_pct = (prior_low - rate) / prior_low * 100
            title = f"Курс {copy['genitive']} вышел ниже недавнего диапазона"
            body = (
                f"Последний курс ЦБ на {str(f'{exit_pct:.1f}').replace('.', ',')}% "
                f"ниже нижней границы диапазона предыдущих 20 публикаций. "
                f"Проверьте условия перевода в {copy['country']}."
            )
        elif streak >= 2:
            template_id = "ml_momentum_down"
            explanation = "momentum_down"
            title = f"Курс {copy['genitive']} снижается"
            body = (
                f"По данным ЦБ курс снижался в каждой из последних {streak} "
                f"публикаций. Проверьте актуальные условия перевода "
                f"в {copy['country']}."
            )
        else:
            template_id = "ml_neutral_fallback"
            explanation = "current_rate"
            title = f"Обновился ориентир по курсу {copy['genitive']}"
            body = (
                f"Последний опубликованный курс ЦБ — {_format_rate(rate)} ₽ "
                f"за {copy['unit']}. Актуальный курс перевода доступен в приложении."
            )

        rows.append(
            {
                "date": pd.Timestamp(row["date"]),
                "corridor": f"RUB→{currency}",
                "indicator": "ml_good_push",
                "direction": "foreign_currency_down_is_better",
                "strength": float(row["p_good_push_rank"]),
                # Positive speed means movement in the favourable direction.
                "speed": (
                    -float(row["return_mean_3"]) * 10_000
                    if pd.notna(row["return_mean_3"])
                    else np.nan
                ),
                "recommended_scenario": "favourable_now",
                "signal_source": "ml",
                "explanation_indicator": explanation,
                "template_id": template_id,
                "push_title": title,
                "push_text": body,
                "currency": currency,
                "rub_per_unit": rate,
            }
        )
    return pd.DataFrame(rows, columns=PRODUCTION_SIGNAL_COLUMNS)


def calendar_reminders_as_of_date(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    prefill_available: bool = False,
) -> pd.DataFrame:
    """Return neutral reminders exactly on the 5th/20th, with no FX claim."""
    as_of = pd.Timestamp(as_of).normalize()
    if as_of.day not in (5, 20):
        return pd.DataFrame(columns=PRODUCTION_SIGNAL_COLUMNS)
    causal = prices.loc[
        prices["date"].le(as_of) & prices["currency"].isin(TARGET_CURRENCIES)
    ].copy()
    latest = causal.sort_values("date").groupby("currency", as_index=False).tail(1)
    rows: list[dict[str, object]] = []
    for _, row in latest.iterrows():
        currency = str(row["currency"])
        copy = CURRENCY_COPY[currency]
        suffix = "prefilled" if prefill_available else "plain"
        if prefill_available:
            body = (
                f"Сегодня {as_of.day}-е число. Сумма и реквизиты прошлого "
                "перевода уже заполнены — проверьте их перед подтверждением."
            )
        else:
            body = (
                f"Сегодня {as_of.day}-е число. Если планировали перевод, "
                "проверьте актуальные условия в приложении."
            )
        rows.append(
            {
                "date": as_of,
                "corridor": f"RUB→{currency}",
                "indicator": f"calendar_{as_of.day}",
                "direction": "calendar_neutral",
                "strength": np.nan,
                "speed": np.nan,
                "recommended_scenario": "scheduled_transfer",
                "signal_source": "calendar_reminder",
                "explanation_indicator": "calendar_date",
                "template_id": f"calendar_{as_of.day}_{suffix}",
                "push_title": f"Напоминание о переводе в {copy['country']}",
                "push_text": body,
                "currency": currency,
                "rub_per_unit": float(row["rub_per_unit"]),
            }
        )
    return pd.DataFrame(rows, columns=PRODUCTION_SIGNAL_COLUMNS)


def combined_signals_as_of_date(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    prefill_available: bool = False,
) -> pd.DataFrame:
    """Build the final causal ML + 5/20 communication stream for one cutoff.

    ML is limited to the first two candidates per currency and Monday-Sunday
    week, with no ML messages on adjacent calendar days. Calendar reminders are
    added afterwards and deliberately do not consume the ML quota.
    """
    as_of = pd.Timestamp(as_of).normalize()
    causal_prices = prices.loc[prices["date"].le(as_of)].copy()
    candidates = cross_h_ml_candidates_as_of_date(causal_prices, as_of)

    # The first calendar week can start in the preceding policy year. Include
    # just enough of that previous frozen fold to enforce the cap and day gap.
    week_start = as_of.to_period("W-SUN").start_time
    required_start = min(week_start, as_of - timedelta(days=2))
    active_year_start = pd.Timestamp(f"{as_of.year}-01-01")
    previous_cutoff = active_year_start - timedelta(days=1)
    has_previous_policy = any(
        year <= previous_cutoff.year for year in CROSS_H_LEGACY_CHOICES
    )
    if required_start < active_year_start and has_previous_policy:
        previous = cross_h_ml_candidates_as_of_date(
            causal_prices, previous_cutoff
        )
        candidates = pd.concat(
            [previous.loc[previous["date"].ge(required_start)], candidates],
            ignore_index=True,
        ).drop_duplicates(["date", "currency"], keep="last")

    if candidates.empty:
        ml_today = candidates
    else:
        sent_ml = apply_ml_communication_limits(
            candidates,
            max_pushes_per_week=2,
            min_gap_calendar_days=1,
        )
        ml_today = sent_ml.loc[sent_ml["date"].eq(as_of)].copy()
    rendered_ml = render_ml_pushes(ml_today, causal_prices)
    reminders = calendar_reminders_as_of_date(
        causal_prices, as_of, prefill_available=prefill_available
    )
    streams = [stream for stream in (rendered_ml, reminders) if not stream.empty]
    if not streams:
        return pd.DataFrame(columns=PRODUCTION_SIGNAL_COLUMNS)
    return (
        pd.concat(streams, ignore_index=True)
        .sort_values(["date", "corridor", "signal_source"])
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=(
            "nested",
            "training-windows",
            "candidate-gate",
            "binary-truth",
            "regret-regression",
            "joint-policy",
            "horizon-tuning",
            "nightly-tuning",
            "cross-h-reproduction",
            "cross-h-multi-eval",
            "calendar-reminders",
            "waiting-cost",
        ),
        default="nested",
        help="Run the main policy or one of the isolated ML ablations.",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Calculate production-style signals using only data available by YYYY-MM-DD.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        choices=range(1, 21),
        help="Horizon for --as-of mode.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        choices=range(1, 21),
        default=None,
        help="Horizons for an ablation experiment.",
    )
    parser.add_argument(
        "--outer-start-year",
        type=int,
        default=2023,
        help="First OOT calendar fold for an ablation experiment.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel tasks for an ablation experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    horizons = tuple(args.horizons) if args.horizons is not None else HORIZONS

    if args.as_of is None and args.experiment == "nested":
        run_backtest(root)
        return

    if args.as_of is None and args.experiment == "training-windows":
        run_training_window_experiment(
            root,
            horizons,
            args.outer_start_year,
            args.workers,
        )
        return


    if args.as_of is None and args.experiment == "candidate-gate":
        run_candidate_gate_experiment(
            root,
            horizons,
            args.outer_start_year,
            args.workers,
        )
        return

    if args.as_of is None and args.experiment == "binary-truth":
        run_candidate_gate_experiment(
            root,
            horizons,
            args.outer_start_year,
            args.workers,
            gates=("all_days",),
            output_subdir="ml_binary_truth",
            experiment_label="all-days binary truth_now nested walk-forward",
        )
        return


    if args.as_of is None and args.experiment == "regret-regression":
        run_regret_regression_experiment(
            root,
            horizons,
            args.outer_start_year,
            args.workers,
        )
        return

    if args.as_of is None and args.experiment == "joint-policy":
        run_joint_policy_experiment(
            root,
            horizons,
            args.outer_start_year,
            args.workers,
        )
        return

    if args.as_of is None and args.experiment == "horizon-tuning":
        run_horizon_tuning_experiment(
            root,
            horizons,
            args.outer_start_year,
            args.workers,
        )
        return

    if args.as_of is None and args.experiment == "nightly-tuning":
        nightly_horizons = (
            tuple(args.horizons)
            if args.horizons is not None
            else NIGHTLY_HORIZONS
        )
        run_horizon_tuning_experiment(
            root,
            nightly_horizons,
            args.outer_start_year,
            args.workers,
            model_configs=NIGHTLY_MODEL_CONFIGS,
            score_lookbacks=NIGHTLY_SCORE_LOOKBACKS,
            policy_quantiles=NIGHTLY_POLICY_QUANTILES,
            output_subdir="ml_nightly_tuning",
            resume=True,
        )
        return

    if args.as_of is None and args.experiment == "cross-h-reproduction":
        run_cross_h_reproduction(root, args.outer_start_year)
        return

    if args.as_of is None and args.experiment == "cross-h-multi-eval":
        run_cross_h_multi_evaluation(root)
        return

    if args.as_of is None and args.experiment == "calendar-reminders":
        run_calendar_reminders(root)
        return

    if args.as_of is None and args.experiment == "waiting-cost":
        run_waiting_cost_analysis(root)
        return

    prices = load_ml_prices(latest_default_input())
    if args.experiment == "cross-h-reproduction":
        result = cross_h_signals_as_of_date(prices, pd.Timestamp(args.as_of))
        output_name = f"cross_h_signals_as_of_{args.as_of}.csv"
    else:
        result = signals_as_of_date(prices, pd.Timestamp(args.as_of), args.horizon)
        output_name = f"signals_as_of_{args.as_of}_h{args.horizon}.csv"
    output_dir = root / "data/processed/ml_signals"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False) if not result.empty else "No signals on this as-of date.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
