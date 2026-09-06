"""Frozen-model training and causal inference for production FX pushes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from backtest_ml_signals import (
    CROSS_H_EVAL_HORIZON,
    CROSS_H_LEGACY_CHOICES,
    CROSS_H_TRAIN_HORIZON,
    NIGHTLY_MODEL_CONFIGS,
    TARGET_CURRENCIES,
    add_message_facts,
    apply_ml_communication_limits,
    build_features,
    build_labels,
    calendar_reminders_as_of_date,
    causal_score_percentiles,
    feature_columns,
    make_tuned_good_push_model,
    mature_training_rows,
    model_train_start,
    render_ml_pushes,
)

BUNDLE_FORMAT_VERSION = 1
DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "models" / "ml_good_push_2026.joblib"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_bundle(
    prices: pd.DataFrame,
    *,
    policy_year: int,
    input_path: Path | None = None,
) -> dict[str, Any]:
    """Fit the frozen yearly policy using only labels mature before the year."""
    if policy_year not in CROSS_H_LEGACY_CHOICES:
        raise ValueError(f"No frozen policy for {policy_year}")

    fold_start = pd.Timestamp(f"{policy_year}-01-01")
    choice = CROSS_H_LEGACY_CHOICES[policy_year]
    config_name = str(choice["model_config"])
    config = NIGHTLY_MODEL_CONFIGS[config_name]
    lookback = int(choice["score_lookback"])
    quantile = float(choice["joint_quantile"])

    features = build_features(prices)
    data = features.merge(
        build_labels(prices, CROSS_H_TRAIN_HORIZON),
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

    manifest: dict[str, Any] = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "model_version": f"ml-good-push-{policy_year}-v1",
        "model_family": "sklearn.HistGradientBoostingClassifier",
        "policy_year": policy_year,
        "training_cutoff_exclusive": str(fold_start.date()),
        "training_rows": int(len(train)),
        "latest_mature_training_date": str(pd.Timestamp(train["date"].max()).date()),
        "train_horizon": CROSS_H_TRAIN_HORIZON,
        "evaluation_horizon": CROSS_H_EVAL_HORIZON,
        "target": "truth_now AND benefit_bps > 0",
        "model_config": config_name,
        "model_parameters": dict(config),
        "score_lookback": lookback,
        "joint_quantile": quantile,
        "feature_columns": columns,
        "target_currencies": list(TARGET_CURRENCIES),
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "input_path": str(input_path) if input_path is not None else None,
        "input_sha256": _sha256(input_path) if input_path is not None else None,
    }
    return {
        "manifest": manifest,
        "model": model,
        # Percentile calibration is part of the policy, not optional diagnostics.
        "reference_scores": reference[["date", "currency", "p_good_push"]].copy(),
    }


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(path)
    manifest_path = path.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(bundle["manifest"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_bundle(path: Path = DEFAULT_BUNDLE) -> dict[str, Any]:
    bundle = joblib.load(path)
    manifest = bundle.get("manifest", {})
    if manifest.get("bundle_format_version") != BUNDLE_FORMAT_VERSION:
        raise ValueError("Unsupported ML bundle format")
    if manifest.get("sklearn_version") != sklearn.__version__:
        raise ValueError(
            "ML bundle requires scikit-learn "
            f"{manifest.get('sklearn_version')}; installed {sklearn.__version__}"
        )
    return bundle


def candidates_from_bundle(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """Score all published dates in the bundle year without fitting a model."""
    as_of = pd.Timestamp(as_of).normalize()
    manifest = bundle["manifest"]
    policy_year = int(manifest["policy_year"])
    if as_of.year != policy_year:
        raise ValueError(
            f"Bundle is frozen for {policy_year}; requested as-of belongs to {as_of.year}"
        )

    causal_prices = prices.loc[prices["date"].le(as_of)].copy()
    features = build_features(causal_prices)
    columns = list(manifest["feature_columns"])
    missing = sorted(set(columns).difference(features.columns))
    if missing:
        raise ValueError(f"Input series cannot produce model features: {missing}")

    fold_start = pd.Timestamp(manifest["training_cutoff_exclusive"])
    score_stream = features.loc[
        features["date"].ge(fold_start) & features["date"].le(as_of)
    ].copy()
    if score_stream.empty:
        return score_stream
    score_stream["p_good_push"] = bundle["model"].predict_proba(
        score_stream[columns]
    )[:, 1]
    ranked = causal_score_percentiles(
        bundle["reference_scores"],
        score_stream,
        ("p_good_push",),
        lookback=int(manifest["score_lookback"]),
    )
    selected = ranked.loc[
        ranked["p_good_push_rank"].ge(float(manifest["joint_quantile"]))
    ].copy()
    if selected.empty:
        return selected
    selected = add_message_facts(selected)
    selected["train_horizon"] = int(manifest["train_horizon"])
    selected["evaluation_horizon"] = int(manifest["evaluation_horizon"])
    selected["scenario"] = "favourable_now"
    selected["direction"] = "lower_rub_per_unit_is_better"
    selected["strength"] = selected["p_good_push_rank"]
    selected["speed"] = selected["return_mean_3"] * 10_000
    selected["model_config"] = str(manifest["model_config"])
    selected["score_lookback"] = int(manifest["score_lookback"])
    selected["joint_quantile"] = float(manifest["joint_quantile"])
    return selected.sort_values(["date", "currency"]).reset_index(drop=True)


def signals_from_bundle(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    bundle: dict[str, Any],
    *,
    include_calendar_reminders: bool = False,
    prefill_available: bool = False,
) -> pd.DataFrame:
    """Produce today's thresholded ML pushes and optional neutral reminders."""
    as_of = pd.Timestamp(as_of).normalize()
    candidates = candidates_from_bundle(prices, as_of, bundle)
    sent = (
        apply_ml_communication_limits(
            candidates, max_pushes_per_week=2, min_gap_calendar_days=1
        )
        if not candidates.empty
        else candidates
    )
    ml_today = sent.loc[sent["date"].eq(as_of)].copy()
    rendered = render_ml_pushes(ml_today, prices.loc[prices["date"].le(as_of)])
    streams = [rendered] if not rendered.empty else []
    if include_calendar_reminders:
        reminders = calendar_reminders_as_of_date(
            prices, as_of, prefill_available=prefill_available
        )
        if not reminders.empty:
            streams.append(reminders)
    if not streams:
        return rendered
    return pd.concat(streams, ignore_index=True).sort_values(
        ["date", "corridor", "signal_source"]
    ).reset_index(drop=True)
