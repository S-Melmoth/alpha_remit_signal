#!/usr/bin/env python3
"""Generate the final causal communication stream for an arbitrary cutoff."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib import request

import pandas as pd

from backtest_hard_signals import latest_default_input
from backtest_ml_signals import load_ml_prices
from ml_production import DEFAULT_BUNDLE, load_bundle, signals_from_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ML financial pushes and neutral 5/20 reminders using only "
            "rates available by the requested date."
        )
    )
    parser.add_argument(
        "--as-of",
        required=True,
        help="Decision cutoff in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV with date,currency,rub_per_unit. Defaults to the latest fixed snapshot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV. Defaults to data/processed/production_signals_<date>.csv.",
    )
    parser.add_argument(
        "--prefill-available",
        action="store_true",
        help="Use prefilled 5/20 copy. Without this flag the reminder opens a plain form.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="Frozen joblib model bundle.",
    )
    parser.add_argument(
        "--publish-url",
        default=None,
        help="Optional service endpoint, e.g. http://localhost:8080/api/ml-signals.",
    )
    parser.add_argument(
        "--ingest-token",
        default=None,
        help="Service token. Defaults to ML_INGEST_TOKEN.",
    )
    return parser.parse_args()


def publish_ml_signals(
    signals: pd.DataFrame,
    url: str,
    token: str,
    model_version: str,
) -> dict[str, object]:
    """Send only financial ML rows; calendar reminders remain a separate CRM flow."""
    ml = signals.loc[signals["signal_source"].eq("ml")].copy()
    records = json.loads(ml.to_json(orient="records", date_format="iso"))
    body = json.dumps(
        {"model_version": model_version, "signals": records}, ensure_ascii=False
    ).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    with request.urlopen(req, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    as_of = pd.Timestamp(args.as_of).normalize()
    input_path = args.input or latest_default_input()
    output_path = args.output or (
        root / "data" / "processed" / f"production_signals_{as_of.date()}.csv"
    )

    prices = load_ml_prices(input_path)
    bundle = load_bundle(args.model)
    signals = signals_from_bundle(
        prices,
        as_of,
        bundle,
        include_calendar_reminders=True,
        prefill_available=bool(args.prefill_available),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output_path, index=False)

    print(signals.to_string(index=False) if not signals.empty else "No signals on this date.")
    print(f"Saved: {output_path}")
    if args.publish_url:
        token = args.ingest_token or os.environ.get("ML_INGEST_TOKEN", "")
        if not token:
            raise SystemExit("--publish-url requires --ingest-token or ML_INGEST_TOKEN")
        result = publish_ml_signals(
            signals,
            args.publish_url,
            token,
            str(bundle["manifest"]["model_version"]),
        )
        print(f"Published: {result}")


if __name__ == "__main__":
    main()
