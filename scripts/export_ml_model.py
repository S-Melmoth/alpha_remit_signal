#!/usr/bin/env python3
"""Train and freeze the selected yearly ML push policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from backtest_hard_signals import latest_default_input
from backtest_ml_signals import load_ml_prices
from ml_production import DEFAULT_BUNDLE, save_bundle, train_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--policy-year", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()

    input_path = args.input or latest_default_input()
    bundle = train_bundle(
        load_ml_prices(input_path),
        policy_year=args.policy_year,
        input_path=input_path,
    )
    save_bundle(bundle, args.output)
    manifest = bundle["manifest"]
    print(f"Saved: {args.output}")
    print(
        f"Policy {manifest['policy_year']}; train rows={manifest['training_rows']}; "
        f"latest mature date={manifest['latest_mature_training_date']}"
    )


if __name__ == "__main__":
    main()
