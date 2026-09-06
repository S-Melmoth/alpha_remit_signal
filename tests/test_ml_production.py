import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_ml_signals import load_ml_prices  # noqa: E402
from ml_production import DEFAULT_BUNDLE, load_bundle, signals_from_bundle  # noqa: E402


class FrozenMLProductionTest(unittest.TestCase):
    def test_committed_bundle_scores_without_refitting(self) -> None:
        bundle = load_bundle(DEFAULT_BUNDLE)
        manifest = bundle["manifest"]
        self.assertEqual(manifest["model_version"], "ml-good-push-2026-v1")
        self.assertEqual(manifest["training_cutoff_exclusive"], "2026-01-01")
        self.assertLess(manifest["latest_mature_training_date"], "2026-01-01")

        prices = load_ml_prices(
            ROOT / "data" / "processed" / "cbr_fx_daily_2010-01-01_2026-09-02.csv"
        )
        with patch(
            "sklearn.ensemble.HistGradientBoostingClassifier.fit",
            side_effect=AssertionError("inference must not fit"),
        ):
            signals = signals_from_bundle(
                prices, pd.Timestamp("2026-08-04"), bundle
            )
        self.assertEqual(signals["currency"].tolist(), ["KGS", "TJS"])


if __name__ == "__main__":
    unittest.main()
