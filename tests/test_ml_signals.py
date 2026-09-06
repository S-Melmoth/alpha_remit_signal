import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backtest_ml_signals import (  # noqa: E402
    SELECTION_COOLDOWN,
    attach_target_stream_frequencies,
    apply_observation_cooldown,
    build_features,
    build_labels,
    candidate_gate_mask,
    causal_score_percentiles,
    choose_local_only_thresholds,
    choose_regret_thresholds,
    choose_truth_only_thresholds,
    evaluate_signal_stream_across_horizons,
    feature_columns,
    load_brent_features,
    load_ml_prices,
    mature_validation_rows,
    model_train_start,
    recalibrate_thresholds,
    recipient_holiday_features,
    recency_weights,
    refit_selected_policy,
    truth_threshold_curve,
    joint_policy_mask,
    union_signal_streams,
)


class MlSignalTest(unittest.TestCase):
    @staticmethod
    def synthetic_prices(periods: int = 160) -> pd.DataFrame:
        dates = pd.bdate_range("2023-01-02", periods=periods)
        rows = []
        currencies = ("AMD", "KGS", "KZT", "TJS", "UZS", "USD", "EUR", "CNY")
        for offset, currency in enumerate(currencies):
            for index, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "currency": currency,
                        "rub_per_unit": 10 + offset + index / 100,
                    }
                )
        return pd.DataFrame(rows)

    def test_features_at_t_do_not_change_with_future_prices(self) -> None:
        prices = self.synthetic_prices()
        dates = sorted(prices["date"].unique())
        cutoff = pd.Timestamp(dates[130])
        before = build_features(prices).set_index(["currency", "date"])

        changed = prices.copy()
        changed.loc[changed["date"].gt(cutoff), "rub_per_unit"] *= 10
        after = build_features(changed).set_index(["currency", "date"])

        pd.testing.assert_series_equal(
            before.loc[("AMD", cutoff)],
            after.loc[("AMD", cutoff)],
        )

    def test_brent_is_not_available_before_conservative_lag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brent.csv"
            path.write_text(
                "observation_date,DCOILBRENTEU\n2024-01-01,80\n2024-01-02,90\n"
            )
            dates = pd.date_range("2024-01-01", "2024-01-06")
            features = load_brent_features(path, dates)
            self.assertTrue(
                pd.isna(features.loc[pd.Timestamp("2024-01-03"), "brent_return_lag_0"])
            )
            self.assertAlmostEqual(
                features.loc[pd.Timestamp("2024-01-05"), "brent_return_lag_0"],
                0.1173,
                places=3,
            )

    def test_ml_loader_keeps_target_and_context_currencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            self.synthetic_prices(periods=3).to_csv(path, index=False)
            loaded = load_ml_prices(path)
            self.assertEqual(
                set(loaded["currency"]),
                {"AMD", "KGS", "KZT", "TJS", "UZS", "USD", "EUR", "CNY"},
            )

    def test_constant_missing_and_observation_index_are_removed(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "currency": ["AMD"] * 3,
                "truth_now": [0, 1, 0],
                "local_min": [0, 1, 0],
                "benefit_bps": [1.0, 2.0, 3.0],
                "future_regret_bps": [0.0, 1.0, 2.0],
                "label_available_date": pd.date_range("2024-01-03", periods=3),
                "valid": [True] * 3,
                "varying": [0.0, 1.0, 2.0],
                "constant": [1.0] * 3,
                "missing": [float("nan")] * 3,
                "observation_index": [10, 11, 12],
            }
        )
        self.assertEqual(feature_columns(frame), ["varying"])

    def test_cooldown_uses_market_observation_index(self) -> None:
        frame = pd.DataFrame(
            {
                "currency": ["AMD"] * 4,
                "date": pd.date_range("2024-01-01", periods=4),
                "observation_index": [10, 11, 14, 15],
            }
        )
        selected = apply_observation_cooldown(frame, cooldown=3)
        self.assertEqual(selected["observation_index"].tolist(), [10, 14])
        self.assertEqual(SELECTION_COOLDOWN, 0)

    def test_recency_weight_decays_into_the_past(self) -> None:
        dates = pd.Series(pd.to_datetime(["2023-01-01", "2024-01-01"]))
        weights = recency_weights(dates, pd.Timestamp("2025-01-01"), 365)
        self.assertLess(weights[0], weights[1])

    def test_local_min_is_stricter_than_forward_truth(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=12)
        rows = []
        for offset, currency in enumerate(("AMD", "KGS", "KZT", "TJS", "UZS")):
            for index, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "currency": currency,
                        "rub_per_unit": 10 + offset + abs(index - 5),
                    }
                )
        labels = build_labels(pd.DataFrame(rows), horizon=2).dropna()
        self.assertTrue(labels["local_min"].le(labels["truth_now"]).all())

    def test_multi_horizon_evaluation_uses_identical_signal_dates(self) -> None:
        prices = self.synthetic_prices(periods=80)
        target_dates = pd.bdate_range("2023-02-01", periods=4)
        signals = pd.DataFrame(
            [
                {
                    "date": date,
                    "currency": currency,
                    "observation_index": index,
                }
                for currency in ("AMD", "KGS", "KZT", "TJS", "UZS")
                for index, date in enumerate(target_dates, start=20)
            ]
        )
        metrics, summary, common_end = evaluate_signal_stream_across_horizons(
            signals,
            prices,
            pd.Timestamp("2023-01-01"),
            (1, 3, 5, 10, 20),
        )
        self.assertEqual(metrics.groupby("horizon")["signal_count"].sum().nunique(), 1)
        self.assertEqual(set(summary["horizon"]), {1, 3, 5, 10, 20})
        self.assertLessEqual(common_end, prices["date"].max() + pd.Timedelta(days=1))

    def test_inner_validation_drops_labels_that_mature_after_outer_start(self) -> None:
        outer_start = pd.Timestamp("2024-01-01")
        validation_start = pd.Timestamp("2023-01-01")
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-12-20", "2023-12-21", "2023-12-22"]),
                "label_available_date": pd.to_datetime(
                    ["2023-12-28", "2024-01-02", "2024-01-20"]
                ),
                "valid": [True, True, True],
            }
        )
        eligible = mature_validation_rows(frame, validation_start, outer_start)
        self.assertEqual(eligible["date"].tolist(), [pd.Timestamp("2023-12-20")])
        self.assertTrue(eligible["label_available_date"].lt(outer_start).all())

    def test_level_percentile_uses_exactly_previous_window_observations(self) -> None:
        prices = self.synthetic_prices(periods=140)
        features = build_features(prices)
        amd_prices = (
            prices.loc[prices["currency"].eq("AMD")]
            .sort_values("date")
            .reset_index(drop=True)
        )
        target_index = 80
        target_date = amd_prices.loc[target_index, "date"]
        previous_20 = amd_prices.loc[target_index - 20 : target_index - 1, "rub_per_unit"]
        current = amd_prices.loc[target_index, "rub_per_unit"]
        expected = float(np.mean(previous_20.to_numpy() >= current))
        actual = features.loc[
            features["currency"].eq("AMD") & features["date"].eq(target_date),
            "level_percentile_20",
        ].iloc[0]
        self.assertAlmostEqual(actual, expected, places=12)

    def test_days_since_holiday_is_zero_on_the_holiday(self) -> None:
        dates = pd.date_range("2024-01-01", "2024-01-03")
        features = recipient_holiday_features(dates, "AM")
        self.assertEqual(features.loc[pd.Timestamp("2024-01-01"), "recipient_holiday_today"], 1.0)
        self.assertEqual(features.loc[pd.Timestamp("2024-01-01"), "days_since_recipient_holiday"], 0)

    def test_threshold_curve_uses_inner_validation_cutoffs(self) -> None:
        reference = pd.DataFrame(
            {
                "currency": ["AMD"] * 5,
                "p_truth_now": [0.0, 0.25, 0.5, 0.75, 1.0],
            }
        )
        outer = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-01", periods=5),
                "currency": ["AMD"] * 5,
                "p_truth_now": [0.90, 0.91, 0.92, 0.93, 0.94],
                "truth_now": [0.0, 0.0, 1.0, 1.0, 1.0],
                "local_min": [0.0, 0.0, 0.0, 1.0, 1.0],
                "benefit_bps": [-2.0, -1.0, 0.0, 1.0, 2.0],
            }
        )
        curve = truth_threshold_curve(
            outer,
            reference,
            horizon=1,
            fold_start=pd.Timestamp("2024-01-01"),
            fold_end=pd.Timestamp("2024-02-01"),
        )
        row = curve.loc[curve["score_quantile"].eq(0.80)].iloc[0]
        self.assertAlmostEqual(row["p_truth_threshold"], 0.80)
        self.assertEqual(row["threshold_source"], "inner_validation")

    def test_training_window_start_is_relative_to_current_cutoff(self) -> None:
        cutoff = pd.Timestamp("2025-01-01")
        self.assertEqual(model_train_start("rolling_1y", cutoff), pd.Timestamp("2024-01-01"))
        self.assertEqual(model_train_start("rolling_6y", cutoff), pd.Timestamp("2019-01-01"))
        self.assertEqual(model_train_start("long_history", cutoff), pd.Timestamp("2014-01-01"))

    def test_quantile_policy_is_remapped_after_refit(self) -> None:
        reference = pd.DataFrame(
            {
                "currency": ["AMD"] * 5,
                "p_truth_now": [0.1, 0.2, 0.3, 0.4, 0.5],
                "p_local_min": [0.0, 0.25, 0.5, 0.75, 1.0],
                "predicted_benefit_bps": [-10.0, 0.0, 10.0, 20.0, 30.0],
            }
        )
        policy = {
            "AMD": {
                "truth_quantile": 0.8,
                "local_quantile": 0.5,
                "benefit_quantile": 0.3,
            }
        }
        thresholds = recalibrate_thresholds(policy, reference)
        self.assertAlmostEqual(thresholds["AMD"]["truth_threshold"], 0.42)
        self.assertAlmostEqual(thresholds["AMD"]["local_threshold"], 0.50)
        self.assertAlmostEqual(thresholds["AMD"]["benefit_threshold"], 2.0)

    def test_refit_uses_all_mature_rows_up_to_outer_start(self) -> None:
        dates = pd.bdate_range("2021-01-01", "2023-12-29")
        data = pd.DataFrame(
            {
                "date": dates,
                "currency": "AMD",
                "label_available_date": dates,
                "valid": True,
                "truth_now": np.arange(len(dates)) % 2,
                "local_min": np.arange(len(dates)) % 3 == 0,
                "benefit_bps": np.arange(len(dates), dtype=float),
                "x": np.arange(len(dates), dtype=float),
            }
        )
        reference_scored = data.loc[data["date"].ge("2023-01-01")].copy()
        reference_scored["p_truth_now"] = np.linspace(0.1, 0.9, len(reference_scored))
        reference_scored["p_local_min"] = np.linspace(0.2, 0.8, len(reference_scored))
        reference_scored["predicted_benefit_bps"] = 1.0
        policy = {
            "AMD": {
                "truth_quantile": 0.5,
                "local_quantile": 0.5,
                "benefit_quantile": 0.0,
            }
        }
        fake_models = (object(), object(), object())
        with patch("backtest_ml_signals.fit_models", return_value=fake_models) as fit_mock, patch(
            "backtest_ml_signals.score_models", return_value=reference_scored
        ):
            models, _, _ = refit_selected_policy(
                data,
                ["x"],
                pd.Timestamp("2024-01-01"),
                "rolling_4y",
                policy,
            )
        fitted_rows = fit_mock.call_args.args[0]
        self.assertIs(models, fake_models)
        self.assertEqual(fitted_rows["date"].max(), pd.Timestamp("2023-12-29"))
        self.assertGreaterEqual(fitted_rows["date"].min(), pd.Timestamp("2020-01-01"))

    def test_candidate_gate_is_a_trailing_level_condition(self) -> None:
        frame = pd.DataFrame({"level_percentile_60": [0.69, 0.70, 0.80, 0.91, np.nan]})
        self.assertEqual(
            candidate_gate_mask(frame, "level60_q80").tolist(),
            [False, False, True, True, False],
        )
        self.assertTrue(candidate_gate_mask(frame, "all_days").all())

    def test_candidate_threshold_lift_uses_all_days_random_baseline(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=40)
        pool = pd.DataFrame(
            {
                "date": dates,
                "currency": "AMD",
                "truth_now": [1.0] * 20 + [0.0] * 20,
                "local_min": [0.0] * 40,
                "benefit_bps": [5.0] * 40,
            }
        )
        candidates = pool.iloc[:20].copy()
        candidates["p_truth_now"] = np.linspace(0.1, 0.9, 20)
        choices, diagnostics = choose_truth_only_thresholds(
            candidates,
            pool,
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-04-01"),
        )
        amd = diagnostics.loc[diagnostics["currency"].eq("AMD")].iloc[0]
        self.assertAlmostEqual(amd["truth_lift"], 2.0)
        self.assertEqual(choices["AMD"]["truth_quantile"], 0.0)

    def test_target_stream_frequencies_are_reported_separately(self) -> None:
        metrics = pd.DataFrame({"currency": ["AMD", "KGS"]})
        truth = pd.DataFrame({"currency": ["AMD", "AMD", "KGS"]})
        local = pd.DataFrame({"currency": ["AMD", "KGS", "KGS"]})
        joint = pd.DataFrame({"currency": ["AMD"]})
        result = attach_target_stream_frequencies(
            metrics,
            truth,
            local,
            joint,
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-08"),
        ).set_index("currency")
        self.assertEqual(result.loc["AMD", "truth_stream_count"], 2)
        self.assertEqual(result.loc["AMD", "local_stream_count"], 1)
        self.assertEqual(result.loc["AMD", "target_intersection_count"], 1)
        self.assertEqual(result.loc["KGS", "target_intersection_count"], 0)

    def test_local_threshold_selection_uses_all_days_baseline(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=40)
        pool = pd.DataFrame(
            {
                "date": dates,
                "currency": "AMD",
                "truth_now": [0.0] * 40,
                "local_min": [1.0] * 20 + [0.0] * 20,
                "benefit_bps": [5.0] * 40,
            }
        )
        candidates = pool.iloc[:20].copy()
        candidates["p_local_min"] = np.linspace(0.1, 0.9, 20)
        choices, diagnostics = choose_local_only_thresholds(
            candidates,
            pool,
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-04-01"),
        )
        amd = diagnostics.loc[diagnostics["currency"].eq("AMD")].iloc[0]
        self.assertAlmostEqual(amd["local_lift"], 2.0)
        self.assertEqual(choices["AMD"]["local_quantile"], 0.0)

    def test_future_regret_zero_is_exact_truth_now(self) -> None:
        labels = build_labels(self.synthetic_prices(), horizon=3)
        valid = labels.loc[labels["valid"]]
        self.assertTrue(
            valid["truth_now"].eq(valid["future_regret_bps"].eq(0).astype(float)).all()
        )

    def test_causal_score_percentile_uses_only_earlier_scores(self) -> None:
        reference = pd.DataFrame(
            {
                "date": pd.date_range("2023-12-01", periods=3),
                "currency": ["AMD"] * 3,
                "score": [1.0, 2.0, 3.0],
            }
        )
        target = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=2),
                "currency": ["AMD"] * 2,
                "score": [4.0, 0.0],
            }
        )
        ranked = causal_score_percentiles(reference, target, ("score",))
        self.assertEqual(ranked["score_rank"].tolist(), [1.0, 0.0])

    def test_joint_policy_masks_match_the_two_architectures(self) -> None:
        scored = pd.DataFrame(
            {
                "p_truth_now_rank": [0.9, 0.9, 0.4],
                "predicted_benefit_q25_rank": [0.9, 0.4, 0.9],
                "predicted_benefit_q25": [5.0, 5.0, -1.0],
                "p_good_push_rank": [0.8, 0.6, 0.9],
            }
        )
        two_head = joint_policy_mask(
            scored,
            "two_head",
            truth_quantile=0.8,
            benefit_quantile=0.8,
        )
        direct = joint_policy_mask(
            scored, "joint_classifier", joint_quantile=0.8
        )
        self.assertEqual(two_head.tolist(), [True, False, False])
        self.assertEqual(direct.tolist(), [True, False, True])

    def test_regret_threshold_selects_low_predictions_against_all_days(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=40)
        pool = pd.DataFrame(
            {
                "date": dates,
                "currency": "AMD",
                "truth_now": [1.0] * 20 + [0.0] * 20,
                "local_min": [0.0] * 40,
                "benefit_bps": [5.0] * 40,
            }
        )
        scored = pool.copy()
        scored["predicted_future_regret_bps"] = [0.0] * 20 + [100.0] * 20
        choices, diagnostics = choose_regret_thresholds(
            scored,
            pool,
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-04-01"),
        )
        amd = diagnostics.loc[diagnostics["currency"].eq("AMD")].iloc[0]
        self.assertAlmostEqual(amd["truth_lift"], 2.0)
        self.assertEqual(choices["AMD"]["regret_threshold_bps"], 0.0)

    def test_union_stream_emits_currency_date_once(self) -> None:
        first = pd.DataFrame(
            {
                "currency": ["AMD", "AMD"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "value": [1, 2],
            }
        )
        second = pd.DataFrame(
            {
                "currency": ["AMD", "KGS"],
                "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "value": [2, 3],
            }
        )
        union = union_signal_streams(first, second)
        self.assertEqual(len(union), 3)
        self.assertFalse(union.duplicated(["currency", "date"]).any())


if __name__ == "__main__":
    unittest.main()
