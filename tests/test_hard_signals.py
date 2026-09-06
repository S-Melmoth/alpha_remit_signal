import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backtest_hard_signals import (  # noqa: E402
    Candidate,
    build_level_corridor_signals,
    build_hard_indicator_corridor_report,
    build_signals,
    compute_candidate,
    evaluate_strategies,
    evaluate_level_corridor,
    excluded_momentum_audit,
    level_persistence_outcomes,
    monthly_block_bootstrap_mean,
    outcome_frame,
    scenario_outcomes,
    screen_walk_forward_families,
    screen_fixed_level_policies,
    signals_as_of,
    thin_with_cooldown,
)


class HardSignalTest(unittest.TestCase):
    def test_final_hard_report_separates_trigger_and_message_roles(self) -> None:
        common = {
            "currency": ["AMD", "AMD"],
            "horizon": [2, 3],
            "hit_rate": [0.50, 0.45],
            "random_hit_rate_mean": [0.40, 0.40],
            "benefit_bps": [20.0, 25.0],
            "benefit_p_value_vs_zero": [0.01, 0.02],
        }
        spike = pd.DataFrame(common).assign(hit_lift=[1.25, 1.125])
        report = build_hard_indicator_corridor_report(
            spike,
            pd.DataFrame(),
            currencies=("AMD",),
            horizons=(2, 3),
        ).iloc[0]

        self.assertFalse(bool(report["standalone_trigger"]))
        self.assertTrue(bool(report["ml_message_fact_eligible"]))
        self.assertIn("lift ниже 1.3", report["reason"])

    def test_hard_screen_requires_every_currency_horizon_but_ignores_frequency(self) -> None:
        rows = []
        for currency in ("AMD", "KGS"):
            for horizon in (2, 3):
                rows.append(
                    {
                        "currency": currency,
                        "horizon": horizon,
                        "hit_lift": 1.25,
                        "hit_rate": 0.50,
                        "random_hit_rate_mean": 0.40,
                        "benefit_bps": 0.0,
                        "signals_per_week": 0.01,
                    }
                )
        alternative = pd.DataFrame(rows).assign(signal_family="level")
        result = screen_walk_forward_families(
            pd.DataFrame(),
            alternative,
            currencies=("AMD", "KGS"),
            horizons=(2, 3),
        ).iloc[0]
        self.assertTrue(bool(result["passes_hard_screen"]))
        self.assertEqual(result["evaluated_cells"], 4)

    def test_fixed_level_screen_distinguishes_exploratory_and_case_rules(self) -> None:
        rows = []
        for currency in ("AMD", "KGS"):
            for horizon in (2, 3):
                rows.append(
                    {
                        "signal_family": "level",
                        "strategy_id": "level_L20_p90",
                        "parameters": '{"higher_share": 0.9, "lookback": 20}',
                        "currency": currency,
                        "horizon": horizon,
                        "hit_lift": 1.25,
                        "benefit_bps": 0.0,
                        "benefit_ci_low_95": -2.0,
                    }
                )
        result = screen_fixed_level_policies(
            pd.DataFrame(rows), currencies=("AMD", "KGS"), horizons=(2, 3)
        ).iloc[0]
        self.assertTrue(bool(result["passes_hard_screen"]))
        self.assertFalse(bool(result["passes_case_screen"]))

    def test_level_uses_only_prior_observations(self) -> None:
        prices = pd.Series([10.0, 11.0, 12.0, 9.0, 1000.0])
        candidate = Candidate("level", lookback=3, threshold=0.9)
        before_future_change = compute_candidate(prices, candidate).iloc[3]
        prices.iloc[4] = 0.01
        after_future_change = compute_candidate(prices, candidate).iloc[3]
        self.assertEqual(before_future_change["indicator_value"], 1.0)
        self.assertEqual(
            before_future_change["indicator_value"], after_future_change["indicator_value"]
        )
        self.assertTrue(bool(before_future_change["raw_signal"]))

    def test_stable_range_is_trailing_and_scale_free(self) -> None:
        prices = pd.Series([100.0, 100.2, 99.9, 100.1])
        candidate = Candidate("stable", lookback=4, threshold=50.0)
        result = compute_candidate(prices, candidate).iloc[-1]
        self.assertLess(result["indicator_value"], 50.0)
        self.assertTrue(bool(result["raw_signal"]))

    def test_downward_spike_is_favourable(self) -> None:
        prices = pd.Series([100.0, 99.0, 97.0])
        candidate = Candidate("spike", lookback=2, threshold=200.0)
        result = compute_candidate(prices, candidate).iloc[-1]
        self.assertEqual(result["indicator_value"], 2.0)
        self.assertGreater(result["strength"], 300.0)
        self.assertTrue(bool(result["raw_signal"]))

    def test_spike_requires_consecutive_declines(self) -> None:
        prices = pd.Series([100.0, 98.0, 101.0, 97.0])
        candidate = Candidate("spike", lookback=3, threshold=0.0)
        result = compute_candidate(prices, candidate).iloc[-1]
        self.assertGreater(result["strength"], 300.0)
        self.assertEqual(result["indicator_value"], 1.0)
        self.assertFalse(bool(result["raw_signal"]))

    def test_longer_momentum_streaks_are_less_frequent(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=9)
        prices = pd.DataFrame(
            {
                "date": dates,
                "currency": "AMD",
                "rub_per_unit": [10.0, 9.0, 10.0, 9.0, 8.0, 10.0, 9.0, 8.0, 7.0],
            }
        )
        signals, _ = build_signals(prices, dates[0], cooldown_observations=3)
        counts = signals.groupby("strategy_id").size()
        self.assertEqual(counts["momentum_streak_2d__spike_only"], 2)
        self.assertEqual(counts["momentum_streak_3d__spike_only"], 1)

    def test_shared_cooldown_is_applied_after_union(self) -> None:
        level = pd.Series([True, False, False, False, False, False])
        spike = pd.Series([False, True, False, False, True, False])
        combined = level | spike
        self.assertEqual(thin_with_cooldown(combined, cooldown_observations=3), [0, 4])

    def test_label_availability_is_end_of_forward_horizon(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        outcomes = outcome_frame(pd.Series(range(1, 7), dtype=float), dates, horizon=2)
        self.assertEqual(outcomes.loc[dates[1], "label_available_date"], dates[3])

    def test_benefit_matches_case_symmetric_window_formula(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        outcomes = outcome_frame(pd.Series([100.0, 80.0, 100.0]), dates, horizon=1)
        expected = (((100.0 + 80.0 + 100.0) / 3) / 80.0 - 1.0) * 10_000
        self.assertAlmostEqual(outcomes.loc[dates[1], "benefit_bps"], expected)

    def test_benefit_inference_tests_mean_against_zero(self) -> None:
        dates = pd.date_range("2022-01-31", periods=24, freq="ME")
        positive = pd.Series([10.0, 20.0] * 12, index=dates)
        result = monthly_block_bootstrap_mean(positive, repeats=300, seed=42)
        self.assertGreater(result["ci_low"], 0.0)
        self.assertLess(result["p_value"], 0.05)

    def test_level_hit_requires_literal_condition_through_horizon(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="D")
        series = pd.Series([10.0, 11.0, 9.0, 8.0, 7.0, 12.0, 6.0])
        outcomes = outcome_frame(series, dates, horizon=1)
        parameters = '{"higher_share": 1.0, "lookback": 2}'
        literal = level_persistence_outcomes(
            outcomes, series, dates, parameters, horizon=1
        )
        self.assertEqual(literal.loc[dates[3], "target_hit"], 1.0)
        self.assertEqual(literal.loc[dates[4], "target_hit"], 0.0)
        self.assertEqual(
            literal.loc[dates[3], "target_definition"],
            "level_condition_holds_at_T_and_every_observation_through_T_plus_h",
        )

    def test_level_evaluation_uses_case_favourable_now_hit(self) -> None:
        dates = pd.date_range("2024-01-01", periods=7, freq="D")
        prices = pd.DataFrame(
            {
                "date": dates,
                "currency": "AMD",
                "rub_per_unit": [12.0, 11.0, 10.0, 9.0, 8.0, 10.0, 11.0],
            }
        )
        signals = pd.DataFrame(
            {
                "date": [dates[3]],
                "currency": ["AMD"],
                "strategy_id": ["level_L2_p50"],
                "signal_family": ["level"],
                "scenario": ["favourable_now"],
                "parameters": ['{"higher_share": 0.5, "lookback": 2}'],
            }
        )
        metrics = evaluate_level_corridor(
            prices, signals, dates[0], (1,), repeats=20, seed=42
        ).iloc[0]
        self.assertEqual(
            metrics["target_definition"], "no_lower_price_in_next_h_observations"
        )
        self.assertEqual(metrics["hit_rate"], 0.0)

    def test_frequency_is_independent_of_evaluation_horizon(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=10)
        prices = pd.DataFrame(
            {"date": dates, "currency": "AMD", "rub_per_unit": [100.0] * 10}
        )
        signals = pd.DataFrame(
            {
                "date": [dates[3], dates[8]],
                "currency": ["AMD", "AMD"],
                "strategy_id": ["test", "test"],
                "signal_family": ["level", "level"],
                "scenario": ["favourable_now", "favourable_now"],
                "parameters": [
                    '{"higher_share": 0.5, "lookback": 2}',
                    '{"higher_share": 0.5, "lookback": 2}',
                ],
            }
        )
        metrics = evaluate_level_corridor(
            prices, signals, dates[0], (1, 3), repeats=20, seed=42
        ).set_index("horizon")
        self.assertEqual(metrics.loc[1, "signal_count"], 2)
        self.assertEqual(metrics.loc[3, "signal_count"], 2)
        self.assertEqual(
            metrics.loc[1, "signals_per_week"], metrics.loc[3, "signals_per_week"]
        )
        self.assertGreater(
            metrics.loc[1, "evaluated_signal_count"],
            metrics.loc[3, "evaluated_signal_count"],
        )

    def test_hit_tolerance_is_reported_separately_from_exact_hit(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        outcomes = outcome_frame(pd.Series([100.2, 100.0, 99.8]), dates, horizon=1)
        middle = outcomes.loc[dates[1]]
        self.assertEqual(middle["hit"], 0.0)
        self.assertEqual(middle["hit_10bp"], 0.0)
        self.assertEqual(middle["hit_25bp"], 1.0)
        self.assertEqual(middle["forward_hold_hit"], 0.0)

    def test_local_minimum_target_is_stricter_than_forward_hold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        outcomes = outcome_frame(pd.Series([99.0, 100.0, 101.0]), dates, horizon=1)
        middle = outcomes.loc[dates[1]]
        self.assertEqual(middle["forward_hold_hit"], 1.0)
        self.assertEqual(middle["hit"], 1.0)
        self.assertEqual(middle["local_min_hit"], 0.0)

    def test_as_of_signals_do_not_change_when_future_prices_change(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=30)
        prices = pd.DataFrame(
            {"date": dates, "currency": "AMD", "rub_per_unit": range(30, 0, -1)}
        )
        as_of = dates[24]
        before = signals_as_of(prices, as_of, dates[0])
        changed = prices.copy()
        changed.loc[changed["date"] > as_of, "rub_per_unit"] = 10_000.0
        after = signals_as_of(changed, as_of, dates[0])
        self.assertEqual(before["strategy_id"].tolist(), after["strategy_id"].tolist())

    def test_signal_grid_contains_only_pure_spike_policies(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=30)
        prices = pd.DataFrame(
            {
                "date": dates,
                "currency": "AMD",
                "rub_per_unit": pd.Series(range(130, 100, -1), dtype=float),
            }
        )
        signals, policies = build_signals(prices, dates[0], cooldown_observations=3)
        self.assertEqual(set(signals["policy"]), {"spike_only"})
        self.assertEqual(set(signals["strategy_family"]), {"spike_policy"})
        self.assertFalse(any("calendar" in column for column in signals.columns))
        self.assertEqual(set(policies["signal_type"]), {"spike_policy"})
        self.assertEqual(len(policies), 4)
        self.assertEqual(set(policies["lookback"]), {2, 3, 4, 5})
        self.assertTrue(
            policies["parameters"].str.contains('"consecutive_declines"').all()
        )

    def test_excluded_long_streaks_are_audited_but_not_evaluated(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=25)
        prices = pd.DataFrame(
            {
                "date": dates,
                "currency": "AMD",
                "rub_per_unit": pd.Series(range(125, 100, -1), dtype=float),
            }
        )
        audit = excluded_momentum_audit(prices, dates[0]).set_index("strategy_id")
        self.assertEqual(audit.loc["momentum_streak_10d__spike_only", "signal_count"], 1)
        self.assertEqual(audit.loc["momentum_streak_20d__spike_only", "signal_count"], 1)
        signals, _ = build_signals(prices, dates[0], cooldown_observations=0)
        self.assertFalse(signals["strategy_id"].str.contains("10d|20d").any())

    def test_corridor_exit_is_detected_on_first_breakout(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=12)
        prices = pd.DataFrame(
            {
                "date": dates,
                "currency": "AMD",
                "rub_per_unit": [100.0] * 10 + [98.0, 98.0],
            }
        )
        signals, _ = build_level_corridor_signals(prices, dates[0], 3)
        down = signals.loc[
            signals["signal_family"].eq("corridor_exit_down")
            & signals["date"].eq(dates[10])
        ]
        self.assertFalse(down.empty)
        self.assertEqual(set(down["scenario"]), {"favourable_now"})

    def test_future_prices_do_not_change_past_corridor_exit(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=15)
        prices = pd.DataFrame(
            {
                "date": dates,
                "currency": "AMD",
                "rub_per_unit": [100.0] * 10 + [102.0] + [100.0] * 4,
            }
        )
        before, _ = build_level_corridor_signals(prices, dates[0], 3)
        changed = prices.copy()
        changed.loc[changed["date"] > dates[10], "rub_per_unit"] = 1_000.0
        after, _ = build_level_corridor_signals(changed, dates[0], 3)
        columns = ["date", "strategy_id", "signal_family"]
        before = before.loc[before["date"] <= dates[10], columns].reset_index(drop=True)
        after = after.loc[after["date"] <= dates[10], columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)

    def test_scenarios_change_hit_but_not_case_benefit(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6)
        outcomes = outcome_frame(
            pd.Series([101.0, 100.0, 100.0, 99.0, 102.0, 103.0]),
            dates,
            horizon=2,
        )
        favourable = scenario_outcomes(outcomes, "favourable_now")
        closing = scenario_outcomes(outcomes, "window_closing")
        self.assertEqual(favourable.loc[dates[2], "target_hit"], 0.0)
        self.assertEqual(closing.loc[dates[2], "target_hit"], 1.0)
        self.assertEqual(
            favourable.loc[dates[2], "benefit_bps"],
            closing.loc[dates[2], "benefit_bps"],
        )

    def test_monthly_bootstrap_interval_and_one_sided_tail_are_coherent(self) -> None:
        dates = pd.date_range("2022-01-01", periods=730, freq="D")
        values = pd.Series(10.0, index=dates)
        inference = monthly_block_bootstrap_mean(values, repeats=300, seed=42)
        self.assertGreater(inference["ci_low"], 0.0)
        self.assertLess(inference["p_value"], 0.05)

        negative = monthly_block_bootstrap_mean(-values, repeats=300, seed=42)
        self.assertLess(negative["ci_high"], 0.0)
        self.assertGreater(negative["p_value"], 0.95)


if __name__ == "__main__":
    unittest.main()
