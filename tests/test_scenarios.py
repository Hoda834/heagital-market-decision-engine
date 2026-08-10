from __future__ import annotations

import pytest

from heagital_mde.io.load_icb import load_icb_features
from heagital_mde.model.scenarios import (
    build_sensitivity,
    load_scenarios,
    parse_scenario,
    run_scenarios,
)
from heagital_mde.model.scoring.schema import load_scoring_config


def test_shipped_scenarios_parse(scenario_config_path):
    scenarios = load_scenarios(scenario_config_path)
    assert "base_case" in scenarios
    assert len(scenarios) >= 4
    assert all(s.description for s in scenarios.values())


def test_base_case_leaves_the_config_untouched(scenario_config_path, scoring_config_path):
    base = load_scoring_config(scoring_config_path)
    applied = load_scenarios(scenario_config_path)["base_case"].apply(base)
    assert applied.alpha == pytest.approx(base.alpha)
    assert applied.market_weights.as_dict() == pytest.approx(base.market_weights.as_dict())


def test_scenario_deltas_shift_weights(scoring_config_path):
    base = load_scoring_config(scoring_config_path)
    scenario = parse_scenario("tilt", {"market_weight_deltas": {"treatment_gap": 0.5}, "alpha_delta": 0.2})
    applied = scenario.apply(base)

    assert applied.market_weights.treatment_gap > base.market_weights.treatment_gap
    assert applied.alpha == pytest.approx(min(1.0, base.alpha + 0.2))
    assert sum(applied.market_weights.as_dict().values()) == pytest.approx(1.0)


def test_alpha_delta_is_clamped(scoring_config_path):
    base = load_scoring_config(scoring_config_path)
    assert parse_scenario("high", {"alpha_delta": 5.0}).apply(base).alpha == pytest.approx(1.0)
    assert parse_scenario("low", {"alpha_delta": -5.0}).apply(base).alpha == pytest.approx(0.0)


def test_unknown_scenario_key_raises():
    with pytest.raises(ValueError, match="unknown key"):
        parse_scenario("bad", {"alpha_dleta": 0.1})


def test_unknown_weight_key_raises():
    with pytest.raises(ValueError, match="unknown key"):
        parse_scenario("bad", {"market_weight_deltas": {"registr": 0.1}})


def test_scenario_zeroing_every_weight_raises(scoring_config_path):
    base = load_scoring_config(scoring_config_path)
    scenario = parse_scenario(
        "wipe",
        {"market_weight_deltas": {k: -10.0 for k in ("register", "prevalence", "treatment_gap", "warfarin_proxy")}},
    )
    with pytest.raises(ValueError, match="zeroes out every market weight"):
        scenario.apply(base)


def test_run_scenarios_produces_one_ranking_each(write_csv, sample_rows, scoring_config_path, scenario_config_path):
    df = load_icb_features(write_csv(sample_rows))
    base = load_scoring_config(scoring_config_path)
    results = run_scenarios(df, base, load_scenarios(scenario_config_path))

    assert set(results) == set(load_scenarios(scenario_config_path))
    for result in results.values():
        assert len(result.ranking) == len(df)


def test_sensitivity_classifies_stability(write_csv, sample_rows, scoring_config_path, scenario_config_path):
    df = load_icb_features(write_csv(sample_rows))
    base = load_scoring_config(scoring_config_path)
    from heagital_mde.model.scoring.schema import with_overrides

    results = run_scenarios(df, with_overrides(base, top_n=2), load_scenarios(scenario_config_path))
    summary = build_sensitivity(results, base_scenario="base_case")

    assert len(summary) == len(df)
    assert set(summary["stability"]) <= {"robust", "fragile", "excluded"}
    assert (summary["worst_rank"] >= summary["best_rank"]).all()
    assert (summary["rank_spread"] == summary["worst_rank"] - summary["best_rank"]).all()
    assert "base_rank" in summary.columns


def test_sensitivity_needs_results():
    with pytest.raises(ValueError, match="No scenario results"):
        build_sensitivity({})


def test_missing_scenario_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scenarios(tmp_path / "nope.yml")
