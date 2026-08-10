from __future__ import annotations

import pandas as pd
import pytest

from heagital_mde.io.load_icb import load_icb_features
from heagital_mde.model.normalise import NormalisationConfig
from heagital_mde.model.scoring import ScoringResult, score_and_rank
from heagital_mde.model.scoring.rank import rank_and_flag
from heagital_mde.model.scoring.schema import (
    MarketWeightConfig,
    ReadinessWeightConfig,
    ScoringConfig,
    load_scoring_config,
)


def _config(**overrides) -> ScoringConfig:
    base = dict(
        market_weights=MarketWeightConfig(0.25, 0.25, 0.25, 0.25),
        readiness_weights=ReadinessWeightConfig(0.5, 0.5, 0.0),
        alpha=0.6,
        friction_weight=0.0,
        normalisation=NormalisationConfig(),
        top_n=2,
    )
    base.update(overrides)
    return ScoringConfig(**base)


# --------------------------------------------------------------------------- config


def test_shipped_config_weights_are_actually_applied(scoring_config_path):
    """The pre-0.2 loader silently ignored the file and used built-in defaults."""
    cfg = load_scoring_config(scoring_config_path)
    assert cfg.market_weights.register == pytest.approx(0.30)
    assert cfg.market_weights.treatment_gap == pytest.approx(0.30)
    assert cfg.readiness_weights.digital_maturity == pytest.approx(0.40)
    assert cfg.alpha == pytest.approx(0.60)
    assert cfg.friction_weight == pytest.approx(0.20)
    assert cfg.top_n == 15


def test_weights_are_renormalised_to_sum_to_one(tmp_path):
    path = tmp_path / "cfg.yml"
    path.write_text(
        "weights:\n  market:\n    register: 2\n    prevalence: 2\n    treatment_gap: 2\n    warfarin_proxy: 2\n",
        encoding="utf-8",
    )
    cfg = load_scoring_config(path)
    assert sum(cfg.market_weights.as_dict().values()) == pytest.approx(1.0)


def test_legacy_config_format_raises_instead_of_being_ignored(tmp_path):
    path = tmp_path / "legacy.yml"
    path.write_text(
        "weights:\n  clinical_risk: 0.45\n  adoption_readiness: 0.35\n  procurement_friction: 0.20\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pre-0.2 weight format"):
        load_scoring_config(path)


def test_unknown_config_key_raises(tmp_path):
    path = tmp_path / "typo.yml"
    path.write_text("weights:\n  market:\n    registr: 0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown key"):
        load_scoring_config(path)


def test_out_of_range_alpha_raises(tmp_path):
    path = tmp_path / "cfg.yml"
    path.write_text("alpha: 1.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        load_scoring_config(path)


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scoring_config(tmp_path / "nope.yml")


# --------------------------------------------------------------------------- scoring


def test_ranking_is_ordered_and_flagged(write_csv, sample_rows):
    df = load_icb_features(write_csv(sample_rows))
    ranked = score_and_rank(df, config=_config())

    assert ranked["rank"].tolist() == [1, 2, 3, 4]
    assert ranked["final_score"].is_monotonic_decreasing
    assert ranked["recommended_included"].tolist() == [True, True, False, False]


def test_score_is_a_convex_blend_of_the_two_pillars(write_csv, sample_rows):
    df = load_icb_features(write_csv(sample_rows))
    ranked = score_and_rank(df, config=_config(alpha=0.25))
    expected = 0.25 * ranked["market_score"] + 0.75 * ranked["readiness_score"]
    pd.testing.assert_series_equal(ranked["final_score"], expected, check_names=False)


def test_alpha_extremes_reduce_to_a_single_pillar(write_csv, sample_rows):
    df = load_icb_features(write_csv(sample_rows))
    market_only = score_and_rank(df, config=_config(alpha=1.0))
    pd.testing.assert_series_equal(
        market_only["final_score"], market_only["market_score"], check_names=False
    )
    readiness_only = score_and_rank(df, config=_config(alpha=0.0))
    pd.testing.assert_series_equal(
        readiness_only["final_score"], readiness_only["readiness_score"], check_names=False
    )


def test_empty_input_raises_a_clear_error_not_an_index_error(write_csv):
    """The pre-0.2 CLI crashed with IndexError on a header-only file."""
    df = pd.DataFrame(
        columns=["icb_code", "icb_name", "register", "prevalence", "treatment_gap", "warfarin_proxy"]
    )
    with pytest.raises(ValueError, match="No scorable rows"):
        score_and_rank(df, config=_config())


def test_rows_with_missing_signals_are_dropped_and_reported(write_csv, sample_rows):
    rows = sample_rows + "QE5,Epsilon,,2.0,10,500,London\n"
    df = load_icb_features(write_csv(rows))
    result = score_and_rank(df, config=_config(), return_result=True)
    assert isinstance(result, ScoringResult)
    assert len(result.ranking) == 4
    assert result.dropped_rows == 1
    assert any("dropped" in w for w in result.warnings)


def test_ties_break_deterministically_regardless_of_input_order(write_csv):
    rows = (
        "QB2,Beta,1000,2.0,10,500,London\n"
        "QA1,Alpha,1000,2.0,10,500,London\n"
        "QC3,Gamma,1000,2.0,10,500,London\n"
    )
    df = load_icb_features(write_csv(rows))
    first = score_and_rank(df, config=_config())
    second = score_and_rank(df.iloc[::-1].reset_index(drop=True), config=_config())
    assert first["icb_code"].tolist() == second["icb_code"].tolist() == ["QA1", "QB2", "QC3"]


def test_region_column_is_optional_in_output(write_csv):
    header = "ICB ODS code,ICB name,Register,Prevalence (%),Treatment Gap (%),Warfarin Item icb"
    rows = "QA1,Alpha,1000,2.0,10,500\nQB2,Beta,2000,3.0,20,600\nQC3,Gamma,3000,4.0,30,700\n"
    ranked = score_and_rank(load_icb_features(write_csv(rows, header=header)), config=_config())
    assert "region" not in ranked.columns
    assert len(ranked) == 3


# --------------------------------------------------------------------------- optional signals


def test_digital_maturity_feeds_readiness_when_present(write_csv, sample_rows_with_optional):
    from tests.conftest import OPTIONAL_HEADER

    df = load_icb_features(write_csv(sample_rows_with_optional, header=OPTIONAL_HEADER))
    result = score_and_rank(
        df,
        config=_config(readiness_weights=ReadinessWeightConfig(0.0, 0.0, 1.0)),
        return_result=True,
    )
    # Readiness is now purely digital maturity, so it must track that column.
    assert "digital_maturity" in result.signals_used
    top = result.ranking.iloc[0]
    assert top["icb_code"] == "QA1"  # highest digital maturity
    assert top["readiness_score"] == pytest.approx(1.0)


def test_absent_readiness_signal_redistributes_its_weight(write_csv, sample_rows):
    df = load_icb_features(write_csv(sample_rows))
    result = score_and_rank(
        df,
        config=_config(readiness_weights=ReadinessWeightConfig(0.25, 0.25, 0.50)),
        return_result=True,
    )
    effective = result.effective_readiness_weights
    assert effective["digital_maturity"] == 0.0
    assert effective["treatment_gap"] == pytest.approx(0.5)
    assert effective["warfarin_proxy"] == pytest.approx(0.5)
    assert "digital_maturity" in result.signals_absent


def test_friction_penalises_high_friction_icbs(write_csv, sample_rows_with_optional):
    from tests.conftest import OPTIONAL_HEADER

    df = load_icb_features(write_csv(sample_rows_with_optional, header=OPTIONAL_HEADER))
    without = score_and_rank(df, config=_config(friction_weight=0.0)).set_index("icb_code")
    with_friction = score_and_rank(df, config=_config(friction_weight=1.0)).set_index("icb_code")
    # QD4 has the highest procurement friction (0.90).
    assert with_friction.loc["QD4", "final_score"] < without.loc["QD4", "final_score"]


def test_friction_weight_without_the_column_warns_and_is_ignored(write_csv, sample_rows):
    df = load_icb_features(write_csv(sample_rows))
    result = score_and_rank(df, config=_config(friction_weight=0.5), return_result=True)
    assert any("friction" in w for w in result.warnings)
    assert (result.ranking["friction_score"] == 0.0).all()


def test_collinearity_is_reported(write_csv, sample_rows):
    df = load_icb_features(write_csv(sample_rows))
    result = score_and_rank(df, config=_config(), return_result=True)
    assert result.pillar_correlation is not None
    assert -1.0 <= result.pillar_correlation <= 1.0


# --------------------------------------------------------------------------- overrides & audit


def test_overrides_take_precedence_over_config(write_csv, sample_rows, scoring_config_path):
    df = load_icb_features(write_csv(sample_rows))
    result = score_and_rank(
        df,
        scoring_config_path=scoring_config_path,
        alpha_override=0.0,
        top_n_override=1,
        return_result=True,
    )
    assert result.config.alpha == pytest.approx(0.0)
    assert result.config.top_n == 1
    assert int(result.ranking["recommended_included"].sum()) == 1


def test_audit_record_is_json_serialisable(write_csv, sample_rows):
    import json

    df = load_icb_features(write_csv(sample_rows))
    result = score_and_rank(df, config=_config(), return_result=True)
    payload = json.loads(json.dumps(result.audit()))
    assert payload["rows_scored"] == 4
    assert payload["config"]["alpha"] == pytest.approx(0.6)


def test_score_and_rank_requires_a_config():
    with pytest.raises(ValueError, match="Provide either"):
        score_and_rank(pd.DataFrame())


# --------------------------------------------------------------------------- rank helper


def test_rank_and_flag_rejects_bad_top_n():
    df = pd.DataFrame({"icb_code": ["A", "B"], "final_score": [1.0, 2.0]})
    with pytest.raises(ValueError, match="top_n must be >= 1"):
        rank_and_flag(df, "final_score", 0)


def test_rank_and_flag_rejects_missing_score_column():
    df = pd.DataFrame({"icb_code": ["A"]})
    with pytest.raises(ValueError, match="missing column"):
        rank_and_flag(df, "final_score", 1)
