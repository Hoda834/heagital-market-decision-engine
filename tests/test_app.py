"""Tests for the Streamlit app's pure helpers.

The app module is imported directly; only functions that do not touch the
Streamlit runtime are exercised here. These cover the regional summary
builders, which previously demanded score columns the engine never produced
and therefore raised on every run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("streamlit", reason="Streamlit is an optional app dependency.")
pytest.importorskip("plotly", reason="Plotly is an optional app dependency.")


def _load_app_module():
    spec = importlib.util.spec_from_file_location("heagital_app", PROJECT_ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["heagital_app"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


app = _load_app_module()


@pytest.fixture
def ranked(write_csv, sample_rows, scoring_config_path):
    from heagital_mde.io.load_icb import load_icb_features
    from heagital_mde.model.scoring import score_and_rank

    df = load_icb_features(write_csv(sample_rows))
    return score_and_rank(df, scoring_config_path=scoring_config_path, top_n_override=2)


def test_region_pivot_builds_from_real_engine_output(ranked):
    """Regression: the pivot must only require columns the engine actually emits."""
    pivot = app.build_region_pivot(ranked)

    assert set(pivot["region"]) == {"North West", "London", "Midlands", "South West"}
    assert pivot["total_icbs"].sum() == len(ranked)
    assert pivot["included_icbs"].sum() == int(ranked["recommended_included"].sum())
    assert "avg_final_score" in pivot.columns
    assert "avg_n_register" in pivot.columns


def test_region_pivot_inclusion_rate_is_a_percentage(ranked):
    pivot = app.build_region_pivot(ranked)
    assert ((pivot["included_rate"] >= 0) & (pivot["included_rate"] <= 100)).all()


def test_region_scores_are_sorted_descending(ranked):
    scores = app.build_region_scores(ranked)
    assert list(scores.columns) == ["region", "avg_final_score"]
    assert scores["avg_final_score"].is_monotonic_decreasing
    assert len(scores) == 4


def test_blank_regions_group_as_unknown(ranked):
    frame = ranked.copy()
    frame.loc[0, "region"] = None
    pivot = app.build_region_pivot(frame)
    assert "Unknown" in set(pivot["region"])


def test_region_helpers_report_missing_columns():
    with pytest.raises(ValueError, match="Missing columns"):
        app.build_region_pivot(pd.DataFrame({"region": ["London"]}))
    with pytest.raises(ValueError, match="Missing columns"):
        app.build_region_scores(pd.DataFrame({"region": ["London"]}))


def test_pivot_adapts_to_whichever_signals_are_present(ranked):
    """A run without the optional signals must still summarise cleanly."""
    pivot = app.build_region_pivot(ranked)
    assert "avg_n_digital_maturity" not in pivot.columns
    assert "avg_n_treatment_gap" in pivot.columns


def test_centroids_drop_unknown_regions():
    scores = pd.DataFrame({"region": ["London", "Atlantis"], "avg_final_score": [0.8, 0.5]})
    mapped = app.apply_region_centroids(scores)
    assert mapped["region"].tolist() == ["London"]


def test_empty_geojson_placeholder_is_treated_as_absent(tmp_path):
    placeholder = tmp_path / "empty.geojson"
    placeholder.write_text("\r\n", encoding="utf-8")
    assert app.load_region_geojson(placeholder) is None
    assert app.load_region_geojson(tmp_path / "missing.geojson") is None


def test_populated_geojson_is_loaded_and_keyed(tmp_path):
    geojson = tmp_path / "regions.geojson"
    geojson.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature",'
        '"properties":{"nhser22nm":"London"},'
        '"geometry":{"type":"Polygon","coordinates":[[[0,0],[0,1],[1,1],[0,0]]]}}]}',
        encoding="utf-8",
    )
    loaded = app.load_region_geojson(geojson)
    assert loaded is not None
    assert app.geojson_name_key(loaded) == "nhser22nm"


def test_malformed_geojson_does_not_raise(tmp_path):
    broken = tmp_path / "broken.geojson"
    broken.write_text("{not valid json at all, but long enough to pass size check}", encoding="utf-8")
    assert app.load_region_geojson(broken) is None


def test_humanise_produces_readable_labels():
    assert app.humanise("avg_n_register") == "Average normalised register"
    assert app.humanise("n_treatment_gap") == "Normalised treatment gap"


def test_normalise_region_label_handles_blanks():
    assert app.normalise_region_label(None) == "Unknown"
    assert app.normalise_region_label(float("nan")) == "Unknown"
    assert app.normalise_region_label("  ") == "Unknown"
    assert app.normalise_region_label("  London ") == "London"
