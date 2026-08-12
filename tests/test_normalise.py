from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heagital_mde.model.normalise import (
    NormalisationConfig,
    minmax_normalise,
    normalise_columns,
    rank_normalise,
)


def test_minmax_maps_to_unit_interval():
    out = minmax_normalise(pd.Series([10.0, 20.0, 30.0]))
    assert out.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_constant_series_returns_neutral_midpoint_not_zero():
    """A signal with no spread must not silently forfeit its weight."""
    out = minmax_normalise(pd.Series([5.0, 5.0, 5.0]))
    assert out.tolist() == pytest.approx([0.5, 0.5, 0.5])


def test_single_row_returns_neutral_midpoint():
    out = minmax_normalise(pd.Series([42.0]))
    assert out.tolist() == pytest.approx([0.5])


def test_constant_fill_is_configurable():
    out = minmax_normalise(pd.Series([5.0, 5.0]), constant_fill=1.0)
    assert out.tolist() == pytest.approx([1.0, 1.0])


def test_winsorize_compresses_outlier_influence():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 1000.0])
    plain = minmax_normalise(series)
    trimmed = minmax_normalise(series, winsorize=0.2)
    # Without trimming the outlier squashes everything else towards zero.
    assert plain.iloc[3] < 0.01
    assert trimmed.iloc[3] > plain.iloc[3]


def test_rank_normalise_is_outlier_insensitive():
    out = rank_normalise(pd.Series([1.0, 2.0, 3.0, 1000.0]))
    assert out.tolist() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])


def test_rank_normalise_averages_ties():
    out = rank_normalise(pd.Series([1.0, 1.0, 3.0]))
    assert out.iloc[0] == pytest.approx(out.iloc[1])
    assert out.iloc[2] == pytest.approx(1.0)


def test_empty_series_stays_empty():
    assert len(minmax_normalise(pd.Series([], dtype=float))) == 0
    assert len(rank_normalise(pd.Series([], dtype=float))) == 0


def test_all_nan_series_returns_nan():
    out = minmax_normalise(pd.Series([np.nan, np.nan]))
    assert out.isna().all()


def test_unsupported_method_raises_on_construction():
    with pytest.raises(ValueError, match="Unsupported normalisation method"):
        NormalisationConfig(method="zscore")


def test_invalid_winsorize_raises():
    with pytest.raises(ValueError, match="winsorize"):
        NormalisationConfig(winsorize=0.6)


def test_normalise_columns_reports_missing_columns():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing columns"):
        normalise_columns(df, ["a", "b"], NormalisationConfig())


def test_normalise_columns_adds_prefixed_companions():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})
    out = normalise_columns(df, ["a", "b"], NormalisationConfig())
    assert out["n_a"].tolist() == pytest.approx([0.0, 1.0])
    assert out["n_b"].tolist() == pytest.approx([0.0, 1.0])
    assert "a" in out.columns  # originals preserved
