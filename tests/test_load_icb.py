from __future__ import annotations

import warnings

import pandas as pd
import pytest

from heagital_mde.io.load_icb import load_icb_features, resolve_input_path


def test_loads_canonical_columns(write_csv, sample_rows):
    df = load_icb_features(write_csv(sample_rows))

    assert list(df.columns[:6]) == [
        "icb_code",
        "icb_name",
        "register",
        "prevalence",
        "treatment_gap",
        "warfarin_proxy",
    ]
    assert len(df) == 4


def test_parses_thousands_separators(write_csv, sample_rows):
    df = load_icb_features(write_csv(sample_rows))
    assert df.loc[0, "register"] == pytest.approx(74319.0)


def test_percent_gap_converted_to_fraction(write_csv, sample_rows):
    df = load_icb_features(write_csv(sample_rows), gap_units="percent")
    assert df.loc[0, "treatment_gap"] == pytest.approx(0.1037)


def test_fraction_gap_is_left_alone(write_csv):
    rows = "QA1,Alpha,1000,2.0,0.18,500,London\nQB2,Beta,2000,3.0,0.22,600,London\n"
    df = load_icb_features(write_csv(rows), gap_units="fraction")
    assert df.loc[0, "treatment_gap"] == pytest.approx(0.18)


def test_auto_detects_fraction_gaps(write_csv):
    """Values that are all <= 1 are read as fractions, not as sub-1% percentages."""
    rows = "QA1,Alpha,1000,2.0,0.18,500,London\nQB2,Beta,2000,3.0,0.22,600,London\n"
    df = load_icb_features(write_csv(rows), gap_units="auto")
    assert df.loc[0, "treatment_gap"] == pytest.approx(0.18)


def test_explicit_percent_on_fraction_data_warns(write_csv):
    """Reading 0.18 as a percentage yields a 0.18% gap: a silent 100x error."""
    rows = "QA1,Alpha,1000,2.0,0.18,500,London\nQB2,Beta,2000,3.0,0.22,600,London\n"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = load_icb_features(write_csv(rows), gap_units="percent")
    assert df.loc[0, "treatment_gap"] == pytest.approx(0.0018)
    assert [w for w in caught if "read as percentages" in str(w.message)]


def test_auto_detected_fractions_are_announced(write_csv):
    rows = "QA1,Alpha,1000,2.0,0.18,500,London\nQB2,Beta,2000,3.0,0.22,600,London\n"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_icb_features(write_csv(rows), gap_units="auto")
    assert [w for w in caught if "read as fractions" in str(w.message)]


def test_ordinary_percent_data_is_not_warned_about(write_csv, sample_rows):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_icb_features(write_csv(sample_rows), gap_units="auto")
    assert not [w for w in caught if "Treatment Gap" in str(w.message)]


def test_resolved_gap_units_are_recorded(write_csv, sample_rows):
    df = load_icb_features(write_csv(sample_rows), gap_units="auto")
    assert df.attrs["gap_units_resolved"] == "percent"

    rows = "QA1,Alpha,1000,2.0,0.18,500,London\nQB2,Beta,2000,3.0,0.22,600,London\n"
    assert load_icb_features(write_csv(rows), gap_units="auto").attrs["gap_units_resolved"] == "fraction"


def test_missing_required_column_raises(write_csv, sample_rows):
    header = "ICB ODS code,ICB name,Register,Prevalence (%),Warfarin Item icb,Region"
    rows = "QA1,Alpha,1000,2.0,500,London\n"
    with pytest.raises(ValueError, match="treatment_gap"):
        load_icb_features(write_csv(rows, header=header))


def test_region_is_optional(write_csv):
    header = "ICB ODS code,ICB name,Register,Prevalence (%),Treatment Gap (%),Warfarin Item icb"
    rows = "QA1,Alpha,1000,2.0,10,500\nQB2,Beta,2000,3.0,20,600\n"
    df = load_icb_features(write_csv(rows, header=header))
    assert "region" not in df.columns
    assert len(df) == 2


def test_optional_columns_are_loaded(write_csv, sample_rows_with_optional):
    from tests.conftest import OPTIONAL_HEADER

    df = load_icb_features(write_csv(sample_rows_with_optional, header=OPTIONAL_HEADER))
    assert df["digital_maturity"].tolist() == [0.80, 0.30, 0.60, 0.20]
    assert df["procurement_friction"].tolist() == [0.10, 0.70, 0.40, 0.90]


def test_headers_are_matched_case_insensitively(write_csv):
    header = "icb ods code,ICB NAME,register,prevalence (%),treatment gap (%),warfarin item icb,region"
    rows = "QA1,Alpha,1000,2.0,10,500,London\nQB2,Beta,2000,3.0,20,600,London\n"
    df = load_icb_features(write_csv(rows, header=header))
    assert df.loc[0, "icb_code"] == "QA1"


def test_blank_region_becomes_na_not_the_string_nan(write_csv):
    rows = "QA1,Alpha,1000,2.0,10,500,\nQB2,Beta,2000,3.0,20,600,London\n"
    df = load_icb_features(write_csv(rows))
    assert pd.isna(df.loc[0, "region"])


def test_directory_with_one_file_resolves(tmp_path, write_csv, sample_rows):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    (data_dir / "icb.csv").write_text(
        "ICB ODS code,ICB name,Register,Prevalence (%),Treatment Gap (%),Warfarin Item icb,Region\n" + sample_rows,
        encoding="utf-8",
    )
    assert resolve_input_path(data_dir).name == "icb.csv"
    assert len(load_icb_features(data_dir)) == 4


def test_directory_with_several_files_raises(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    (data_dir / "a.csv").write_text("x\n", encoding="utf-8")
    (data_dir / "b.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains 2 data files"):
        resolve_input_path(data_dir)


def test_empty_directory_raises(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No .csv or .xlsx file"):
        resolve_input_path(data_dir)


def test_extensionless_csv_is_sniffed(tmp_path, sample_rows):
    path = tmp_path / "raw"
    path.write_text(
        "ICB ODS code,ICB name,Register,Prevalence (%),Treatment Gap (%),Warfarin Item icb,Region\n" + sample_rows,
        encoding="utf-8",
    )
    assert len(load_icb_features(path)) == 4


def test_unsupported_file_type_raises(tmp_path):
    path = tmp_path / "data.pdf"
    path.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_icb_features(path)


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_icb_features(tmp_path / "nope.csv")


def test_header_only_file_raises(write_csv):
    with pytest.raises(ValueError, match="no data rows"):
        load_icb_features(write_csv(""))
