from __future__ import annotations

import pytest

from heagital_mde.io.load_icb import load_icb_features
from heagital_mde.io.validate import ValidationError, check_icb_features, validate_icb_features


def test_valid_data_passes(write_csv, sample_rows):
    report = check_icb_features(load_icb_features(write_csv(sample_rows)))
    assert report.ok
    assert report.errors == []


def test_duplicate_icb_code_is_rejected(write_csv):
    rows = "QA1,Alpha,1000,2.0,10,500,London\nQA1,Alpha again,2000,3.0,20,600,London\n"
    report = check_icb_features(load_icb_features(write_csv(rows)))
    assert not report.ok
    assert any("unique" in e for e in report.errors)


def test_prevalence_above_100_is_rejected(write_csv):
    rows = "QA1,Alpha,1000,999,10,500,London\nQB2,Beta,2000,3.0,20,600,London\n"
    report = check_icb_features(load_icb_features(write_csv(rows)))
    assert any("prevalence" in e and "cannot exceed" in e for e in report.errors)


def test_negative_register_is_rejected(write_csv):
    rows = "QA1,Alpha,-5,2.0,10,500,London\nQB2,Beta,2000,3.0,20,600,London\n"
    report = check_icb_features(load_icb_features(write_csv(rows)))
    assert any("register" in e and "negative" in e for e in report.errors)


def test_treatment_gap_out_of_range_is_rejected(write_csv):
    rows = "QA1,Alpha,1000,2.0,150,500,London\nQB2,Beta,2000,3.0,20,600,London\n"
    report = check_icb_features(load_icb_features(write_csv(rows)))
    assert any("treatment_gap" in e for e in report.errors)


def test_blank_icb_code_is_rejected(write_csv):
    rows = ",Alpha,1000,2.0,10,500,London\nQB2,Beta,2000,3.0,20,600,London\n"
    report = check_icb_features(load_icb_features(write_csv(rows)))
    assert any("icb_code" in e for e in report.errors)


def test_all_errors_are_collected_not_just_the_first(write_csv):
    rows = "QA1,Alpha,-5,999,10,500,London\nQA1,Beta,2000,3.0,20,600,London\n"
    report = check_icb_features(load_icb_features(write_csv(rows)))
    assert len(report.errors) >= 3


def test_small_panel_warns_but_passes(write_csv):
    rows = "QA1,Alpha,1000,2.0,10,500,London\nQB2,Beta,2000,3.0,20,600,London\n"
    report = check_icb_features(load_icb_features(write_csv(rows)))
    assert report.ok
    assert any("indicative only" in w for w in report.warnings)


def test_blank_region_warns(write_csv, sample_rows):
    rows = sample_rows + "QE5,Epsilon,1000,2.0,10,500,\n"
    report = check_icb_features(load_icb_features(write_csv(rows)))
    assert report.ok
    assert any("region" in w for w in report.warnings)


def test_validate_raises_with_every_problem_listed(write_csv):
    rows = "QA1,Alpha,-5,999,10,500,London\nQA1,Beta,2000,3.0,20,600,London\n"
    df = load_icb_features(write_csv(rows))
    with pytest.raises(ValidationError) as exc:
        validate_icb_features(df, warn=False)
    message = str(exc.value)
    assert "register" in message and "prevalence" in message and "unique" in message


def test_optional_column_out_of_range_is_rejected(write_csv):
    from tests.conftest import OPTIONAL_HEADER

    rows = (
        "QA1,Alpha,1000,2.0,10,500,London,1.8,0.1\n"
        "QB2,Beta,2000,3.0,20,600,London,0.4,0.2\n"
        "QC3,Gamma,3000,4.0,30,700,London,0.5,0.3\n"
    )
    report = check_icb_features(load_icb_features(write_csv(rows, header=OPTIONAL_HEADER)))
    assert any("digital_maturity" in e for e in report.errors)
