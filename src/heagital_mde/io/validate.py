from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import pandas as pd

from heagital_mde.data.schema import (
    OPTIONAL_CANONICAL_COLUMNS,
    REQUIRED_CANONICAL_COLUMNS,
    UNIT_INTERVAL_COLUMNS,
)

#: Prevalence is recorded as a percentage of the registered population. Values
#: above this are treated as a unit error rather than a real observation.
MAX_PREVALENCE_PCT = 100.0

#: Below this row count, min-max normalisation cannot produce a defensible
#: spread. Reported as a warning, not an error.
MIN_ROWS_WARNING = 3


class ValidationError(ValueError):
    """Raised when input data cannot be scored."""


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            bullets = "\n  - ".join(self.errors)
            raise ValidationError(f"Input data failed validation:\n  - {bullets}")


def _check_required_columns(df: pd.DataFrame, report: ValidationReport) -> bool:
    missing = [c for c in REQUIRED_CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        report.errors.append(f"Missing required canonical columns: {missing}")
        return False
    return True


def _check_identifier(df: pd.DataFrame, col: str, report: ValidationReport) -> None:
    blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
    if blank.any():
        rows = [int(i) + 2 for i in df.index[blank][:5]]  # +2: 1-based, past header
        report.errors.append(f"{col} is missing or blank in {int(blank.sum())} row(s), e.g. spreadsheet row {rows}.")


def _check_non_negative(df: pd.DataFrame, col: str, report: ValidationReport) -> None:
    if df[col].isna().any():
        count = int(df[col].isna().sum())
        report.errors.append(f"{col} could not be parsed as a number in {count} row(s).")
    negative = df[col] < 0
    if negative.any():
        report.errors.append(f"{col} contains {int(negative.sum())} negative value(s).")


def _check_unit_interval(df: pd.DataFrame, col: str, report: ValidationReport, required: bool) -> None:
    if col not in df.columns:
        return
    if df[col].isna().any():
        count = int(df[col].isna().sum())
        message = f"{col} could not be parsed as a number in {count} row(s)."
        (report.errors if required else report.warnings).append(message)
    out_of_range = (df[col] < 0) | (df[col] > 1)
    if out_of_range.any():
        observed = df.loc[out_of_range, col]
        report.errors.append(
            f"{col} must be a proportion between 0 and 1; found {int(out_of_range.sum())} value(s) "
            f"outside that range (min {observed.min():.4g}, max {observed.max():.4g})."
        )


def check_icb_features(df: pd.DataFrame) -> ValidationReport:
    """Validate a loaded ICB frame and return every problem found.

    Collects all findings rather than stopping at the first, so a user fixing
    a spreadsheet sees the whole list in one pass.
    """
    report = ValidationReport()

    if not _check_required_columns(df, report):
        return report

    if len(df) == 0:
        report.errors.append("Input contains no rows.")
        return report

    if len(df) < MIN_ROWS_WARNING:
        report.warnings.append(
            f"Only {len(df)} ICB row(s) supplied. Normalisation needs at least {MIN_ROWS_WARNING} "
            "rows to spread scores meaningfully; results will be indicative only."
        )

    _check_identifier(df, "icb_code", report)
    _check_identifier(df, "icb_name", report)

    duplicated = df["icb_code"].astype(str).str.strip().duplicated(keep=False)
    if duplicated.any():
        codes = sorted(set(df.loc[duplicated, "icb_code"].astype(str)))
        report.errors.append(
            f"icb_code must be unique; {len(codes)} duplicated code(s): {codes[:10]}"
            f"{' ...' if len(codes) > 10 else ''}."
        )

    _check_non_negative(df, "register", report)
    _check_non_negative(df, "warfarin_proxy", report)
    _check_non_negative(df, "prevalence", report)

    if "prevalence" in df.columns:
        too_high = df["prevalence"] > MAX_PREVALENCE_PCT
        if too_high.any():
            report.errors.append(
                f"prevalence is a percentage and cannot exceed {MAX_PREVALENCE_PCT:g}; "
                f"found {int(too_high.sum())} value(s) above it (max {df['prevalence'].max():.4g})."
            )

    _check_unit_interval(df, "treatment_gap", report, required=True)
    for col in OPTIONAL_CANONICAL_COLUMNS:
        if col in UNIT_INTERVAL_COLUMNS:
            _check_unit_interval(df, col, report, required=False)

    if "region" in df.columns and df["region"].isna().any():
        report.warnings.append(
            f"region is blank in {int(df['region'].isna().sum())} row(s); "
            "those ICBs are grouped as 'Unknown' in regional summaries."
        )

    return report


def validate_icb_features(df: pd.DataFrame, warn: bool = True) -> ValidationReport:
    """Validate and raise :class:`ValidationError` on the first failing report."""
    report = check_icb_features(df)
    report.raise_if_failed()

    if warn and report.warnings:
        import warnings as _warnings

        for message in report.warnings:
            _warnings.warn(message, UserWarning, stacklevel=2)

    return report
