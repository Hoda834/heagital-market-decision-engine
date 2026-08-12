from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Literal, Optional

import pandas as pd

from heagital_mde.data.schema import (
    OPTIONAL_CANONICAL_COLUMNS,
    REQUIRED_CANONICAL_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    UNIT_INTERVAL_COLUMNS,
    resolve_header,
)

CSV_SUFFIXES = {".csv", ".txt", ".tsv"}
EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm"}

GapUnits = Literal["percent", "fraction", "auto"]

#: Treatment gaps at or below this value look like fractions rather than
#: percentages (a 1% gap is implausible for AF anticoagulation).
FRACTION_HEURISTIC_MAX = 1.0


def resolve_input_path(path: str | Path) -> Path:
    """Resolve a user-supplied input location to a single readable data file.

    Accepts a file path, or a directory containing exactly one data file.
    A directory holding several candidates is an error rather than a guess,
    because silently picking one would change the ranking without telling
    anybody which file produced it.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {p}")

    if p.is_file():
        return p

    if p.is_dir():
        candidates = sorted(
            c for c in p.iterdir() if c.is_file() and c.suffix.lower() in (CSV_SUFFIXES | EXCEL_SUFFIXES)
        )
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(
                f"No .csv or .xlsx file found in directory: {p}. "
                "Point the loader at a data file, or place exactly one in this directory."
            )
        raise ValueError(
            f"Directory {p} contains {len(candidates)} data files "
            f"({[c.name for c in candidates]}). Specify which one to load."
        )

    raise ValueError(f"Input path is neither a file nor a directory: {p}")


def _looks_like_csv(p: Path) -> bool:
    """Sniff an extensionless file for delimited text."""
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
    except OSError:
        return False
    return "," in first_line or "\t" in first_line


def _read_table(p: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    suffix = p.suffix.lower()

    if suffix in CSV_SUFFIXES:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(p, sep=sep)

    if suffix in EXCEL_SUFFIXES:
        frame = pd.read_excel(p, sheet_name=sheet_name if sheet_name else 0)
        if isinstance(frame, dict):  # sheet_name=None returns every sheet
            raise ValueError(f"Multiple sheets returned for {p}. Pass an explicit sheet_name.")
        return frame

    if suffix == "" and _looks_like_csv(p):
        return pd.read_csv(p)

    raise ValueError(
        f"Unsupported file type {suffix or '(no extension)'!r} for {p}. "
        f"Provide one of: {sorted(CSV_SUFFIXES | EXCEL_SUFFIXES)}."
    )


def _drop_empty_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [c for c in df.columns if str(c).strip().lower().startswith("unnamed")]
    to_drop = [c for c in unnamed if df[c].isna().all()]
    return df.drop(columns=to_drop) if to_drop else df


def _parse_numeric(series: pd.Series) -> pd.Series:
    """Parse a column that may carry thousands separators, % signs or blanks."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)

    s = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    s = s.replace({"nan": None, "None": None, "NaN": None, "": None, "-": None, "N/A": None, "n/a": None})
    return pd.to_numeric(s, errors="coerce").astype(float)


def _clean_text(series: pd.Series) -> pd.Series:
    """Strip whitespace, mapping true blanks to NA instead of the string 'nan'."""
    s = series.astype("object").where(series.notna(), other=None)
    s = s.map(lambda v: str(v).strip() if v is not None else None)
    return s.map(lambda v: None if v in {"", "nan", "None"} else v)


def _resolve_gap_units(gap: pd.Series, gap_units: GapUnits) -> str:
    if gap_units in {"percent", "fraction"}:
        return gap_units
    if gap_units != "auto":
        raise ValueError(f"gap_units must be one of 'percent', 'fraction', 'auto'; got {gap_units!r}.")

    observed = gap.dropna()
    if observed.empty:
        return "percent"
    if float(observed.max()) <= FRACTION_HEURISTIC_MAX:
        return "fraction"
    return "percent"


def _warn_ambiguous_gap(gap: pd.Series, units: str, explicit: bool) -> None:
    """Warn on the two readings of the gap column that could silently be 100x wrong."""
    observed = gap.dropna()
    if observed.empty:
        return

    if units == "percent" and float(observed.max()) <= FRACTION_HEURISTIC_MAX:
        warnings.warn(
            "Treatment Gap values are all <= 1 but are being read as percentages, which yields "
            "gaps under 1%. If the column already holds fractions, pass gap_units='fraction'.",
            UserWarning,
            stacklevel=3,
        )
    elif units == "fraction" and not explicit:
        warnings.warn(
            "Treatment Gap values are all <= 1, so they were read as fractions rather than "
            "percentages. If the column holds percentages, pass gap_units='percent'.",
            UserWarning,
            stacklevel=3,
        )


def load_icb_features(
    path: str | Path,
    sheet_name: Optional[str] = None,
    gap_units: GapUnits = "auto",
    extra_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Load an ICB feature file and return canonical, typed columns.

    Parameters
    ----------
    path
        A data file, or a directory containing exactly one data file.
    sheet_name
        Worksheet to read for Excel inputs. Defaults to the first sheet.
    gap_units
        How to interpret the treatment gap column. ``'percent'`` divides by
        100, ``'fraction'`` takes the values as-is, and ``'auto'`` (default)
        infers from the observed range and warns when the choice is ambiguous.
    extra_columns
        Additional raw column names to carry through untouched.
    """
    resolved = resolve_input_path(path)
    df_raw = _drop_empty_unnamed_columns(_read_table(resolved, sheet_name=sheet_name))

    if df_raw.empty:
        raise ValueError(f"Input file contains no data rows: {resolved}")

    # Map every recognised header to its canonical name, first match wins.
    rename: dict[object, str] = {}
    seen: set[str] = set()
    for column in df_raw.columns:
        canonical = resolve_header(column)
        if canonical is None or canonical in seen:
            continue
        rename[column] = canonical
        seen.add(canonical)

    missing = [c for c in REQUIRED_CANONICAL_COLUMNS if c not in seen]
    if missing:
        raise ValueError(
            f"Missing required column(s) in {resolved.name}: {missing}. "
            f"Expected headers such as: {list(REQUIRED_RAW_COLUMNS)}. "
            f"Found: {[str(c) for c in df_raw.columns]}"
        )

    keep = [c for c in df_raw.columns if c in rename]
    keep += [c for c in extra_columns if c in df_raw.columns and c not in keep]
    df = df_raw[keep].rename(columns=rename).copy()

    df["icb_code"] = _clean_text(df["icb_code"])
    df["icb_name"] = _clean_text(df["icb_name"])
    if "region" in df.columns:
        df["region"] = _clean_text(df["region"])

    df["register"] = _parse_numeric(df["register"])
    df["prevalence"] = _parse_numeric(df["prevalence"])
    df["warfarin_proxy"] = _parse_numeric(df["warfarin_proxy"])

    gap = _parse_numeric(df["treatment_gap"])
    units = _resolve_gap_units(gap, gap_units)
    _warn_ambiguous_gap(gap, units, explicit=(gap_units != "auto"))
    df["treatment_gap"] = gap / 100.0 if units == "percent" else gap

    for col in OPTIONAL_CANONICAL_COLUMNS:
        if col in df.columns and col in UNIT_INTERVAL_COLUMNS:
            df[col] = _parse_numeric(df[col])

    df = df.reset_index(drop=True)
    # Record the interpretation actually applied, so the audit trail can report
    # 'percent' or 'fraction' rather than the user's 'auto' request.
    df.attrs["gap_units_resolved"] = units
    df.attrs["source_file"] = str(resolved)
    return df
