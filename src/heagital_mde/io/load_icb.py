from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from heagital_mde.data.schema import ColumnSpec


def _drop_empty_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [c for c in df.columns if str(c).strip().lower().startswith("unnamed")]
    if not unnamed:
        return df
    to_drop = []
    for c in unnamed:
        series = df[c]
        if series.isna().all():
            to_drop.append(c)
    if to_drop:
        df = df.drop(columns=to_drop)
    return df


def _parse_int_with_commas(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"nan": None, "None": None, "": None})
    out = pd.to_numeric(s, errors="coerce")
    return out


def _parse_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"nan": None, "None": None, "": None})
    out = pd.to_numeric(s, errors="coerce")
    return out


def load_icb_features(path: str | Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    if p.suffix.lower() in {".csv"}:
        df_raw = pd.read_csv(p)
    elif p.suffix.lower() in {".xlsx", ".xls"}:
        df_raw = pd.read_excel(p, sheet_name=sheet_name if sheet_name else 0)
    else:
        raise ValueError("Unsupported file type. Please provide a .csv, .xlsx, or .xls file.")

    df_raw = _drop_empty_unnamed_columns(df_raw)

    spec = ColumnSpec()

    missing_raw = [c for c in spec.raw_required if c not in df_raw.columns]
    if missing_raw:
        raise ValueError(f"Missing required columns in input file: {missing_raw}")

    df = df_raw[list(spec.raw_required)].rename(columns=spec.raw_to_canonical)

    df["icb_code"] = df["icb_code"].astype(str).str.strip()
    df["icb_name"] = df["icb_name"].astype(str).str.strip()
    df["region"] = df["region"].astype(str).str.strip()

    df["af_register"] = _parse_int_with_commas(df["af_register"])
    df["warfarin_proxy"] = _parse_float(df["warfarin_proxy"])

    gap_pct = _parse_float(df["treatment_gap"])
    df["treatment_gap"] = gap_pct / 100.0

    return df
