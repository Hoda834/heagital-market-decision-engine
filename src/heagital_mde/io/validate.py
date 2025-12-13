from __future__ import annotations

from typing import List

import pandas as pd

from heagital_mde.data.schema import REQUIRED_CANONICAL_COLUMNS


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required canonical columns: {missing}")


def validate_icb_features(df: pd.DataFrame) -> None:
    _require_columns(df, list(REQUIRED_CANONICAL_COLUMNS))

    if df["icb_code"].isna().any() or (df["icb_code"].astype(str).str.strip() == "").any():
        raise ValueError("icb_code contains missing or blank values.")

    if df["icb_name"].isna().any() or (df["icb_name"].astype(str).str.strip() == "").any():
        raise ValueError("icb_name contains missing or blank values.")

    if df["af_register"].isna().any():
        raise ValueError("af_register contains missing values after parsing.")
    if (df["af_register"] < 0).any():
        raise ValueError("af_register contains negative values.")

    if df["warfarin_proxy"].isna().any():
        raise ValueError("warfarin_proxy contains missing values after parsing.")
    if (df["warfarin_proxy"] < 0).any():
        raise ValueError("warfarin_proxy contains negative values.")

    if df["treatment_gap"].isna().any():
        raise ValueError("treatment_gap contains missing values after parsing.")
    if (df["treatment_gap"] < 0).any() or (df["treatment_gap"] > 1).any():
        raise ValueError("treatment_gap must be between 0 and 1 after conversion from percent.")
