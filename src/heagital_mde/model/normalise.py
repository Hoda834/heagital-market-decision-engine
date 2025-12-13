from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


NormalisationMethod = Literal["minmax"]


@dataclass(frozen=True)
class NormalisationConfig:
    method: NormalisationMethod = "minmax"
    clip: bool = True


def minmax_normalise(series: pd.Series, clip: bool = True) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    if x.isna().all():
        return pd.Series([np.nan] * len(x), index=x.index)

    xmin = float(np.nanmin(x.to_numpy()))
    xmax = float(np.nanmax(x.to_numpy()))

    if np.isclose(xmax, xmin):
        out = pd.Series([0.0] * len(x), index=x.index, dtype=float)
        return out

    out = (x - xmin) / (xmax - xmin)

    if clip:
        out = out.clip(lower=0.0, upper=1.0)

    return out


def normalise_columns(
    df: pd.DataFrame,
    columns: list[str],
    cfg: NormalisationConfig,
    prefix: str = "n_",
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            raise ValueError(f"Cannot normalise missing column: {col}")
        if cfg.method != "minmax":
            raise ValueError(f"Unsupported normalisation method: {cfg.method}")
        out[f"{prefix}{col}"] = minmax_normalise(out[col], clip=cfg.clip)
    return out
