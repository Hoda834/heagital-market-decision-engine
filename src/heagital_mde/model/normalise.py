from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

SUPPORTED_METHODS: tuple[str, ...] = ("minmax", "rank")


@dataclass(frozen=True)
class NormalisationConfig:
    """How raw signals are put onto a common 0-1 scale.

    method
        ``minmax`` preserves the shape of the distribution but is sensitive to
        outliers. ``rank`` uses the percentile position instead, which is far
        more stable on small panels (42 ICBs) at the cost of discarding
        magnitude information.
    clip
        Clamp the result into [0, 1] after scaling.
    winsorize
        Fraction trimmed from *each* tail before min-max scaling, e.g. 0.05
        clamps to the 5th/95th percentile first. Ignored by ``rank``.
    constant_fill
        Value used when a signal has no spread (every ICB identical, or only
        one row). 0.5 is deliberate: a signal that cannot discriminate should
        contribute a neutral mid-point, not silently zero out the weight it
        was given.
    """

    method: str = "minmax"
    clip: bool = True
    winsorize: float = 0.0
    constant_fill: float = 0.5

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported normalisation method: {self.method!r}. Supported: {list(SUPPORTED_METHODS)}"
            )
        if not 0.0 <= float(self.winsorize) < 0.5:
            raise ValueError(f"winsorize must be in [0, 0.5), got {self.winsorize}")
        if not 0.0 <= float(self.constant_fill) <= 1.0:
            raise ValueError(f"constant_fill must be in [0, 1], got {self.constant_fill}")


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def minmax_normalise(
    series: pd.Series,
    clip: bool = True,
    winsorize: float = 0.0,
    constant_fill: float = 0.5,
) -> pd.Series:
    """Scale a series to [0, 1] by its own min and max."""
    x = _as_float(series)

    if len(x) == 0:
        return pd.Series([], index=x.index, dtype=float)
    if x.isna().all():
        return pd.Series(np.nan, index=x.index, dtype=float)

    if winsorize > 0.0:
        lower = float(x.quantile(winsorize))
        upper = float(x.quantile(1.0 - winsorize))
        if upper > lower:
            x = x.clip(lower=lower, upper=upper)

    xmin = float(np.nanmin(x.to_numpy()))
    xmax = float(np.nanmax(x.to_numpy()))

    if np.isclose(xmax, xmin):
        # No spread: the signal cannot rank anything. Return a neutral value
        # rather than 0.0, which would silently delete this signal's weight.
        return pd.Series(float(constant_fill), index=x.index, dtype=float)

    out = (x - xmin) / (xmax - xmin)
    if clip:
        out = out.clip(lower=0.0, upper=1.0)
    return out


def rank_normalise(series: pd.Series, constant_fill: float = 0.5) -> pd.Series:
    """Scale a series to [0, 1] by percentile position, averaging ties."""
    x = _as_float(series)

    if len(x) == 0:
        return pd.Series([], index=x.index, dtype=float)
    if x.isna().all():
        return pd.Series(np.nan, index=x.index, dtype=float)

    n_distinct = int(x.nunique(dropna=True))
    if n_distinct <= 1:
        return pd.Series(float(constant_fill), index=x.index, dtype=float)

    ranks = x.rank(method="average", na_option="keep")
    n = float(ranks.max())
    if n <= 1.0:
        return pd.Series(float(constant_fill), index=x.index, dtype=float)
    return (ranks - 1.0) / (n - 1.0)


def normalise_series(series: pd.Series, cfg: NormalisationConfig) -> pd.Series:
    if cfg.method == "minmax":
        return minmax_normalise(
            series,
            clip=cfg.clip,
            winsorize=float(cfg.winsorize),
            constant_fill=float(cfg.constant_fill),
        )
    if cfg.method == "rank":
        return rank_normalise(series, constant_fill=float(cfg.constant_fill))
    # Unreachable: NormalisationConfig validates the method on construction.
    raise ValueError(f"Unsupported normalisation method: {cfg.method!r}")


def normalise_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    cfg: NormalisationConfig,
    prefix: str = "n_",
) -> pd.DataFrame:
    """Add a normalised ``{prefix}{col}`` companion for each requested column."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot normalise missing columns: {missing}")

    out = df.copy()
    for col in columns:
        out[f"{prefix}{col}"] = normalise_series(out[col], cfg)
    return out
