from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from heagital_mde.data.schema import (
    FRICTION_SIGNAL,
    MARKET_SIGNALS,
    READINESS_SIGNALS,
)
from heagital_mde.model.normalise import normalise_columns
from heagital_mde.model.scoring.combine import compute_final_score
from heagital_mde.model.scoring.market import compute_market_score
from heagital_mde.model.scoring.rank import rank_and_flag
from heagital_mde.model.scoring.readiness import (
    compute_readiness_score,
    effective_readiness_weights,
)
from heagital_mde.model.scoring.schema import (
    MarketWeightConfig,
    ReadinessWeightConfig,
    ScoringConfig,
    load_scoring_config,
    with_overrides,
)

__all__ = [
    "MarketWeightConfig",
    "ReadinessWeightConfig",
    "ScoringConfig",
    "ScoringResult",
    "load_scoring_config",
    "score_and_rank",
    "with_overrides",
]

#: Above this correlation, the market and readiness pillars are measuring
#: substantially the same thing and alpha stops being a meaningful lever.
COLLINEARITY_WARNING_THRESHOLD = 0.85

#: Below this many ICBs, min-max normalisation has too little spread to
#: produce a defensible ranking.
MIN_ROWS_FOR_STABLE_RANKING = 3


@dataclass
class ScoringResult:
    """A ranking plus everything needed to explain and reproduce it."""

    ranking: pd.DataFrame
    config: ScoringConfig
    effective_readiness_weights: Dict[str, float]
    signals_used: List[str] = field(default_factory=list)
    signals_absent: List[str] = field(default_factory=list)
    dropped_rows: int = 0
    pillar_correlation: float | None = None
    warnings: List[str] = field(default_factory=list)

    def audit(self) -> Dict[str, Any]:
        """A JSON-serialisable record of how this ranking was produced."""
        return {
            "config": self.config.as_dict(),
            "effective_readiness_weights": self.effective_readiness_weights,
            "signals_used": list(self.signals_used),
            "signals_absent": list(self.signals_absent),
            "rows_scored": int(len(self.ranking)),
            "rows_dropped_incomplete": int(self.dropped_rows),
            "market_readiness_correlation": (
                None if self.pillar_correlation is None else round(float(self.pillar_correlation), 4)
            ),
            "warnings": list(self.warnings),
        }


def _build_signals(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Coerce signal columns to numeric and drop rows missing a required one."""
    out = df.copy()

    # Backwards compatibility with files loaded by pre-0.2 versions.
    if "register" not in out.columns and "af_register" in out.columns:
        out = out.rename(columns={"af_register": "register"})

    missing = [c for c in MARKET_SIGNALS if c not in out.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for scoring: {missing}. "
            f"Expected all of {list(MARKET_SIGNALS)} from the loader."
        )

    numeric_cols = list(MARKET_SIGNALS)
    for col in list(READINESS_SIGNALS) + [FRICTION_SIGNAL]:
        if col in out.columns and col not in numeric_cols:
            numeric_cols.append(col)

    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    before = len(out)
    out = out.dropna(subset=list(MARKET_SIGNALS)).reset_index(drop=True)
    return out, before - len(out)


def score_and_rank(
    df: pd.DataFrame,
    scoring_config_path: str | Path | None = None,
    config: ScoringConfig | None = None,
    market_weights_override: MarketWeightConfig | None = None,
    readiness_weights_override: ReadinessWeightConfig | None = None,
    alpha_override: float | None = None,
    friction_weight_override: float | None = None,
    top_n_override: int | None = None,
    return_result: bool = False,
) -> pd.DataFrame | ScoringResult:
    """Score, rank and flag ICBs.

    Pass either ``scoring_config_path`` or an already-loaded ``config``.
    Returns the ranking DataFrame by default, or the full :class:`ScoringResult`
    (ranking + audit metadata) when ``return_result=True``.
    """
    if config is None:
        if scoring_config_path is None:
            raise ValueError("Provide either scoring_config_path or config.")
        config = load_scoring_config(scoring_config_path)

    config = with_overrides(
        config,
        market_weights=market_weights_override,
        readiness_weights=readiness_weights_override,
        alpha=alpha_override,
        friction_weight=friction_weight_override,
        top_n=top_n_override,
    )

    warnings: List[str] = []
    base, dropped = _build_signals(df)

    if dropped:
        warnings.append(f"{dropped} row(s) dropped for missing or non-numeric market signals.")

    if len(base) == 0:
        raise ValueError(
            "No scorable rows remain after dropping records with missing market signals. "
            "Check that the input file has at least one complete ICB row."
        )
    if len(base) < MIN_ROWS_FOR_STABLE_RANKING:
        warnings.append(
            f"Only {len(base)} ICB(s) scored. Normalisation needs at least "
            f"{MIN_ROWS_FOR_STABLE_RANKING} rows to produce a meaningful spread; "
            "treat this ranking as indicative only."
        )

    signals_used = list(MARKET_SIGNALS)
    signals_absent: List[str] = []
    for col in READINESS_SIGNALS:
        if col in base.columns:
            if col not in signals_used:
                signals_used.append(col)
        else:
            signals_absent.append(col)

    has_friction = FRICTION_SIGNAL in base.columns
    if has_friction:
        signals_used.append(FRICTION_SIGNAL)
    else:
        signals_absent.append(FRICTION_SIGNAL)
        if config.friction_weight > 0.0:
            warnings.append(
                f"friction_weight={config.friction_weight:.2f} was requested but the input has no "
                f"'{FRICTION_SIGNAL}' column; the friction penalty is not applied."
            )

    base = normalise_columns(base, columns=signals_used, cfg=config.normalisation, prefix="n_")

    # Rows are only dropped for missing *market* signals, so an optional signal
    # with blank cells would otherwise propagate NaN all the way to final_score
    # and silently sink those ICBs to the bottom of the ranking. Score the gaps
    # at the neutral value instead, and say how many were filled.
    neutral = float(config.normalisation.constant_fill)
    for signal in signals_used:
        if signal in MARKET_SIGNALS:
            continue
        column = f"n_{signal}"
        blank = base[column].isna()
        if blank.any():
            base.loc[blank, column] = neutral
            warnings.append(
                f"{signal} is blank for {int(blank.sum())} ICB(s); those rows were scored at the "
                f"neutral value {neutral:g} for this signal rather than being excluded."
            )

    base["market_score"] = compute_market_score(base, config.market_weights)
    base["readiness_score"] = compute_readiness_score(base, config.readiness_weights)

    friction_series = base[f"n_{FRICTION_SIGNAL}"] if has_friction else None
    base["friction_score"] = friction_series if has_friction else 0.0

    base["final_score"] = compute_final_score(
        base["market_score"],
        base["readiness_score"],
        alpha=config.alpha,
        friction_score=friction_series,
        friction_weight=config.friction_weight if has_friction else 0.0,
    )

    correlation: float | None = None
    # Correlation is undefined when either pillar has no spread across ICBs.
    if len(base) > 2 and base["market_score"].nunique() > 1 and base["readiness_score"].nunique() > 1:
        corr = base["market_score"].corr(base["readiness_score"])
        if pd.notna(corr):
            correlation = float(corr)
            if correlation >= COLLINEARITY_WARNING_THRESHOLD:
                remedy = (
                    "Supply a 'digital_maturity' column to give readiness an independent basis."
                    if "digital_maturity" in signals_absent
                    else "Give digital_maturity more readiness weight, or add a further independent "
                    "readiness signal, to make alpha a meaningful lever."
                )
                warnings.append(
                    f"Market and readiness scores correlate at {correlation:.2f}. They are built from "
                    f"overlapping signals, so alpha has limited influence on the ranking. {remedy}"
                )

    id_cols = ["icb_code", "icb_name"]
    missing_ids = [c for c in id_cols if c not in base.columns]
    if missing_ids:
        raise ValueError(f"Missing identifier column(s) required for output: {missing_ids}")

    cols: List[str] = list(id_cols)
    if "region" in base.columns:
        cols.append("region")
    cols += ["market_score", "readiness_score", "friction_score", "final_score"]
    cols += [f"n_{s}" for s in signals_used]

    out = base[cols].copy()
    out = rank_and_flag(out, score_col="final_score", top_n=config.top_n)

    result = ScoringResult(
        ranking=out,
        config=config,
        effective_readiness_weights=effective_readiness_weights(base, config.readiness_weights),
        signals_used=signals_used,
        signals_absent=signals_absent,
        dropped_rows=dropped,
        pillar_correlation=correlation,
        warnings=warnings,
    )

    return result if return_result else out
