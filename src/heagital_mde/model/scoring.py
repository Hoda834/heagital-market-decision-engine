from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml

from heagital_mde.model.normalise import NormalisationConfig, normalise_columns


@dataclass(frozen=True)
class WeightConfig:
    clinical_risk: float
    adoption_readiness: float
    procurement_friction: float


def _load_yaml(path: str | Path) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scoring_config(path: str | Path) -> Tuple[WeightConfig, NormalisationConfig, int]:
    cfg = _load_yaml(path)

    w = cfg.get("weights", {})
    weights = WeightConfig(
        clinical_risk=float(w.get("clinical_risk", 0.45)),
        adoption_readiness=float(w.get("adoption_readiness", 0.35)),
        procurement_friction=float(w.get("procurement_friction", 0.20)),
    )

    n = cfg.get("normalisation", {})
    norm_cfg = NormalisationConfig(
        method=str(n.get("method", "minmax")),
        clip=bool(n.get("clip", True)),
    )

    cutoff_cfg = cfg.get("cutoff", {})
    top_n = int(cutoff_cfg.get("top_n", 15))

    return weights, norm_cfg, top_n


def _normalise_weights_to_one(w: WeightConfig) -> WeightConfig:
    total = w.clinical_risk + w.adoption_readiness + w.procurement_friction
    if total <= 0:
        raise ValueError("Weight sum must be greater than 0.")
    return WeightConfig(
        clinical_risk=w.clinical_risk / total,
        adoption_readiness=w.adoption_readiness / total,
        procurement_friction=w.procurement_friction / total,
    )


def _build_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = ["af_register", "treatment_gap", "warfarin_proxy"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for scoring: {missing}")

    out["clinical_risk"] = out["treatment_gap"].astype(float)
    out["adoption_readiness"] = out["warfarin_proxy"].astype(float)

    if "procurement_friction" in out.columns:
        out["procurement_friction"] = pd.to_numeric(out["procurement_friction"], errors="coerce").astype(float)
    else:
        out["procurement_friction"] = 0.5

    return out


def score_and_rank(
    df: pd.DataFrame,
    scoring_config_path: str | Path,
    weights_override: WeightConfig | None = None,
) -> pd.DataFrame:
    weights, norm_cfg, top_n = load_scoring_config(scoring_config_path)

    if weights_override is not None:
        weights = weights_override

    weights = _normalise_weights_to_one(weights)

    base = _build_signals(df)

    base = normalise_columns(
        base,
        columns=["clinical_risk", "adoption_readiness", "procurement_friction"],
        cfg=norm_cfg,
        prefix="n_",
    )

    base["opportunity_score"] = (
        weights.clinical_risk * base["n_clinical_risk"]
        + weights.adoption_readiness * base["n_adoption_readiness"]
        - weights.procurement_friction * base["n_procurement_friction"]
    )

    cols = ["icb_code", "icb_name"]
    if "region" in base.columns:
        cols.append("region")
    cols += [
        "opportunity_score",
        "n_clinical_risk",
        "n_adoption_readiness",
        "n_procurement_friction",
    ]

    out = base[cols].copy()
    out = out.sort_values(by="opportunity_score", ascending=False, kind="mergesort").reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)

    out["recommended_cutoff_top_n"] = top_n
    out["recommended_included"] = out["rank"] <= top_n

    return out

