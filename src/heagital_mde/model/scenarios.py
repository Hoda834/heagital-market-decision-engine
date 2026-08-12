from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd
import yaml

from heagital_mde.model.scoring import ScoringResult, score_and_rank
from heagital_mde.model.scoring.schema import (
    MARKET_WEIGHT_KEYS,
    READINESS_WEIGHT_KEYS,
    MarketWeightConfig,
    ReadinessWeightConfig,
    ScoringConfig,
    clip_01,
)

KNOWN_SCENARIO_KEYS = {
    "description",
    "alpha_delta",
    "friction_weight_delta",
    "market_weight_deltas",
    "readiness_weight_deltas",
}


@dataclass(frozen=True)
class Scenario:
    """A named perturbation of the base scoring configuration.

    Deltas are additive on the *pre-normalisation* weights. Because each weight
    block is renormalised to sum to 1 afterwards, a delta expresses relative
    emphasis rather than an absolute weight.
    """

    name: str
    description: str = ""
    alpha_delta: float = 0.0
    friction_weight_delta: float = 0.0
    market_weight_deltas: Dict[str, float] = field(default_factory=dict)
    readiness_weight_deltas: Dict[str, float] = field(default_factory=dict)

    def apply(self, base: ScoringConfig) -> ScoringConfig:
        market = {
            k: max(0.0, v + float(self.market_weight_deltas.get(k, 0.0)))
            for k, v in base.market_weights.as_dict().items()
        }
        readiness = {
            k: max(0.0, v + float(self.readiness_weight_deltas.get(k, 0.0)))
            for k, v in base.readiness_weights.as_dict().items()
        }

        if sum(market.values()) <= 0.0:
            raise ValueError(f"Scenario '{self.name}' zeroes out every market weight.")
        if sum(readiness.values()) <= 0.0:
            raise ValueError(f"Scenario '{self.name}' zeroes out every readiness weight.")

        from heagital_mde.model.scoring.schema import with_overrides

        return with_overrides(
            base,
            market_weights=MarketWeightConfig(**market),
            readiness_weights=ReadinessWeightConfig(**readiness),
            alpha=clip_01(base.alpha + float(self.alpha_delta)),
            friction_weight=clip_01(base.friction_weight + float(self.friction_weight_delta)),
        )


def _read_deltas(raw: Any, keys: tuple[str, ...], block: str, name: str) -> Dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario '{name}': {block} must be a mapping, got {type(raw).__name__}.")
    unknown = sorted(set(raw) - set(keys))
    if unknown:
        raise ValueError(f"Scenario '{name}': unknown key(s) in {block}: {unknown}. Expected any of {list(keys)}.")
    return {k: float(v) for k, v in raw.items()}


def parse_scenario(name: str, raw: Mapping[str, Any]) -> Scenario:
    if not isinstance(raw, Mapping):
        raise ValueError(f"Scenario '{name}' must be a mapping, got {type(raw).__name__}.")

    unknown = sorted(set(raw) - KNOWN_SCENARIO_KEYS)
    if unknown:
        raise ValueError(
            f"Scenario '{name}': unknown key(s) {unknown}. Expected any of {sorted(KNOWN_SCENARIO_KEYS)}."
        )

    return Scenario(
        name=name,
        description=str(raw.get("description", "")),
        alpha_delta=float(raw.get("alpha_delta", 0.0)),
        friction_weight_delta=float(raw.get("friction_weight_delta", 0.0)),
        market_weight_deltas=_read_deltas(
            raw.get("market_weight_deltas"), MARKET_WEIGHT_KEYS, "market_weight_deltas", name
        ),
        readiness_weight_deltas=_read_deltas(
            raw.get("readiness_weight_deltas"), READINESS_WEIGHT_KEYS, "readiness_weight_deltas", name
        ),
    )


def load_scenarios(path: str | Path) -> Dict[str, Scenario]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scenario config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Scenario config must be a YAML mapping, got {type(data).__name__}: {p}")
    return {name: parse_scenario(name, raw) for name, raw in data.items()}


def run_scenarios(
    df: pd.DataFrame,
    base_config: ScoringConfig,
    scenarios: Mapping[str, Scenario],
) -> Dict[str, ScoringResult]:
    """Score the same data under every scenario."""
    results: Dict[str, ScoringResult] = {}
    for name, scenario in scenarios.items():
        result = score_and_rank(df, config=scenario.apply(base_config), return_result=True)
        assert isinstance(result, ScoringResult)
        results[name] = result
    return results


def build_sensitivity(results: Mapping[str, ScoringResult], base_scenario: str | None = None) -> pd.DataFrame:
    """Summarise how each ICB's rank moves across scenarios.

    ``rank_spread`` is the width of the rank band an ICB occupies. ``stability``
    labels the decision: an ICB inside the cut-off under every scenario is
    ``robust``, one outside under every scenario is ``excluded``, and one that
    crosses the line is ``fragile`` — the only group whose inclusion actually
    depends on strategic assumptions.
    """
    if not results:
        raise ValueError("No scenario results to summarise.")

    frames = []
    for name, result in results.items():
        frame = result.ranking[["icb_code", "icb_name", "rank", "final_score", "recommended_included"]].copy()
        frame["scenario"] = name
        frames.append(frame)

    stacked = pd.concat(frames, ignore_index=True)

    summary = (
        stacked.groupby(["icb_code", "icb_name"], dropna=False)
        .agg(
            best_rank=("rank", "min"),
            worst_rank=("rank", "max"),
            mean_rank=("rank", "mean"),
            mean_final_score=("final_score", "mean"),
            scenarios_included=("recommended_included", "sum"),
            scenario_count=("recommended_included", "size"),
        )
        .reset_index()
    )

    summary["rank_spread"] = summary["worst_rank"] - summary["best_rank"]
    summary["mean_rank"] = summary["mean_rank"].round(2)

    def classify(row: pd.Series) -> str:
        if row["scenarios_included"] == row["scenario_count"]:
            return "robust"
        if row["scenarios_included"] == 0:
            return "excluded"
        return "fragile"

    summary["stability"] = summary.apply(classify, axis=1)

    if base_scenario and base_scenario in results:
        base_ranks = results[base_scenario].ranking.set_index("icb_code")["rank"]
        summary["base_rank"] = summary["icb_code"].map(base_ranks)
        summary = summary.sort_values(by=["base_rank"], kind="mergesort").reset_index(drop=True)
    else:
        summary = summary.sort_values(by=["mean_rank"], kind="mergesort").reset_index(drop=True)

    return summary
