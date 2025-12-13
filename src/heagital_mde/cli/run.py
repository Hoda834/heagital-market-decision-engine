from __future__ import annotations

from pathlib import Path

import pandas as pd

from heagital_mde.io.load_icb import load_icb_features
from heagital_mde.io.validate import validate_icb_features
from heagital_mde.model.scoring import score_and_rank


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    input_data_path = project_root / "data" / "raw" / "data.csv"
    scoring_config_path = project_root / "src" / "heagital_mde" / "config" / "scoring_config.yml"

    output_dir = project_root / "data" / "outputs" / "rankings"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ICB data...")
    df = load_icb_features(input_data_path)

    print("Validating input data...")
    validate_icb_features(df)

    print("Scoring and ranking ICBs...")
    ranked = score_and_rank(df, scoring_config_path)

    output_path = output_dir / "icb_opportunity_ranking_basecase.csv"
    ranked.to_csv(output_path, index=False)

    print(f"Ranking completed. Output written to: {output_path}")
    print(f"Top {ranked['recommended_cutoff_top_n'].iloc[0]} ICBs flagged for initial rollout.")


if __name__ == "__main__":
    main()
