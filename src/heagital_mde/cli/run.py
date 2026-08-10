from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from heagital_mde import __version__
from heagital_mde.io.load_icb import load_icb_features, resolve_input_path
from heagital_mde.io.validate import ValidationError, check_icb_features
from heagital_mde.model.scenarios import build_sensitivity, load_scenarios, run_scenarios
from heagital_mde.model.scoring import ScoringResult, score_and_rank
from heagital_mde.model.scoring.schema import load_scoring_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw"
DEFAULT_SCORING_CONFIG = PACKAGE_ROOT / "config" / "scoring_config.yml"
DEFAULT_SCENARIO_CONFIG = PACKAGE_ROOT / "config" / "scenarios.yml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "rankings"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_output_dir(path: Path) -> Path:
    """Create the output directory, refusing to delete an existing file."""
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(
            f"Output path exists and is a file, not a directory: {path}. "
            "Move or remove it, or pass a different --output-dir."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="heagital-mde",
        description="Rank NHS ICBs for market entry and write an auditable ranking to disk.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Input data file or directory (default: {DEFAULT_INPUT}).")
    parser.add_argument("--config", type=Path, default=DEFAULT_SCORING_CONFIG, help="Scoring config YAML.")
    parser.add_argument("--scenario-config", type=Path, default=DEFAULT_SCENARIO_CONFIG, help="Scenario config YAML.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV and audit output.")
    parser.add_argument("--top-n", type=int, default=None, help="Override the recommended cut-off.")
    parser.add_argument("--alpha", type=float, default=None, help="Override the market/readiness blend (0-1).")
    parser.add_argument(
        "--gap-units",
        choices=("auto", "percent", "fraction"),
        default="auto",
        help="How to read the Treatment Gap column (default: auto-detect).",
    )
    parser.add_argument("--scenarios", action="store_true", help="Also run every scenario and write a sensitivity table.")
    parser.add_argument("--strict", action="store_true", help="Treat validation warnings as errors.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        input_file = resolve_input_path(args.input)
        output_dir = _prepare_output_dir(args.output_dir)

        print(f"Loading ICB data from {input_file} ...")
        df = load_icb_features(input_file, gap_units=args.gap_units)

        print("Validating input data ...")
        report = check_icb_features(df)
        for message in report.warnings:
            print(f"  warning: {message}")
        if args.strict and report.warnings:
            report.errors.extend(f"(strict) {m}" for m in report.warnings)
        report.raise_if_failed()

        base_config = load_scoring_config(args.config)

        print("Scoring and ranking ICBs ...")
        result = score_and_rank(
            df,
            config=base_config,
            alpha_override=args.alpha,
            top_n_override=args.top_n,
            return_result=True,
        )
        assert isinstance(result, ScoringResult)

        for message in result.warnings:
            print(f"  warning: {message}")

        ranking_path = output_dir / "icb_opportunity_ranking_basecase.csv"
        result.ranking.to_csv(ranking_path, index=False)

        audit = {
            "engine_version": __version__,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input_file": str(input_file),
            "input_sha256": _file_digest(input_file),
            "scoring_config": str(args.config),
            "gap_units": args.gap_units,
            "outputs": {"ranking": ranking_path.name},
            **result.audit(),
        }

        if args.scenarios:
            print("Running scenarios ...")
            scenarios = load_scenarios(args.scenario_config)
            scenario_results = run_scenarios(df, result.config, scenarios)

            scenario_dir = output_dir / "scenarios"
            scenario_dir.mkdir(parents=True, exist_ok=True)
            for name, scenario_result in scenario_results.items():
                scenario_result.ranking.to_csv(scenario_dir / f"icb_ranking_{name}.csv", index=False)

            base_name = "base_case" if "base_case" in scenario_results else next(iter(scenario_results))
            sensitivity = build_sensitivity(scenario_results, base_scenario=base_name)
            sensitivity_path = output_dir / "icb_sensitivity.csv"
            sensitivity.to_csv(sensitivity_path, index=False)

            counts = sensitivity["stability"].value_counts().to_dict()
            audit["scenarios"] = {
                "config": str(args.scenario_config),
                "names": list(scenario_results),
                "stability_counts": {k: int(v) for k, v in counts.items()},
            }
            audit["outputs"]["sensitivity"] = sensitivity_path.name
            audit["outputs"]["scenario_dir"] = scenario_dir.name

            print(
                f"  robust: {counts.get('robust', 0)} | "
                f"fragile: {counts.get('fragile', 0)} | "
                f"excluded: {counts.get('excluded', 0)}"
            )
            print(f"Sensitivity written to: {sensitivity_path}")

        audit_path = output_dir / "run_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

        included = int(result.ranking["recommended_included"].sum())
        print(f"Ranking completed. Output written to: {ranking_path}")
        print(f"Audit record written to: {audit_path}")
        print(f"{included} of {len(result.ranking)} ICBs flagged for initial rollout.")
        return 0

    except (FileNotFoundError, NotADirectoryError, ValidationError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
