from __future__ import annotations

import json
from pathlib import Path

import pytest

from heagital_mde.cli.run import main

HEADER = "ICB ODS code,ICB name,Register,Prevalence (%),Treatment Gap (%),Warfarin Item icb,Region"


def _args(input_path: Path, output_dir: Path, config: Path, *extra: str) -> list[str]:
    return [
        "--input", str(input_path),
        "--output-dir", str(output_dir),
        "--config", str(config),
        *extra,
    ]


def test_cli_writes_ranking_and_audit(tmp_path, write_csv, sample_rows, scoring_config_path):
    input_path = write_csv(sample_rows)
    output_dir = tmp_path / "out"

    assert main(_args(input_path, output_dir, scoring_config_path)) == 0

    ranking = output_dir / "icb_opportunity_ranking_basecase.csv"
    audit = output_dir / "run_audit.json"
    assert ranking.exists() and audit.exists()

    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["rows_scored"] == 4
    assert len(payload["input_sha256"]) == 64
    assert payload["config"]["alpha"] == pytest.approx(0.60)


def test_cli_scenarios_write_sensitivity(tmp_path, write_csv, sample_rows, scoring_config_path, scenario_config_path):
    output_dir = tmp_path / "out"
    exit_code = main(
        _args(
            write_csv(sample_rows),
            output_dir,
            scoring_config_path,
            "--scenarios",
            "--scenario-config", str(scenario_config_path),
            "--top-n", "2",
        )
    )
    assert exit_code == 0
    assert (output_dir / "icb_sensitivity.csv").exists()
    assert (output_dir / "scenarios" / "icb_ranking_base_case.csv").exists()

    payload = json.loads((output_dir / "run_audit.json").read_text(encoding="utf-8"))
    assert "base_case" in payload["scenarios"]["names"]


def test_cli_never_deletes_a_file_sitting_at_the_output_path(tmp_path, write_csv, sample_rows, scoring_config_path):
    """The pre-0.2 CLI unlinked whatever file occupied the output directory path."""
    output_path = tmp_path / "rankings"
    output_path.write_text("important tracked placeholder\n", encoding="utf-8")

    assert main(_args(write_csv(sample_rows), output_path, scoring_config_path)) == 1
    assert output_path.read_text(encoding="utf-8") == "important tracked placeholder\n"


def test_cli_reports_validation_failure_without_traceback(tmp_path, write_csv, scoring_config_path, capsys):
    rows = "QA1,Alpha,1000,2.0,10,500,London\nQA1,Beta,2000,3.0,20,600,London\n"
    assert main(_args(write_csv(rows), tmp_path / "out", scoring_config_path)) == 1
    assert "unique" in capsys.readouterr().err


def test_cli_reports_missing_input(tmp_path, scoring_config_path, capsys):
    assert main(_args(tmp_path / "nope.csv", tmp_path / "out", scoring_config_path)) == 1
    assert "not found" in capsys.readouterr().err


def test_cli_strict_mode_promotes_warnings_to_errors(tmp_path, write_csv, scoring_config_path):
    rows = "QA1,Alpha,1000,2.0,10,500,London\nQB2,Beta,2000,3.0,20,600,London\n"
    args = _args(write_csv(rows), tmp_path / "out", scoring_config_path)
    assert main(args) == 0
    assert main(args + ["--strict"]) == 1


def test_cli_top_n_override_is_applied(tmp_path, write_csv, sample_rows, scoring_config_path):
    import pandas as pd

    output_dir = tmp_path / "out"
    main(_args(write_csv(sample_rows), output_dir, scoring_config_path, "--top-n", "1"))
    ranked = pd.read_csv(output_dir / "icb_opportunity_ranking_basecase.csv")
    assert int(ranked["recommended_included"].sum()) == 1
