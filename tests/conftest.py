from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "src" / "heagital_mde" / "config"

HEADER = (
    "ICB ODS code,ICB name,Register,Prevalence (%),Treatment Gap (%),Warfarin Item icb,Region"
)

OPTIONAL_HEADER = HEADER + ",Digital Maturity (0-1),Procurement Friction (0-1)"


@pytest.fixture
def scoring_config_path() -> Path:
    return CONFIG_DIR / "scoring_config.yml"


@pytest.fixture
def scenario_config_path() -> Path:
    return CONFIG_DIR / "scenarios.yml"


@pytest.fixture
def write_csv(tmp_path: Path):
    """Write a CSV with the standard header and return its path."""

    def _write(rows: str, name: str = "input.csv", header: str = HEADER) -> Path:
        path = tmp_path / name
        path.write_text(header + "\n" + rows, encoding="utf-8")
        return path

    return _write


@pytest.fixture
def sample_rows() -> str:
    return (
        "QA1,Alpha ICB,\"74,319\",2.68,10.37,205371,North West\n"
        "QB2,Beta ICB,\"51,004\",2.10,14.20,140233,London\n"
        "QC3,Gamma ICB,\"33,870\",1.95,18.60,90112,Midlands\n"
        "QD4,Delta ICB,\"22,145\",1.60,22.40,60451,South West\n"
    )


@pytest.fixture
def sample_rows_with_optional() -> str:
    return (
        "QA1,Alpha ICB,\"74,319\",2.68,10.37,205371,North West,0.80,0.10\n"
        "QB2,Beta ICB,\"51,004\",2.10,14.20,140233,London,0.30,0.70\n"
        "QC3,Gamma ICB,\"33,870\",1.95,18.60,90112,Midlands,0.60,0.40\n"
        "QD4,Delta ICB,\"22,145\",1.60,22.40,60451,South West,0.20,0.90\n"
    )
