from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from heagital_mde.io.load_icb import load_icb_features
from heagital_mde.io.validate import validate_icb_features
from heagital_mde.model.scoring import score_and_rank


st.set_page_config(page_title="Heagital Market Decision Engine", layout="wide")


def _normalise_weights_to_one(w1: float, w2: float, w3: float) -> Tuple[float, float, float]:
    total = w1 + w2 + w3
    if total <= 0:
        return 0.45, 0.35, 0.20
    return w1 / total, w2 / total, w3 / total


def _read_template_bytes(template_path: Path) -> bytes:
    if not template_path.exists():
        return b""
    return template_path.read_bytes()


def _write_temp_csv(uploaded_bytes: bytes) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / "uploaded_data.csv"
    tmp_path.write_bytes(uploaded_bytes)
    return tmp_path


def _to_csv_bytes(rows: List[Dict[str, object]]) -> bytes:
    import csv

    if not rows:
        return b""

    buffer = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def main() -> None:
    st.title("Heagital Market Decision Engine")
    st.caption(
        "Upload an ICB level dataset using the provided template. Then set decision weights and run the ranking."
    )

    project_root = Path(__file__).resolve().parent
    template_path = project_root / "data" / "template" / "icb_input_template.csv"
    scoring_config_path = project_root / "src" / "heagital_mde" / "config" / "scoring_config.yml"

    with st.sidebar:
        st.header("Step 1  Download template")
        template_bytes = _read_template_bytes(template_path)

        if template_bytes:
            st.download_button(
                label="Download CSV template",
                data=template_bytes,
                file_name="icb_input_template.csv",
                mime="text/csv",
            )
        else:
            st.warning("Template file not found. Expected at data/template/icb_input_template.csv")

        st.divider()
        st.header("Step 2  Upload your data")
        uploaded = st.file_uploader("Upload completed CSV", type=["csv"])

        st.divider()
        st.header("Step 3  Decision weights")
        w1 = st.slider("Clinical risk weight", 0.0, 1.0, 0.45, 0.01)
        w2 = st.slider("Adoption readiness weight", 0.0, 1.0, 0.35, 0.01)
        w3 = st.slider("Procurement friction weight", 0.0, 1.0, 0.20, 0.01)

        w1n, w2n, w3n = _normalise_weights_to_one(w1, w2, w3)
        st.caption(f"Normalised weights: clinical {w1n:.2f}, adoption {w2n:.2f}, friction {w3n:.2f}")

        top_n = st.number_input("Recommended cut off Top N", min_value=1, max_value=100, value=15, step=1)

        run = st.button("Run ranking", type="primary", disabled=(uploaded is None))

    if uploaded is None:
        st.info("Download the template, fill it, upload the CSV, then run the ranking.")
        return

    if not scoring_config_path.exists():
        st.error("Missing scoring config. Expected at src/heagital_mde/config/scoring_config.yml")
        return

    if not run:
        st.stop()

    try:
        data_path = _write_temp_csv(uploaded.getvalue())

        df = load_icb_features(data_path)
        validate_icb_features(df)

        ranked = score_and_rank(df, scoring_config_path)

        if isinstance(ranked, pd.DataFrame):
            ranked["recommended_cutoff_top_n"] = int(top_n)
            ranked["recommended_included"] = ranked["rank"].astype(int) <= int(top_n)
            csv_bytes = ranked.to_csv(index=False).encode("utf-8")

            included_count = int(ranked["recommended_included"].sum())
        else:
            for r in ranked:
                r["recommended_cutoff_top_n"] = int(top_n)
                r["recommended_included"] = int(r["rank"]) <= int(top_n)

            csv_bytes = _to_csv_bytes(ranked)
            included_count = len([r for r in ranked if bool(r.get("recommended_included"))])

    except Exception as e:
        st.error(f"Run failed: {e}")
        return

    st.success("Ranking completed.")

    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        st.subheader("ICB ranking")
        st.dataframe(ranked, use_container_width=True, hide_index=True)

    with c2:
        st.subheader("Download")
        st.download_button(
            label="Download ranking CSV",
            data=csv_bytes,
            file_name="icb_opportunity_ranking.csv",
            mime="text/csv",
        )
        st.metric("Included ICBs", included_count)


if __name__ == "__main__":
    main()
