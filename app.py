import io
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from heagital_mde.io.load_icb import load_icb_features
from heagital_mde.io.validate import validate_icb_features
from heagital_mde.model.scoring import score_and_rank


st.set_page_config(page_title="Heagital Market Decision Engine", layout="wide")


def normalise_weights_to_one(w1: float, w2: float, w3: float) -> Tuple[float, float, float]:
    total = w1 + w2 + w3
    if total <= 0:
        return 0.45, 0.35, 0.20
    return w1 / total, w2 / total, w3 / total


def read_template_bytes(template_path: Path) -> bytes:
    if not template_path.exists():
        return b""
    return template_path.read_bytes()


def write_temp_csv(uploaded_bytes: bytes) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / "uploaded_data.csv"
    tmp_path.write_bytes(uploaded_bytes)
    return tmp_path


def ensure_dataframe(ranked: object) -> pd.DataFrame:
    if isinstance(ranked, pd.DataFrame):
        return ranked.copy()
    if isinstance(ranked, list):
        return pd.DataFrame(ranked)
    raise TypeError("Unsupported results type returned by score_and_rank.")


def normalise_region_label(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "Unknown"
    return str(x).strip()


def build_region_pivot(df_view: pd.DataFrame) -> pd.DataFrame:
    df = df_view.copy()

    required = {
        "region",
        "icb_code",
        "recommended_included",
        "opportunity_score",
        "n_clinical_risk",
        "n_adoption_readiness",
        "n_procurement_friction",
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot build regional summary. Missing columns: {missing}")

    df["region"] = df["region"].map(normalise_region_label)
    df["recommended_included"] = df["recommended_included"].astype(bool)

    pivot = (
        df.groupby("region", dropna=False)
        .agg(
            total_icbs=("icb_code", "count"),
            included_icbs=("recommended_included", "sum"),
            included_rate=("recommended_included", "mean"),
            avg_opportunity_score=("opportunity_score", "mean"),
            avg_n_clinical_risk=("n_clinical_risk", "mean"),
            avg_n_adoption_readiness=("n_adoption_readiness", "mean"),
            avg_n_procurement_friction=("n_procurement_friction", "mean"),
        )
        .reset_index()
    )

    pivot["included_rate"] = (pivot["included_rate"] * 100.0).round(1)

    pivot = pivot.sort_values(
        by=["included_icbs", "avg_opportunity_score", "total_icbs"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    return pivot


def build_region_opportunity(df_view: pd.DataFrame) -> pd.DataFrame:
    df = df_view.copy()

    required = {"region", "opportunity_score"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot build region opportunity summary. Missing columns: {missing}")

    df["region"] = df["region"].map(normalise_region_label)

    out = (
        df.groupby("region", dropna=False)
        .agg(avg_opportunity_score=("opportunity_score", "mean"))
        .reset_index()
    )

    out["avg_opportunity_score"] = pd.to_numeric(out["avg_opportunity_score"], errors="coerce")
    out = out.dropna(subset=["avg_opportunity_score"]).reset_index(drop=True)
    out = out.sort_values(by="avg_opportunity_score", ascending=False, kind="mergesort").reset_index(drop=True)
    return out


def apply_region_centroids(region_scores: pd.DataFrame) -> pd.DataFrame:
    centroids = {
        "North East and Yorkshire": {"lat": 54.8, "lon": -1.8},
        "North West": {"lat": 54.0, "lon": -2.8},
        "Midlands": {"lat": 52.8, "lon": -1.5},
        "East of England": {"lat": 52.3, "lon": 0.5},
        "London": {"lat": 51.5, "lon": -0.1},
        "South East": {"lat": 51.2, "lon": 0.8},
        "South West": {"lat": 50.9, "lon": -3.5},
    }

    df = region_scores.copy()
    df["lat"] = df["region"].map(lambda r: centroids.get(str(r), {}).get("lat"))
    df["lon"] = df["region"].map(lambda r: centroids.get(str(r), {}).get("lon"))
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return df


def render_web_map(region_scores_with_coords: pd.DataFrame) -> None:
    fig = px.scatter_mapbox(
        region_scores_with_coords,
        lat="lat",
        lon="lon",
        color="avg_opportunity_score",
        size="avg_opportunity_score",
        hover_name="region",
        hover_data={"avg_opportunity_score": ":.3f"},
        zoom=5,
        height=520,
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        title="UK opportunity score by region",
    )

    st.plotly_chart(fig, use_container_width=True)


def prettify_columns_for_display(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns=mapping)
    return out


def main() -> None:
    st.title("Heagital Market Decision Engine")
    st.caption("Upload an ICB level dataset using the provided template. Then run the ranking and review results.")

    project_root = Path(__file__).resolve().parent
    template_path = project_root / "data" / "template" / "icb_input_template.csv"
    scoring_config_path = project_root / "src" / "heagital_mde" / "config" / "scoring_config.yml"

    with st.sidebar:
        st.header("Step 1  Download template")
        template_bytes = read_template_bytes(template_path)

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

        w1n, w2n, w3n = normalise_weights_to_one(w1, w2, w3)
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
        data_path = write_temp_csv(uploaded.getvalue())

        df_in = load_icb_features(data_path)
        validate_icb_features(df_in)

        ranked = score_and_rank(df_in, scoring_config_path)
        df_ranked = ensure_dataframe(ranked)

        df_ranked["recommended_cutoff_top_n"] = int(top_n)
        df_ranked["recommended_included"] = df_ranked["rank"].astype(int) <= int(top_n)

        csv_bytes = df_ranked.to_csv(index=False).encode("utf-8")
        included_count = int(df_ranked["recommended_included"].sum())

        region_pivot = build_region_pivot(df_ranked)
        region_scores = build_region_opportunity(df_ranked)
        region_scores_map = apply_region_centroids(region_scores)

    except Exception as e:
        st.error(f"Run failed: {e}")
        return

    st.success("Ranking completed.")

    display_ranked_cols = {
        "rank": "Rank",
        "icb_code": "ICB Code",
        "icb_name": "ICB Name",
        "region": "Region",
        "opportunity_score": "Opportunity Score",
        "n_clinical_risk": "Clinical risk score",
        "n_adoption_readiness": "Adoption readiness score",
        "n_procurement_friction": "Procurement friction score",
        "recommended_cutoff_top_n": "Top N cut off",
        "recommended_included": "Included in Top N",
    }

    display_pivot_cols = {
        "region": "Region",
        "total_icbs": "Total ICBs",
        "included_icbs": "ICBs included in Top N",
        "included_rate": "Inclusion rate (percent)",
        "avg_opportunity_score": "Average opportunity score",
        "avg_n_clinical_risk": "Average clinical risk score",
        "avg_n_adoption_readiness": "Average adoption readiness score",
        "avg_n_procurement_friction": "Average procurement friction score",
    }

    display_region_scores_cols = {
        "region": "Region",
        "avg_opportunity_score": "Average opportunity score",
        "lat": "Latitude",
        "lon": "Longitude",
    }

    df_ranked_display = prettify_columns_for_display(df_ranked, display_ranked_cols)
    region_pivot_display = prettify_columns_for_display(region_pivot, display_pivot_cols)
    region_scores_display = prettify_columns_for_display(region_scores, {"region": "Region", "avg_opportunity_score": "Average opportunity score"})
    region_scores_map_display = prettify_columns_for_display(region_scores_map, display_region_scores_cols)

    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        st.subheader("ICB ranking")
        st.dataframe(df_ranked_display, use_container_width=True, hide_index=True)

        st.subheader("Regional summary")
        st.dataframe(region_pivot_display, use_container_width=True, hide_index=True)

        st.subheader("UK opportunity map")
        render_web_map(region_scores_map)

        st.subheader("Region opportunity score summary")
        st.dataframe(region_scores_display, use_container_width=True, hide_index=True)

        with st.expander("Map coordinates used"):
            st.dataframe(region_scores_map_display, use_container_width=True, hide_index=True)

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
