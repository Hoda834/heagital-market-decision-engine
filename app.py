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
         "final_score",
         "market_score",
         "readiness_score",
         "n_register",
         "n_prevalence",
         "n_treatment_gap",
         "n_warfarin_proxy",
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
             avg_market_score=("market_score", "mean"),
             avg_readiness_score=("readiness_score", "mean"),
             avg_final_score=("final_score", "mean"),
             avg_n_register=("n_register", "mean"),
             avg_n_prevalence=("n_prevalence", "mean"),
             avg_n_treatment_gap=("n_treatment_gap", "mean"),
             avg_n_warfarin_proxy=("n_warfarin_proxy", "mean"),
         )
         .reset_index()
     )
 
     pivot["included_rate"] = (pivot["included_rate"] * 100.0).round(1)
 
     pivot = pivot.sort_values(
         by=["included_icbs", "avg_final_score", "total_icbs"],
         ascending=[False, False, False],
         kind="mergesort",
     ).reset_index(drop=True)
 
     return pivot
 
 
 def build_region_opportunity(df_view: pd.DataFrame) -> pd.DataFrame:
     df = df_view.copy()
 
     required = {"region", "final_score"}
     missing = [c for c in required if c not in df.columns]
     if missing:
         raise ValueError(f"Cannot build region opportunity summary. Missing columns: {missing}")
 
     df["region"] = df["region"].map(normalise_region_label)
 
     out = (
         df.groupby("region", dropna=False)
         .agg(avg_final_score=("final_score", "mean"))
         .reset_index()
     )
 
     out["avg_final_score"] = pd.to_numeric(out["avg_final_score"], errors="coerce")
     out = out.dropna(subset=["avg_final_score"]).reset_index(drop=True)
     out = out.sort_values(by="avg_final_score", ascending=False, kind="mergesort").reset_index(drop=True)
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
@@ -158,165 +153,167 @@ def render_web_map(region_scores_with_coords: pd.DataFrame) -> None:
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
     return df.copy().rename(columns=mapping)
 
 
 def main() -> None:
     st.title("Heagital Market Decision Engine")
     st.caption("Upload an ICB level dataset using the provided template. Then run the ranking and review results.")
 
     project_root = Path(__file__).resolve().parent
     template_path = project_root / "data" / "template" / "icb_input_template.csv"
     scoring_config_path = project_root / "src" / "heagital_mde" / "config" / "scoring_config.yml"
     template_bytes = read_template_bytes(template_path)

     st.download_button(
         label="Download input template (CSV)",
         data=template_bytes,
         file_name="icb_input_template.csv",
         mime="text/csv",
     )
 
     uploaded = st.file_uploader("Upload completed ICB CSV", type=["csv"])
 
     with st.expander("Weight controls", expanded=True):
         w1 = st.slider("AF register weight", 0.0, 1.0, 0.30, 0.01)
         w2 = st.slider("Prevalence weight", 0.0, 1.0, 0.20, 0.01)
         w3 = st.slider("Treatment gap weight", 0.0, 1.0, 0.30, 0.01)
         w4 = st.slider("Warfarin proxy weight", 0.0, 1.0, 0.20, 0.01)
 
     w1n, w2n, w3n = normalise_weights_to_one(w1, w2, w3)
     st.caption(f"Normalised first three weights: register {w1n:.2f}, prevalence {w2n:.2f}, treatment gap {w3n:.2f}")
 
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
 
         market_weights = MarketWeightConfig(
             register=float(w1),
             prevalence=float(w2),
             treatment_gap=float(w3),
             warfarin_proxy=float(w4),
         )
 
         if w4 <= 0:
             readiness_weights = ReadinessWeightConfig(treatment_gap=1.0, warfarin_proxy=0.0)
         elif w4 >= 1:
             readiness_weights = ReadinessWeightConfig(treatment_gap=0.0, warfarin_proxy=1.0)
         else:
             readiness_weights = ReadinessWeightConfig(treatment_gap=float(1.0 - w4), warfarin_proxy=float(w4))
 
         ranked = score_and_rank(
             df_in,
             scoring_config_path,
             market_weights_override=market_weights,
             readiness_weights_override=readiness_weights,
             alpha_override=float(w1n),
         )
 
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
         "n_register": "Normalised register",
         "n_prevalence": "Normalised prevalence",
         "n_treatment_gap": "Normalised treatment gap",
         "n_warfarin_proxy": "Normalised warfarin proxy",
         "final_score": "Final score",
         "market_score": "Market score",
         "readiness_score": "Readiness score",
         "recommended_cutoff_top_n": "Top N cut off",
         "recommended_included": "Included in Top N",
     }
 
     display_pivot_cols = {
         "region": "Region",
         "total_icbs": "Total ICBs",
         "included_icbs": "ICBs included in Top N",
         "included_rate": "Inclusion rate (percent)",
         "avg_market_score": "Average market score",
         "avg_readiness_score": "Average readiness score",
         "avg_n_register": "Average normalised register",
         "avg_n_prevalence": "Average normalised prevalence",
         "avg_n_treatment_gap": "Average normalised treatment gap",
         "avg_n_warfarin_proxy": "Average normalised warfarin proxy",
         "avg_final_score": "Average final score",
     }
 
     df_ranked_display = prettify_columns_for_display(df_ranked, display_ranked_cols)
     region_pivot_display = prettify_columns_for_display(region_pivot, display_pivot_cols)
     region_scores_display = prettify_columns_for_display(
         region_scores, {"region": "Region", "avg_final_score": "Average final score"}
     )
 
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
 
     with c2:
         st.subheader("Download")
         st.download_button(
             label="Download ranking CSV",
             data=csv_bytes,
             file_name="icb_opportunity_ranking.csv",
             mime="text/csv",

         )
 
 
 if __name__ == "__main__":
     main()
