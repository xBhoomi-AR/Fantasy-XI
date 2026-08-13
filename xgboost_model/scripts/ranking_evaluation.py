from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "predictions" / "test_2025_26_predictions.csv"
FEATURES = ROOT / "data" / "processed" / "model_features.csv"
PLAYERS = ROOT / "data" / "raw" / "players.csv"
OUT_CSV = ROOT / "reports" / "ranking_evaluation.csv"
OUT_TXT = ROOT / "reports" / "ranking_summary.txt"


def spearman(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 2:
        return np.nan
    return float(x[valid].rank(method="average").corr(y[valid].rank(method="average")))


def top_overlap(group: pd.DataFrame, k: int) -> float:
    pred_top = set(group.nlargest(k, "predicted_points")["row_key"])
    actual_top = set(group.nlargest(k, "target_points")["row_key"])
    denom = min(k, len(group))
    if denom == 0:
        return np.nan
    return float(len(pred_top & actual_top) / denom * 100)


def load_joined() -> pd.DataFrame:
    preds = pd.read_csv(PREDICTIONS)
    actual_cols = [
        "player_id",
        "fixture_id",
        "season",
        "gameweek",
        "team_id",
        "opponent_team_id",
        "position",
        "target_points",
    ]
    actual = pd.read_csv(FEATURES, usecols=actual_cols, engine="python")
    actual = actual[actual["season"].astype(str).eq("2025-26")].copy()
    keys = ["player_id", "fixture_id", "season", "gameweek", "team_id", "opponent_team_id", "position"]
    actual = actual.drop_duplicates(keys, keep="last")
    joined = preds.merge(actual, on=keys, how="inner", validate="one_to_one")
    joined["row_key"] = (
        joined["season"].astype(str)
        + "|"
        + joined["gameweek"].astype(str)
        + "|"
        + joined["fixture_id"].astype(str)
        + "|"
        + joined["player_id"].astype(str)
    )
    joined["actual_rank"] = joined.groupby("gameweek")["target_points"].rank(method="min", ascending=False).astype(int)
    joined["predicted_rank"] = joined.groupby("gameweek")["predicted_points"].rank(method="min", ascending=False).astype(int)
    if PLAYERS.exists():
        names = pd.read_csv(PLAYERS, usecols=["player_id", "player_name", "web_name"])
        names = names.drop_duplicates("player_id", keep="last")
        joined = joined.merge(names, on="player_id", how="left")
    else:
        joined["player_name"] = ""
        joined["web_name"] = ""
    return joined


def build_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "metric_scope": "overall",
            "gameweek": np.nan,
            "position": "ALL",
            "n": len(joined),
            "spearman": spearman(joined["predicted_points"], joined["target_points"]),
            "top10_overlap_pct": np.nan,
            "top20_overlap_pct": np.nan,
            "top50_overlap_pct": np.nan,
        }
    ]
    for gw, group in joined.groupby("gameweek", sort=True):
        rows.append(
            {
                "metric_scope": "gameweek",
                "gameweek": int(gw),
                "position": "ALL",
                "n": len(group),
                "spearman": spearman(group["predicted_points"], group["target_points"]),
                "top10_overlap_pct": top_overlap(group, 10),
                "top20_overlap_pct": top_overlap(group, 20),
                "top50_overlap_pct": top_overlap(group, 50),
            }
        )
    for position, group in joined.groupby("position", sort=True):
        rows.append(
            {
                "metric_scope": "position",
                "gameweek": np.nan,
                "position": position,
                "n": len(group),
                "spearman": spearman(group["predicted_points"], group["target_points"]),
                "top10_overlap_pct": np.nan,
                "top20_overlap_pct": np.nan,
                "top50_overlap_pct": np.nan,
            }
        )
    gw_metrics = pd.DataFrame([row for row in rows if row["metric_scope"] == "gameweek"])
    rows.extend(
        [
            {
                "metric_scope": "gameweek_mean",
                "gameweek": np.nan,
                "position": "ALL",
                "n": int(gw_metrics["n"].sum()),
                "spearman": float(gw_metrics["spearman"].mean()),
                "top10_overlap_pct": float(gw_metrics["top10_overlap_pct"].mean()),
                "top20_overlap_pct": float(gw_metrics["top20_overlap_pct"].mean()),
                "top50_overlap_pct": float(gw_metrics["top50_overlap_pct"].mean()),
            },
            {
                "metric_scope": "gameweek_median",
                "gameweek": np.nan,
                "position": "ALL",
                "n": int(gw_metrics["n"].median()),
                "spearman": float(gw_metrics["spearman"].median()),
                "top10_overlap_pct": float(gw_metrics["top10_overlap_pct"].median()),
                "top20_overlap_pct": float(gw_metrics["top20_overlap_pct"].median()),
                "top50_overlap_pct": float(gw_metrics["top50_overlap_pct"].median()),
            },
        ]
    )
    return pd.DataFrame(rows)


def representative_top10(joined: pd.DataFrame) -> str:
    available = sorted(joined["gameweek"].dropna().unique().astype(int).tolist())
    wanted = [1, 10, 20, 30, max(available)]
    selected = []
    for gw in wanted:
        if gw in available and gw not in selected:
            selected.append(gw)
    lines = []
    for gw in selected:
        top = joined[joined["gameweek"].eq(gw)].sort_values("predicted_points", ascending=False).head(10)
        lines.append(f"\nRepresentative GW{gw} predicted top 10:")
        lines.append("rank, player_id, player, position, predicted_points, target_points, actual_rank")
        for i, row in enumerate(top.itertuples(index=False), start=1):
            name = getattr(row, "web_name", "") or getattr(row, "player_name", "") or ""
            lines.append(
                f"{i}, {row.player_id}, {name}, {row.position}, "
                f"{row.predicted_points:.3f}, {row.target_points:.1f}, {row.actual_rank}"
            )
    return "\n".join(lines)


def assessment(summary: pd.Series) -> str:
    mean_spearman = summary["spearman"]
    top20 = summary["top20_overlap_pct"]
    top50 = summary["top50_overlap_pct"]
    if mean_spearman >= 0.25 and top20 >= 20 and top50 >= 30:
        return (
            "Final assessment: ranking quality is reasonably strong for a research/downstream RL prediction layer. "
            "The current model is suitable as the prediction layer for the downstream RL system and should be frozen. "
            "It should not be described as production-ready."
        )
    weakness = "low gameweek-level Spearman rank correlation"
    if top20 < 20 or top50 < 30:
        weakness = "limited overlap between predicted and actual top player groups"
    return (
        "Final assessment: ranking quality is weak, so the current model should NOT yet be considered suitable. "
        f"The single most important weakness is {weakness}. Do not describe it as production-ready."
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    joined = load_joined()
    metrics = build_metrics(joined)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_CSV, index=False)

    overall = metrics[metrics["metric_scope"].eq("overall")].iloc[0]
    gw_mean = metrics[metrics["metric_scope"].eq("gameweek_mean")].iloc[0]
    gw_median = metrics[metrics["metric_scope"].eq("gameweek_median")].iloc[0]
    position_lines = []
    for row in metrics[metrics["metric_scope"].eq("position")].sort_values("position").itertuples(index=False):
        position_lines.append(f"{row.position}: Spearman {row.spearman:.4f} over {row.n} rows")

    no_component_note = (
        "Individual XGBoost/Random Forest prediction CSVs were not present, so component ranking metrics were not recalculated."
    )

    text = "\n".join(
        [
            "Fantasy XI Ranking Evaluation",
            "=============================",
            "",
            f"Joined prediction rows: {len(joined):,}",
            "",
            f"Overall Spearman: {overall['spearman']:.4f}",
            f"Mean GW Spearman: {gw_mean['spearman']:.4f}",
            f"Median GW Spearman: {gw_median['spearman']:.4f}",
            f"Mean Top-10 overlap: {gw_mean['top10_overlap_pct']:.2f}%",
            f"Mean Top-20 overlap: {gw_mean['top20_overlap_pct']:.2f}%",
            f"Mean Top-50 overlap: {gw_mean['top50_overlap_pct']:.2f}%",
            "",
            f"Median Top-10 overlap: {gw_median['top10_overlap_pct']:.2f}%",
            f"Median Top-20 overlap: {gw_median['top20_overlap_pct']:.2f}%",
            f"Median Top-50 overlap: {gw_median['top50_overlap_pct']:.2f}%",
            "",
            "Position Spearman:",
            *position_lines,
            "",
            no_component_note,
            representative_top10(joined),
            "",
            assessment(gw_mean),
        ]
    )
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved {OUT_CSV}")
    print(f"Saved {OUT_TXT}")


if __name__ == "__main__":
    main()
