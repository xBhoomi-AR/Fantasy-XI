from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "predictions" / "test_2025_26_predictions.csv"
FEATURES = ROOT / "data" / "processed" / "model_features.csv"
PLAYERS = ROOT / "data" / "raw" / "players.csv"
OUT_CSV = ROOT / "reports" / "prediction_range_audit.csv"
OUT_TXT = ROOT / "reports" / "prediction_range_audit.txt"


def load_joined() -> pd.DataFrame:
    preds = pd.read_csv(PREDICTIONS)
    keys = ["player_id", "fixture_id", "season", "gameweek", "team_id", "opponent_team_id", "position"]
    actual = pd.read_csv(FEATURES, usecols=keys + ["target_points"], engine="python")
    actual = actual[actual["season"].astype(str).eq("2025-26")].drop_duplicates(keys, keep="last")
    joined = preds.merge(actual, on=keys, how="inner", validate="one_to_one")
    if PLAYERS.exists():
        names = pd.read_csv(PLAYERS, usecols=["player_id", "web_name", "player_name"]).drop_duplicates("player_id", keep="last")
        joined = joined.merge(names, on="player_id", how="left")
    return joined


def distribution_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col, label in [("target_points", "actual"), ("predicted_points", "predicted")]:
        s = df[col]
        rows.append(
            {
                "section": "distribution",
                "metric": label,
                "n": len(s),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "p25": s.quantile(0.25),
                "p75": s.quantile(0.75),
                "p90": s.quantile(0.90),
                "p95": s.quantile(0.95),
                "p99": s.quantile(0.99),
                "min": s.min(),
                "max": s.max(),
            }
        )
    return rows


def range_rows(df: pd.DataFrame) -> list[dict]:
    ranges = [("0-2", -np.inf, 2), ("3-5", 3, 5), ("6-9", 6, 9), ("10+", 10, np.inf)]
    rows = []
    for name, lo, hi in ranges:
        g = df[(df["target_points"] >= lo) & (df["target_points"] <= hi)]
        rows.append(
            {
                "section": "actual_target_range",
                "metric": name,
                "n": len(g),
                "actual_mean": g["target_points"].mean(),
                "predicted_mean": g["predicted_points"].mean(),
                "actual_min": g["target_points"].min(),
                "actual_max": g["target_points"].max(),
                "predicted_min": g["predicted_points"].min(),
                "predicted_max": g["predicted_points"].max(),
                "mae": (g["target_points"] - g["predicted_points"]).abs().mean(),
            }
        )
    return rows


def high_scorer_recovery_rows(df: pd.DataFrame) -> list[dict]:
    high = df[df["target_points"] >= 10].copy()
    return [
        {
            "section": "high_scorer_recovery",
            "metric": "actual_10_plus",
            "n": len(high),
            "predicted_mean": high["predicted_points"].mean(),
            "predicted_median": high["predicted_points"].median(),
            "predicted_max": high["predicted_points"].max(),
            "pct_pred_ge_8": (high["predicted_points"] >= 8).mean() * 100,
            "pct_pred_ge_10": (high["predicted_points"] >= 10).mean() * 100,
            "pct_pred_ge_12": (high["predicted_points"] >= 12).mean() * 100,
        }
    ]


def top_player_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for gw, group in df.groupby("gameweek", sort=True):
        top = group.sort_values(["target_points", "predicted_points"], ascending=[False, False]).head(10)
        rows.append(
            {
                "section": "actual_top10_by_gameweek",
                "metric": f"GW{int(gw)}",
                "gameweek": int(gw),
                "n": len(top),
                "actual_top10_avg_predicted": top["predicted_points"].mean(),
                "actual_top10_pred_ge_8_count": int((top["predicted_points"] >= 8).sum()),
                "actual_top10_pred_ge_8_pct": (top["predicted_points"] >= 8).mean() * 100,
                "actual_top10_mean_actual": top["target_points"].mean(),
            }
        )
    return rows


def decile_rows(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    out["prediction_decile"] = pd.qcut(out["predicted_points"], q=10, labels=False, duplicates="drop") + 1
    rows = []
    for decile, group in out.groupby("prediction_decile", sort=True):
        rows.append(
            {
                "section": "prediction_decile",
                "metric": f"decile_{int(decile)}",
                "decile": int(decile),
                "n": len(group),
                "predicted_mean": group["predicted_points"].mean(),
                "actual_mean": group["target_points"].mean(),
                "predicted_min": group["predicted_points"].min(),
                "predicted_max": group["predicted_points"].max(),
            }
        )
    return rows


def representative_actual_top10(df: pd.DataFrame) -> str:
    lines = []
    available = sorted(df["gameweek"].dropna().astype(int).unique().tolist())
    selected = []
    for gw in [1, 10, 20, 30, max(available)]:
        if gw in available and gw not in selected:
            selected.append(gw)
    for gw in selected:
        top = df[df["gameweek"].eq(gw)].sort_values(["target_points", "predicted_points"], ascending=[False, False]).head(10)
        lines.append(f"\nActual GW{gw} top 10 and predictions:")
        lines.append("actual_rank, player_id, player, position, actual_points, predicted_points")
        for rank, row in enumerate(top.itertuples(index=False), start=1):
            name = getattr(row, "web_name", "") or getattr(row, "player_name", "") or ""
            lines.append(f"{rank}, {row.player_id}, {name}, {row.position}, {row.target_points:.1f}, {row.predicted_points:.3f}")
    return "\n".join(lines)


def verdict(df: pd.DataFrame, deciles: pd.DataFrame, high_row: pd.Series) -> tuple[str, str]:
    actual_std = df["target_points"].std()
    pred_std = df["predicted_points"].std()
    compression_ratio = pred_std / actual_std if actual_std else np.nan
    decile_gain = deciles.sort_values("decile")["actual_mean"].iloc[-1] - deciles.sort_values("decile")["actual_mean"].iloc[0]
    high_pred_mean = high_row["predicted_mean"]
    pct_ge_8 = high_row["pct_pred_ge_8"]

    compressed = compression_ratio < 0.75
    meaningful = decile_gain >= 2.0 and high_pred_mean >= 6.0 and pct_ge_8 >= 30
    severe = compression_ratio < 0.45 or decile_gain < 1.0 or pct_ge_8 < 20

    explanation = (
        f"Prediction std / actual std = {compression_ratio:.3f}; "
        f"top decile actual mean minus bottom decile actual mean = {decile_gain:.3f}; "
        f"actual 10+ players have mean prediction {high_pred_mean:.3f}, with {pct_ge_8:.2f}% predicted >= 8."
    )
    if meaningful and not compressed:
        return "SAFE TO FREEZE", explanation
    if severe or not meaningful:
        return "NOT SUITABLE", explanation
    return "USABLE WITH LIMITATIONS", explanation


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = load_joined()
    rows = []
    rows.append(
        {
            "section": "raw_range",
            "metric": "range",
            "actual_min": df["target_points"].min(),
            "actual_max": df["target_points"].max(),
            "predicted_min": df["predicted_points"].min(),
            "predicted_max": df["predicted_points"].max(),
            "n": len(df),
        }
    )
    rows.extend(distribution_rows(df))
    rows.extend(range_rows(df))
    rows.extend(high_scorer_recovery_rows(df))
    rows.extend(top_player_rows(df))
    rows.extend(decile_rows(df))
    audit = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT_CSV, index=False)

    dist = audit[audit["section"].eq("distribution")].set_index("metric")
    ranges = audit[audit["section"].eq("actual_target_range")]
    high = audit[audit["section"].eq("high_scorer_recovery")].iloc[0]
    top10 = audit[audit["section"].eq("actual_top10_by_gameweek")]
    deciles = audit[audit["section"].eq("prediction_decile")].copy()
    final_verdict, verdict_explanation = verdict(df, deciles, high)

    actual_std = dist.loc["actual", "std"]
    predicted_std = dist.loc["predicted", "std"]
    compression_ratio = predicted_std / actual_std
    report = "\n".join(
        [
            "Prediction Range + Calibration Audit",
            "====================================",
            "",
            f"Joined rows: {len(df):,}",
            "Note: the current prediction pipeline clips predictions to the range 0-20, so the predicted maximum is bounded by that clipping rule.",
            "",
            "Raw Range:",
            f"Actual target_points min/max: {df['target_points'].min():.3f} / {df['target_points'].max():.3f}",
            f"Predicted_points min/max: {df['predicted_points'].min():.3f} / {df['predicted_points'].max():.3f}",
            "",
            "Distribution:",
            dist[["mean", "median", "std", "p25", "p75", "p90", "p95", "p99", "min", "max"]].to_string(float_format=lambda x: f"{x:.3f}"),
            "",
            "High-Score Compression by Actual Range:",
            ranges[["metric", "n", "actual_mean", "predicted_mean", "actual_min", "actual_max", "predicted_min", "predicted_max", "mae"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"),
            "",
            "High-Scorer Recovery for Actual 10+ Players:",
            f"N: {int(high['n'])}",
            f"Mean predicted_points: {high['predicted_mean']:.3f}",
            f"Median predicted_points: {high['predicted_median']:.3f}",
            f"Maximum predicted_points: {high['predicted_max']:.3f}",
            f"Predicted >= 8: {high['pct_pred_ge_8']:.2f}%",
            f"Predicted >= 10: {high['pct_pred_ge_10']:.2f}%",
            f"Predicted >= 12: {high['pct_pred_ge_12']:.2f}%",
            "",
            "Actual Top-10 Players by Gameweek:",
            f"Mean predicted_points of actual top 10: {top10['actual_top10_avg_predicted'].mean():.3f}",
            f"Median predicted_points of actual top 10: {top10['actual_top10_avg_predicted'].median():.3f}",
            f"Mean count with prediction >= 8: {top10['actual_top10_pred_ge_8_count'].mean():.2f} / 10",
            "",
            "Prediction Deciles:",
            deciles[["metric", "n", "predicted_mean", "actual_mean", "predicted_min", "predicted_max"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"),
            representative_actual_top10(df),
            "",
            "Calibration / Compression Check:",
            f"Are predictions heavily compressed compared with actual points? {'Yes' if compression_ratio < 0.75 else 'No'}; predicted std is {compression_ratio:.3f}x actual std.",
            f"Can the model meaningfully distinguish high-scoring players? {'Yes, with limitations' if final_verdict != 'NOT SUITABLE' else 'No, not reliably enough'}; {verdict_explanation}",
            f"Is compression severe enough to make predictions unsuitable for downstream RL ranking input? {'Yes' if final_verdict == 'NOT SUITABLE' else 'No, but calibration limitations should be understood.'}",
            "",
            f"FINAL VERDICT: {final_verdict}",
        ]
    )
    OUT_TXT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved {OUT_CSV}")
    print(f"Saved {OUT_TXT}")


if __name__ == "__main__":
    main()
