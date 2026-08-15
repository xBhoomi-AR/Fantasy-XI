from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

FEATURES_FILE = (
    BASE_DIR / "data" / "processed" / "model_features.csv"
)

PREDICTIONS_FILE = (
    BASE_DIR / "predictions" / "test_2025_26_predictions.csv"
)

COMPONENTS_FILE = (
    BASE_DIR / "reports" / "evaluation_components_2025_26.csv"
)

FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_and_merge_predictions():
    """Combine actual target points with the final model predictions."""

    features = pd.read_csv(
        FEATURES_FILE,
        engine="python"
    )

    predictions = pd.read_csv(
        PREDICTIONS_FILE,
        engine="python"
    )

    # Keep only the columns needed to match the prediction rows.
    actual = features[
        [
            "player_id",
            "season",
            "gameweek",
            "position",
            "target_points",
        ]
    ].copy()

    predictions = predictions[
        [
            "player_id",
            "season",
            "gameweek",
            "position",
            "predicted_points",
        ]
    ].copy()

    # Keep only the final 2025-26 test season.
    actual = actual[actual["season"] == "2025-26"].copy()
    predictions = predictions[
        predictions["season"] == "2025-26"
    ].copy()

    # Remove possible duplicate rows before merging.
    actual = actual.drop_duplicates(
        subset=["player_id", "season", "gameweek", "position"]
    )

    predictions = predictions.drop_duplicates(
        subset=["player_id", "season", "gameweek", "position"]
    )

    merged = predictions.merge(
        actual,
        on=[
            "player_id",
            "season",
            "gameweek",
            "position",
        ],
        how="inner",
        validate="one_to_one",
    )

    merged["target_points"] = pd.to_numeric(
        merged["target_points"],
        errors="coerce",
    )

    merged["predicted_points"] = pd.to_numeric(
        merged["predicted_points"],
        errors="coerce",
    )

    merged = merged.dropna(
        subset=["target_points", "predicted_points"]
    )

    print(f"Matched rows: {len(merged):,}")

    if len(merged) == 0:
        raise ValueError(
            "No rows matched between model_features.csv "
            "and test_2025_26_predictions.csv."
        )

    return merged


def plot_actual_vs_predicted(df):
    """Create the main actual-vs-predicted scatter plot."""

    plt.figure(figsize=(10, 7))

    plt.scatter(
        df["target_points"],
        df["predicted_points"],
        alpha=0.18,
        s=14,
    )

    # Perfect prediction reference line.
    minimum = min(
        df["target_points"].min(),
        df["predicted_points"].min(),
    )

    maximum = max(
        df["target_points"].max(),
        df["predicted_points"].max(),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=2,
        label="Perfect prediction",
    )

    plt.xlabel("Actual FPL Points")
    plt.ylabel("Predicted FPL Points")
    plt.title("Actual vs Predicted FPL Points")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output = FIGURES_DIR / "actual_vs_predicted.png"

    plt.savefig(
        output,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output}")


def plot_spearman_by_position(df):
    """Calculate and plot Spearman correlation by position."""

    positions = ["GK", "DEF", "MID", "FWD"]

    rows = []

    for position in positions:
        subset = df[df["position"] == position]

        correlation = subset["target_points"].corr(
            subset["predicted_points"],
            method="spearman",
        )

        rows.append(
            {
                "position": position,
                "spearman": correlation,
            }
        )

    overall = df["target_points"].corr(
        df["predicted_points"],
        method="spearman",
    )

    rows.append(
        {
            "position": "Overall",
            "spearman": overall,
        }
    )

    result = pd.DataFrame(rows)

    print("\nSpearman correlation:")
    print(result.to_string(index=False))

    plt.figure(figsize=(9, 6))

    plt.bar(
        result["position"],
        result["spearman"],
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.ylim(0, 1)

    plt.xlabel("Position")
    plt.ylabel("Spearman correlation")
    plt.title("Player Ranking Correlation by Position")

    for index, value in enumerate(result["spearman"]):
        plt.text(
            index,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
        )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    output = FIGURES_DIR / "spearman_by_position.png"

    plt.savefig(
        output,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output}")


def plot_actual_vs_predicted_ranges(df):
    """Compare actual and predicted means across scoring ranges."""

    bins = [-float("inf"), 2, 5, 9, float("inf")]
    labels = ["0–2", "3–5", "6–9", "10+"]

    df = df.copy()

    df["actual_range"] = pd.cut(
        df["target_points"],
        bins=bins,
        labels=labels,
    )

    grouped = (
        df.groupby(
            "actual_range",
            observed=False,
        )
        .agg(
            actual_mean=("target_points", "mean"),
            predicted_mean=("predicted_points", "mean"),
        )
        .reindex(labels)
    )

    print("\nActual vs predicted mean by scoring range:")
    print(grouped.to_string())

    x = range(len(grouped))
    width = 0.36

    plt.figure(figsize=(10, 6))

    plt.bar(
        [i - width / 2 for i in x],
        grouped["actual_mean"],
        width=width,
        label="Actual mean",
    )

    plt.bar(
        [i + width / 2 for i in x],
        grouped["predicted_mean"],
        width=width,
        label="Predicted mean",
    )

    plt.xticks(
        list(x),
        labels,
    )

    plt.xlabel("Actual point range")
    plt.ylabel("Mean FPL Points")
    plt.title("Actual vs Predicted Mean Points by Scoring Range")
    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    output = (
        FIGURES_DIR
        / "actual_vs_predicted_by_range.png"
    )

    plt.savefig(
        output,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output}")


def main():
    print("Creating final XGBoost evaluation plots...\n")

    df = load_and_merge_predictions()

    print(
        f"Actual point range: "
        f"{df['target_points'].min():.0f} "
        f"to {df['target_points'].max():.0f}"
    )

    print(
        f"Predicted point range: "
        f"{df['predicted_points'].min():.3f} "
        f"to {df['predicted_points'].max():.3f}"
    )

    plot_actual_vs_predicted(df)
    plot_spearman_by_position(df)
    plot_actual_vs_predicted_ranges(df)

    print("\nAll final evaluation plots created successfully.")


if __name__ == "__main__":
    main()