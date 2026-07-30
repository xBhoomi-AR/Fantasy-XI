from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("pipeline/final_xgboost/results")


MAE = 1.0351
RMSE = 1.9787
CORR = 0.5500


def main():

    df = pd.read_csv(
        RESULTS_DIR / "predictions.csv"
    )

    actual = df["actual_points"]
    predicted = df["predicted_points"]

    plt.figure(figsize=(8, 8))

    plt.scatter(
        actual,
        predicted,
        alpha=0.35,
        s=18,
    )

    max_value = max(
        actual.max(),
        predicted.max(),
    )

    plt.plot(
        [0, max_value],
        [0, max_value],
        color="black",
        linewidth=2,
    )

    plt.xlabel("Actual FPL Points")
    plt.ylabel("Predicted FPL Points")

    plt.title("XGBoost: Actual vs Predicted")

    plt.text(
        0.98,
        0.02,
        f"MAE = {MAE:.3f}\nRMSE = {RMSE:.3f}\nCorr = {CORR:.3f}",
        transform=plt.gca().transAxes,
        ha="right",
        va="bottom",
        bbox=dict(
            facecolor="white",
            alpha=0.85,
        ),
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "actual_vs_predicted.png",
        dpi=300,
    )

    plt.close()

    print("Saved actual_vs_predicted.png")


if __name__ == "__main__":
    main()