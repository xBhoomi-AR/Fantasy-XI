from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path("pipeline/final/results/final_test_predictions.csv")

OUTPUT_DIR = Path("pipeline/final/report/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "actual_vs_predicted.png"


# ------------------------
# Metrics from evaluation
# ------------------------
MAE = 0.959
RMSE = 1.970
CORR = 0.560


df = pd.read_csv(CSV_PATH)

plt.figure(figsize=(8, 7))

plt.scatter(
    df["actual_points"],
    df["predicted_points"],
    alpha=0.25,
    s=10
)

max_val = max(
    df["actual_points"].max(),
    df["predicted_points"].max()
)

plt.plot(
    [0, max_val],
    [0, max_val],
    "r--",
    linewidth=2,
    label="Perfect Prediction"
)

# -------- Metrics Box --------
metrics_text = (
    f"MAE   : {MAE:.3f}\n"
    f"RMSE : {RMSE:.3f}\n"
    f"Corr  : {CORR:.3f}"
)

plt.text(
    0.97,
    0.05,
    metrics_text,
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="black",
        alpha=0.95,
    ),
)

plt.title("Actual vs Predicted Fantasy Points")
plt.xlabel("Actual Points")
plt.ylabel("Predicted Points")

plt.grid(alpha=0.3)
plt.legend(loc="upper left")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved to {OUTPUT_PATH}")