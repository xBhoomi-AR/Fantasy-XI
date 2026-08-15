from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Project root
ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "xgboost_model" / "reports" / "evaluation_2025_26.csv"
OUTPUT_FILE = ROOT / "xgboost_model" / "reports" / "xgboost_range_mae_rmse.png"


# Load evaluation results
df = pd.read_csv(INPUT_FILE)

# Keep only point-range rows
ranges = ["0-2", "3-5", "6-9", "10+"]

df = df[df["slice"].isin(ranges)].copy()

# Keep correct order
df["slice"] = pd.Categorical(
    df["slice"],
    categories=ranges,
    ordered=True
)

df = df.sort_values("slice")


# Create graph
x = range(len(df))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

mae_bars = ax.bar(
    [i - width / 2 for i in x],
    df["mae"],
    width,
    label="MAE"
)

rmse_bars = ax.bar(
    [i + width / 2 for i in x],
    df["rmse"],
    width,
    label="RMSE"
)


# Title and labels
ax.set_title(
    "XGBoost FPL Points Prediction: MAE and RMSE by Points Range",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Actual FPL Points Range")
ax.set_ylabel("Prediction Error")

ax.set_xticks(list(x))
ax.set_xticklabels(ranges)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)


# Add numerical values above bars
for bar in mae_bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.08,
        f"{height:.2f}",
        ha="center",
        va="bottom"
    )

for bar in rmse_bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.08,
        f"{height:.2f}",
        ha="center",
        va="bottom"
    )


plt.tight_layout()

# Save high-resolution image
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nGraph saved to:")
print(OUTPUT_FILE)