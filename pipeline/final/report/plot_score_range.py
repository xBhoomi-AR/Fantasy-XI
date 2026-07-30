from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path("pipeline/final/results/performance_by_points.csv")

OUTPUT_DIR = Path("pipeline/final/report/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "score_range_mae.png"


df = pd.read_csv(CSV_PATH)

# Remove the negative points row
df = df[df["Group"] != "Negative"]

plt.figure(figsize=(7,5))

plt.bar(
    df["Group"],
    df["MAE"],
    edgecolor="black"
)

for i, value in enumerate(df["MAE"]):
    plt.text(
        i,
        value + 0.1,
        f"{value:.2f}",
        ha="center",
        fontsize=10
    )

plt.title("Prediction Error by Actual Score Range")
plt.xlabel("Actual Score Range")
plt.ylabel("MAE")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", OUTPUT_PATH)