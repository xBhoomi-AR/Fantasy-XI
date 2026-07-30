from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("pipeline/final_xgboost/results")


ranges = ["0-2", "3-5", "6-9", "10+"]

baseline = [
    0.6405,
    1.4922,
    4.1218,
    8.9554,
]

weighted = [
    0.9746,
    1.4204,
    3.4937,
    8.3483,
]


x = np.arange(len(ranges))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

bars1 = ax.bar(
    x - width / 2,
    baseline,
    width,
    label="Baseline",
)

bars2 = ax.bar(
    x + width / 2,
    weighted,
    width,
    label="Weighted",
)

ax.set_title("Performance by Score Range")
ax.set_xlabel("Actual Score Range")
ax.set_ylabel("MAE")
ax.set_xticks(x)
ax.set_xticklabels(ranges)

ax.grid(axis="y", alpha=0.3)

ax.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.08,
            f"{h:.2f}",
            ha="center",
            fontsize=9,
        )

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "score_range_mae.png",
    dpi=300,
)

plt.close()

print("Saved score_range_mae.png")