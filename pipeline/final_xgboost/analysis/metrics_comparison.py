from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("pipeline/final_xgboost/results")


metrics = ["MAE", "RMSE", "Correlation"]

baseline = [1.0351, 1.9787, 0.5500]
weighted = [1.2743, 2.1265, 0.5261]


x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

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

ax.set_title("Baseline vs Weighted XGBoost")
ax.set_ylabel("Metric Value")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

ax.grid(axis="y", alpha=0.3)


for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.03,
            f"{h:.3f}",
            ha="center",
            fontsize=9,
        )

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "overall_metrics.png",
    dpi=300,
)

plt.close()

print("Saved overall_metrics.png")