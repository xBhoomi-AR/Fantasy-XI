import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Load targets
# ----------------------------
data_dir = Path("pipeline/final/data")
y = np.load(data_dir / "y.npy")

# ----------------------------
# Create bins
# ----------------------------
labels = ["0-2", "3-5", "6-9", "10+"]
counts = [
    np.sum((y >= 0) & (y <= 2)),
    np.sum((y >= 3) & (y <= 5)),
    np.sum((y >= 6) & (y <= 9)),
    np.sum(y >= 10),
]

total = len(y)
percentages = [c / total * 100 for c in counts]

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(8,5))

bars = plt.bar(
    labels,
    counts,
    edgecolor="black",
    linewidth=1.2
)

plt.title("Fantasy Premier League Points Distribution", fontsize=15, weight="bold")
plt.xlabel("Player Points")
plt.ylabel("Number of Samples")

# Write percentage on top of each bar
for bar, pct in zip(bars, percentages):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        weight="bold"
    )

plt.tight_layout()

output = Path("pipeline/final/report/points_distribution.png")
plt.savefig(output, dpi=300)

plt.show()

print("\n========== Distribution ==========")
for l, c, p in zip(labels, counts, percentages):
    print(f"{l:>4} : {c:6d} ({p:.2f}%)")