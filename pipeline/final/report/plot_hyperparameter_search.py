from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "results" / "lstm_experiments.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "hyperparameter_search.png"


df = pd.read_csv(CSV_PATH)

best_idx = df["val_loss"].idxmin()

colors = ["steelblue"] * len(df)
colors[best_idx] = "crimson"

plt.figure(figsize=(10, 5))

bars = plt.bar(
    df["experiment"].astype(str),
    df["val_loss"],
    color=colors,
    edgecolor="black"
)

best_bar = bars[best_idx]

plt.text(
    best_bar.get_x() + best_bar.get_width() / 2,
    best_bar.get_height() + 0.01,
    "Best",
    ha="center",
    fontsize=10,
    fontweight="bold",
    color="crimson"
)

plt.title("LSTM Hyperparameter Search")
plt.xlabel("Experiment")
plt.ylabel("Validation Loss")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", OUTPUT_PATH)