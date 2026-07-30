from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HISTORY_PATH = Path("pipeline/final/results/training_history.csv")

OUTPUT_DIR = Path("pipeline/final/report/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "training_curve.png"


df = pd.read_csv(HISTORY_PATH)

best_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
best_loss = df["val_loss"].min()

plt.figure(figsize=(8, 5))

plt.plot(
    df["epoch"],
    df["train_loss"],
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    df["epoch"],
    df["val_loss"],
    linewidth=2,
    label="Validation Loss"
)

plt.scatter(
    best_epoch,
    best_loss,
    s=90,
    color="red",
    zorder=5,
    label="Best Model"
)

plt.annotate(
    f"Best Epoch ({int(best_epoch)})",
    xy=(best_epoch, best_loss),
    xytext=(best_epoch + 2, best_loss + 0.08),
    arrowprops=dict(arrowstyle="->")
)

plt.title("LSTM Training Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved to {OUTPUT_PATH}")