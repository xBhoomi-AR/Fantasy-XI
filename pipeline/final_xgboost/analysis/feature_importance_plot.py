from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("pipeline/final_xgboost/results")


def main():

    df = pd.read_csv(
        RESULTS_DIR / "feature_importance.csv"
    )

    top = df.head(10)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top["feature"][::-1],
        top["importance"][::-1],
    )

    plt.xlabel("Importance")
    plt.title("Top 10 Feature Importance")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "feature_importance_top10.png",
        dpi=300,
    )

    plt.close()

    print("Saved feature_importance_top10.png")


if __name__ == "__main__":
    main()