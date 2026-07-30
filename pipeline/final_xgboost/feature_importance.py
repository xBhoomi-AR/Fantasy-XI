from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("pipeline/final_xgboost/models/xgboost_model.pkl")
FEATURE_PATH = Path("pipeline/final_xgboost/data/feature_names.csv")
OUTPUT_DIR = Path("pipeline/final_xgboost/results")


def main():

    print("Loading model...")

    model = joblib.load(MODEL_PATH)

    feature_names = pd.read_csv(FEATURE_PATH)["feature"]

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        by="importance",
        ascending=False,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    print()
    print("Top 20 Features")
    print("----------------------------")
    print(importance.head(20))

    print()
    print("Saved to:")
    print(OUTPUT_DIR / "feature_importance.csv")


if __name__ == "__main__":
    main()