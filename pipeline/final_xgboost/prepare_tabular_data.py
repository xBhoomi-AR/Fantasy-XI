from pathlib import Path

import numpy as np
import pandas as pd


# Reuse the processed dataset created for the LSTM
SOURCE_DIR = Path("pipeline/final/data")

# Save the XGBoost-ready data here
OUTPUT_DIR = Path("pipeline/final_xgboost/data")


def load_data():
    print("Loading processed LSTM dataset...")

    X_sequence = np.load(SOURCE_DIR / "X_sequence.npy")
    X_context = np.load(SOURCE_DIR / "X_context.npy")
    y = np.load(SOURCE_DIR / "y.npy")
    metadata = pd.read_csv(SOURCE_DIR / "metadata.csv")

    sequence_features = pd.read_csv(
        SOURCE_DIR / "sequence_features.csv"
    )["sequence_feature"].tolist()

    context_features = pd.read_csv(
        SOURCE_DIR / "context_features.csv"
    )["context_feature"].tolist()

    print(f"Sequence shape : {X_sequence.shape}")
    print(f"Context shape  : {X_context.shape}")
    print(f"Target shape   : {y.shape}")

    return (
        X_sequence,
        X_context,
        y,
        metadata,
        sequence_features,
        context_features,
    )


def create_tabular_features(
    X_sequence,
    X_context,
    sequence_features,
    context_features,
):
    print()
    print("Creating rolling features...")

    feature_blocks = []
    feature_names = []

    statistics = [
        ("mean", np.mean),
        ("std", np.std),
        ("min", np.min),
        ("max", np.max),
    ]

    for stat_name, stat_function in statistics:

        values = stat_function(X_sequence, axis=1)

        feature_blocks.append(values)

        for feature in sequence_features:
            feature_names.append(f"{feature}_{stat_name}")

    # Most recent match
    last_match = X_sequence[:, -1, :]

    feature_blocks.append(last_match)

    for feature in sequence_features:
        feature_names.append(f"{feature}_last")

    # Append target fixture context
    feature_blocks.append(X_context)

    feature_names.extend(context_features)

    X = np.concatenate(feature_blocks, axis=1).astype("float32")

    print(f"Final feature matrix : {X.shape}")
    print(f"Total features       : {len(feature_names)}")

    return X, feature_names


def split_by_season(X, y, metadata):
    train_mask = metadata["season"].isin([2122, 2223, 2324])
    val_mask = metadata["season"] == 2425
    test_mask = metadata["season"] == 2526

    train = (
        X[train_mask],
        y[train_mask],
    )

    val = (
        X[val_mask],
        y[val_mask],
    )

    test = (
        X[test_mask],
        y[test_mask],
    )

    print()
    print("Dataset split")
    print(f"Train : {train[0].shape}")
    print(f"Val   : {val[0].shape}")
    print(f"Test  : {test[0].shape}")

    return train, val, test


def save_data(train, val, test, feature_names):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "X_train.npy", train[0])
    np.save(OUTPUT_DIR / "y_train.npy", train[1])

    np.save(OUTPUT_DIR / "X_val.npy", val[0])
    np.save(OUTPUT_DIR / "y_val.npy", val[1])

    np.save(OUTPUT_DIR / "X_test.npy", test[0])
    np.save(OUTPUT_DIR / "y_test.npy", test[1])

    pd.Series(
        feature_names,
        name="feature",
    ).to_csv(
        OUTPUT_DIR / "feature_names.csv",
        index=False,
    )

    print()
    print("Saved XGBoost dataset.")
    print("Location:", OUTPUT_DIR)


def main():
    (
        X_sequence,
        X_context,
        y,
        metadata,
        sequence_features,
        context_features,
    ) = load_data()

    X, feature_names = create_tabular_features(
        X_sequence,
        X_context,
        sequence_features,
        context_features,
    )

    train, val, test = split_by_season(
        X,
        y,
        metadata,
    )

    save_data(
        train,
        val,
        test,
        feature_names,
    )


if __name__ == "__main__":
    main()