from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("pipeline/final/data")
OUTPUT_DIR = DATA_DIR / "splits"


def load_data():
    print("Loading final dataset...")

    X_sequence = np.load(DATA_DIR / "X_sequence.npy")
    X_context = np.load(DATA_DIR / "X_context.npy")
    y = np.load(DATA_DIR / "y.npy")
    metadata = pd.read_csv(DATA_DIR / "metadata.csv")

    print("Sequence:", X_sequence.shape)
    print("Context: ", X_context.shape)
    print("Target:  ", y.shape)

    return X_sequence, X_context, y, metadata


def split_by_season(X_sequence, X_context, y, metadata):
    # Keep the newest season completely unseen for final testing
    train_mask = metadata["season"].isin([2122, 2223, 2324])
    val_mask = metadata["season"] == 2425
    test_mask = metadata["season"] == 2526

    train = (
        X_sequence[train_mask],
        X_context[train_mask],
        y[train_mask],
    )

    val = (
        X_sequence[val_mask],
        X_context[val_mask],
        y[val_mask],
    )

    test = (
        X_sequence[test_mask],
        X_context[test_mask],
        y[test_mask],
    )

    print()
    print("Season split")
    print("Train:", train[0].shape, train[1].shape, train[2].shape)
    print("Val:  ", val[0].shape, val[1].shape, val[2].shape)
    print("Test: ", test[0].shape, test[1].shape, test[2].shape)

    return train, val, test


def scale_data(train, val, test):
    X_seq_train, X_ctx_train, y_train = train
    X_seq_val, X_ctx_val, y_val = val
    X_seq_test, X_ctx_test, y_test = test

    sequence_scaler = StandardScaler()
    context_scaler = StandardScaler()

    # Fit only on training seasons.
    # Flatten temporarily because StandardScaler expects 2D data.
    train_shape = X_seq_train.shape

    seq_train_flat = X_seq_train.reshape(-1, train_shape[-1])

    sequence_scaler.fit(seq_train_flat)
    context_scaler.fit(X_ctx_train)

    def scale_sequence(X):
        original_shape = X.shape

        X = X.reshape(-1, original_shape[-1])
        X = sequence_scaler.transform(X)

        return X.reshape(original_shape).astype("float32")

    X_seq_train = scale_sequence(X_seq_train)
    X_seq_val = scale_sequence(X_seq_val)
    X_seq_test = scale_sequence(X_seq_test)

    X_ctx_train = context_scaler.transform(
        X_ctx_train
    ).astype("float32")

    X_ctx_val = context_scaler.transform(
        X_ctx_val
    ).astype("float32")

    X_ctx_test = context_scaler.transform(
        X_ctx_test
    ).astype("float32")

    print()
    print("Scaling finished")
    print(
        "Sequence train mean:",
        round(float(X_seq_train.mean()), 4)
    )
    print(
        "Sequence train std:",
        round(float(X_seq_train.std()), 4)
    )
    print(
        "Context train mean:",
        round(float(X_ctx_train.mean()), 4)
    )
    print(
        "Context train std:",
        round(float(X_ctx_train.std()), 4)
    )

    return (
        X_seq_train,
        X_ctx_train,
        y_train,
        X_seq_val,
        X_ctx_val,
        y_val,
        X_seq_test,
        X_ctx_test,
        y_test,
        sequence_scaler,
        context_scaler,
    )


def save_data(data):
    (
        X_seq_train,
        X_ctx_train,
        y_train,
        X_seq_val,
        X_ctx_val,
        y_val,
        X_seq_test,
        X_ctx_test,
        y_test,
        sequence_scaler,
        context_scaler,
    ) = data

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    arrays = {
        "X_seq_train.npy": X_seq_train,
        "X_ctx_train.npy": X_ctx_train,
        "y_train.npy": y_train,

        "X_seq_val.npy": X_seq_val,
        "X_ctx_val.npy": X_ctx_val,
        "y_val.npy": y_val,

        "X_seq_test.npy": X_seq_test,
        "X_ctx_test.npy": X_ctx_test,
        "y_test.npy": y_test,
    }

    for filename, array in arrays.items():
        np.save(OUTPUT_DIR / filename, array)

    joblib.dump(
        sequence_scaler,
        OUTPUT_DIR / "sequence_scaler.pkl"
    )

    joblib.dump(
        context_scaler,
        OUTPUT_DIR / "context_scaler.pkl"
    )

    print()
    print("Saved final train/validation/test data.")
    print("Location:", OUTPUT_DIR)


def main():
    X_sequence, X_context, y, metadata = load_data()

    train, val, test = split_by_season(
        X_sequence,
        X_context,
        y,
        metadata
    )

    data = scale_data(train, val, test)

    save_data(data)


if __name__ == "__main__":
    main()