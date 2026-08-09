import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = BASE_DIR / "artifacts" / "experiments"

X_FILE = ARTIFACT_DIR / "X_sequences.npy"
Y_FILE = ARTIFACT_DIR / "y_targets.npy"
METADATA_FILE = ARTIFACT_DIR / "metadata.npy"


# Chronological Gameweek split
TRAIN_END = 29
VALIDATION_START = 30
VALIDATION_END = 34
TEST_START = 35


def load_data():
    print("Loading sequence data...")

    X = np.load(X_FILE)
    y = np.load(Y_FILE)
    metadata = np.load(
        METADATA_FILE,
        allow_pickle=True
    )

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Metadata shape: {metadata.shape}")

    return X, y, metadata


def create_split(X, y, metadata):
    print()
    print("Creating chronological Gameweek split...")

    # Metadata columns:
    # 0 = player_id
    # 1 = target_season
    # 2 = target_gameweek

    target_gameweeks = metadata[:, 2].astype(int)

    train_mask = target_gameweeks <= TRAIN_END

    validation_mask = (
        (target_gameweeks >= VALIDATION_START)
        & (target_gameweeks <= VALIDATION_END)
    )

    test_mask = target_gameweeks >= TEST_START

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_validation = X[validation_mask]
    y_validation = y[validation_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print(
    f"Training Gameweeks: "
    f"6 - {TRAIN_END}"
    )

    print(
        f"Validation Gameweeks: "
        f"{VALIDATION_START} - {VALIDATION_END}"
    )

    print(
        f"Test Gameweeks: "
        f"{TEST_START} - {target_gameweeks.max()}"
    )

    print()
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_validation)}")
    print(f"Test samples: {len(X_test)}")

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test
    )


def save_split(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test
):
    print()
    print("Saving split datasets...")

    np.save(
        ARTIFACT_DIR / "X_train.npy",
        X_train
    )

    np.save(
        ARTIFACT_DIR / "y_train.npy",
        y_train
    )

    np.save(
        ARTIFACT_DIR / "X_validation.npy",
        X_validation
    )

    np.save(
        ARTIFACT_DIR / "y_validation.npy",
        y_validation
    )

    np.save(
        ARTIFACT_DIR / "X_test.npy",
        X_test
    )

    np.save(
        ARTIFACT_DIR / "y_test.npy",
        y_test
    )

    print("Split files saved successfully.")


def main():
    X, y, metadata = load_data()

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test
    ) = create_split(
        X,
        y,
        metadata
    )

    save_split(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test
    )

    print()
    print(
        "Train/validation/test split "
        "completed successfully."
    )


if __name__ == "__main__":
    main()