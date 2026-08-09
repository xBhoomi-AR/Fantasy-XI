import numpy as np
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "player_features.csv"
OUTPUT_DIR = BASE_DIR / "artifacts" / "experiments"

SEQUENCE_LENGTH = 5


def load_data():
    print("Loading feature data...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Input shape: {df.shape}")

    return df


def prepare_features(df):

    columns_to_skip = {
        "player_id",
        "season",
        "gameweek",
        "fixture_id",
        "team_id",
        "opponent_team_id",
        "total_points",
        "target_points"
    }

    feature_df = df.drop(
        columns=[
            column
            for column in columns_to_skip
            if column in df.columns
        ]
    ).copy()

    # Convert boolean columns to numbers.
    for column in feature_df.columns:
        if feature_df[column].dtype == bool:
            feature_df[column] = feature_df[column].astype(int)

    # Convert categorical features such as position
    # into one-hot encoded columns.
    categorical_columns = feature_df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    if categorical_columns:
        print(
            f"Encoding categorical features: {categorical_columns}"
        )

        feature_df = pd.get_dummies(
            feature_df,
            columns=categorical_columns,
            dtype=float
        )

    # Make sure every feature is numeric.
    feature_df = feature_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    feature_df = feature_df.fillna(0)

    feature_columns = feature_df.columns.tolist()

    print(
        f"Number of numeric input features: {len(feature_columns)}"
    )

    return feature_df, feature_columns


def create_sequences(df, feature_df, feature_columns):

    print("Creating player sequences...")

    X = []
    y = []

    metadata = []

    # Create sequences separately for every player and season.
    for (player_id, season), player_data in df.groupby(
        ["player_id", "season"]
    ):

        player_data = player_data.sort_values(
            "gameweek"
        )

        # Five previous Gameweeks are used
        # to predict the following Gameweek.
        for i in range(
            len(player_data) - SEQUENCE_LENGTH
        ):

            history = player_data.iloc[
                i:i + SEQUENCE_LENGTH
            ]

            target_row = player_data.iloc[
                i + SEQUENCE_LENGTH
            ]

            gameweeks = history["gameweek"].tolist()

            # Make sure the five history Gameweeks
            # are consecutive.
            if gameweeks != list(
                range(
                    gameweeks[0],
                    gameweeks[0] + SEQUENCE_LENGTH
                )
            ):
                continue

            # Make sure the target is the
            # immediately following Gameweek.
            if target_row["gameweek"] != gameweeks[-1] + 1:
                continue

            history_indices = history.index

            X.append(
                feature_df.loc[
                    history_indices,
                    feature_columns
                ].values
            )

            # Target = actual FPL points in
            # the next Gameweek.
            y.append(
                target_row["total_points"]
            )

            # Store information about the
            # Gameweek being predicted.
            metadata.append(
                {
                    "player_id": target_row["player_id"],
                    "target_season": target_row["season"],
                    "target_gameweek": target_row["gameweek"]
                }
            )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    metadata = pd.DataFrame(metadata)

    return X, y, metadata


def save_sequences(
    X,
    y,
    metadata,
    feature_columns
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_DIR / "X_sequences.npy",
        X
    )

    np.save(
        OUTPUT_DIR / "y_targets.npy",
        y
    )

    # Save metadata separately so that
    # train/validation/test splitting can
    # be performed chronologically later.
    np.save(
        OUTPUT_DIR / "metadata.npy",
        metadata.to_numpy()
    )

    metadata.to_csv(
        OUTPUT_DIR / "sequence_metadata.csv",
        index=False
    )

    pd.Series(feature_columns).to_csv(
        OUTPUT_DIR / "feature_columns.txt",
        index=False,
        header=False
    )

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Metadata shape: {metadata.shape}")

    print(
        f"Saved sequences to: {OUTPUT_DIR}"
    )


def main():

    df = load_data()

    feature_df, feature_columns = prepare_features(
        df
    )

    X, y, metadata = create_sequences(
        df,
        feature_df,
        feature_columns
    )

    save_sequences(
        X,
        y,
        metadata,
        feature_columns
    )

    print(
        "Sequence creation completed successfully."
    )


if __name__ == "__main__":
    main()