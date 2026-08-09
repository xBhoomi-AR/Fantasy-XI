import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "player_gameweek.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "player_features.csv"


def load_data():
    print("Loading processed Gameweek data...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Input shape: {df.shape}")

    return df


def add_basic_features(df):
    print("Adding basic features...")

    # Home advantage can affect player performance.
    df["was_home"] = df["was_home"].fillna(0)

    # Convert the home/away value into a simple numeric feature.
    df["was_home"] = df["was_home"].astype(float)

    # A player who regularly plays many minutes is generally more
    # reliable than a player who gets only a few minutes.
    df["minutes_per_90"] = df["minutes"] / 90.0

    return df


def add_lag_features(df):
    print("Adding previous Gameweek features...")

    player_group = df.groupby(
        ["player_id", "season"],
        group_keys=False
    )

    # Previous Gameweek points.
    df["points_lag_1"] = player_group["total_points"].shift(1)

    # Previous Gameweek attacking output.
    df["goals_lag_1"] = player_group["goals_scored"].shift(1)
    df["assists_lag_1"] = player_group["assists"].shift(1)

    # Previous Gameweek expected output.
    df["xg_lag_1"] = player_group["xg"].shift(1)
    df["xa_lag_1"] = player_group["xa"].shift(1)

    # Previous Gameweek playing time.
    df["minutes_lag_1"] = player_group["minutes"].shift(1)

    return df


def add_rolling_features(df):
    print("Adding rolling form features...")

    player_group = df.groupby(
        ["player_id", "season"],
        group_keys=False
    )

    # Shift first so the current Gameweek is never included.
    previous_points = player_group["total_points"].shift(1)
    previous_xg = player_group["xg"].shift(1)
    previous_xa = player_group["xa"].shift(1)
    previous_minutes = player_group["minutes"].shift(1)
    previous_bps = player_group["bps"].shift(1)

    # Short-term form.
    df["points_form_3"] = (
        previous_points
        .groupby([df["player_id"], df["season"]])
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # Slightly longer form.
    df["points_form_5"] = (
        previous_points
        .groupby([df["player_id"], df["season"]])
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    # Expected-goal form.
    df["xg_form_3"] = (
        previous_xg
        .groupby([df["player_id"], df["season"]])
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    df["xa_form_3"] = (
        previous_xa
        .groupby([df["player_id"], df["season"]])
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # Minutes form helps the model distinguish regular starters
    # from players whose points came from very small appearances.
    df["minutes_form_3"] = (
        previous_minutes
        .groupby([df["player_id"], df["season"]])
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # BPS is useful because it captures overall FPL performance
    # beyond goals and assists.
    df["bps_form_3"] = (
        previous_bps
        .groupby([df["player_id"], df["season"]])
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    return df


def add_consistency_features(df):
    print("Adding consistency features...")

    player_group = df.groupby(
        ["player_id", "season"],
        group_keys=False
    )

    previous_points = player_group["total_points"].shift(1)

    # Rolling standard deviation tells us how consistent the player's
    # recent points have been.
    df["points_std_5"] = (
        previous_points
        .groupby([df["player_id"], df["season"]])
        .transform(lambda x: x.rolling(5, min_periods=2).std())
    )

    # Recent high-point games are especially important for our target.
    df["high_points_last_5"] = (
        previous_points
        .groupby([df["player_id"], df["season"]])
        .transform(
            lambda x: x.rolling(5, min_periods=1)
            .apply(lambda values: (values >= 6).sum())
        )
    )

    return df


def add_context_features(df):
    print("Adding Gameweek context features...")

    # Gameweek number gives the model some information about where
    # we are in the season.
    df["gameweek_progress"] = (
        df["gameweek"] / 38.0
    )

    # Position is kept as a categorical value for now.
    # The model will handle it later rather than converting it
    # into an arbitrary numerical ranking.
    df["position"] = df["position"].fillna("Unknown")

    return df


def clean_features(df):
    print("Cleaning feature values...")

    # Sort before saving so sequence creation is deterministic.
    df = df.sort_values(
        ["player_id", "season", "gameweek"]
    ).reset_index(drop=True)

    # Rolling features are naturally missing at the beginning
    # of a player's season.
    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    df[numeric_columns] = df[numeric_columns].replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    # For the first few Gameweeks, there may not be enough history
    # for every rolling statistic. Zero is a simple starting value.
    df[numeric_columns] = df[numeric_columns].fillna(0)

    return df


def save_data(df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Feature shape: {df.shape}")
    print(f"Saved to: {OUTPUT_FILE}")


def main():
    df = load_data()

    df = add_basic_features(df)

    df = add_lag_features(df)

    df = add_rolling_features(df)

    df = add_consistency_features(df)

    df = add_context_features(df)

    df = clean_features(df)

    save_data(df)

    print("Feature engineering completed successfully.")


if __name__ == "__main__":
    main()