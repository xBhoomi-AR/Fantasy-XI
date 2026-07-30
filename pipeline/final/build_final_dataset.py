import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")


SEQUENCE_LENGTH = 5
OUTPUT_DIR = Path("pipeline/final/data")


# Stats from matches the player has already played
SEQUENCE_FEATURES = [
    "minutes",
    "total_points",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "shots",
    "key_passes",
    "clean_sheets",
    "goals_conceded",
    "expected_goals_conceded",
    "saves",
    "recoveries",
    "tackles",
    "clearances_blocks_interceptions",
    "defensive_contribution",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "npxg",
    "xg",
    "xa",
    "xgi",
]


# Information we can know before the target fixture starts
CONTEXT_FEATURES = [
    "was_home",
    "fixture_difficulty",
    "value",
    "transfers_balance",
    "team_strength",
    "team_attack_strength",
    "team_defence_strength",
    "opponent_strength",
    "opponent_attack_strength",
    "opponent_defence_strength",
    "position_GK",
    "position_DEF",
    "position_MID",
    "position_FWD",
]


TARGET = "total_points"


def load_data():
    print("Loading final dataset from Supabase...")

    query = """
        SELECT
            pms.player_id,
            pms.fixture_id,
            pms.team_id,
            pms.opponent_team_id,
            pms.gameweek,
            pms.season,
            pms.was_home,
            pms.position,

            pms.minutes,
            pms.total_points,
            pms.goals_scored,
            pms.assists,
            pms.expected_goals,
            pms.expected_assists,
            pms.expected_goal_involvements,
            pms.shots,
            pms.key_passes,
            pms.clean_sheets,
            pms.goals_conceded,
            pms.expected_goals_conceded,
            pms.saves,
            pms.recoveries,
            pms.tackles,
            pms.clearances_blocks_interceptions,
            pms.defensive_contribution,
            pms.yellow_cards,
            pms.red_cards,
            pms.own_goals,
            pms.penalties_saved,
            pms.penalties_missed,
            pms.bonus,
            pms.bps,
            pms.influence,
            pms.creativity,
            pms.threat,
            pms.ict_index,
            pms.npxg,
            pms.xg,
            pms.xa,
            pms.xgi,

            CASE
                WHEN pms.was_home THEN f.team_h_difficulty
                ELSE f.team_a_difficulty
            END AS fixture_difficulty,

            pmh.value,
            pmh.transfers_balance,

            t.strength AS team_strength,

            CASE
                WHEN pms.was_home THEN t.strength_attack_home
                ELSE t.strength_attack_away
            END AS team_attack_strength,

            CASE
                WHEN pms.was_home THEN t.strength_defence_home
                ELSE t.strength_defence_away
            END AS team_defence_strength,

            opp.strength AS opponent_strength,

            CASE
                WHEN pms.was_home THEN opp.strength_attack_away
                ELSE opp.strength_attack_home
            END AS opponent_attack_strength,

            CASE
                WHEN pms.was_home THEN opp.strength_defence_away
                ELSE opp.strength_defence_home
            END AS opponent_defence_strength

        FROM processed.player_match_stats AS pms

        LEFT JOIN processed.fixtures AS f
            ON pms.fixture_id = f.fixture_id
            AND pms.season = f.season

        LEFT JOIN processed.player_market_history AS pmh
            ON pms.player_id = pmh.player_id
            AND pms.fixture_id = pmh.fixture_id
            AND pms.season = pmh.season

        LEFT JOIN processed.teams AS t
            ON pms.team_id = t.team_id

        LEFT JOIN processed.teams AS opp
            ON pms.opponent_team_id = opp.team_id

        ORDER BY
            pms.season,
            pms.player_id,
            pms.gameweek;
    """

    connection = psycopg2.connect(DATABASE_URL)

    try:
        df = pd.read_sql_query(query, connection)
    finally:
        connection.close()

    print(f"Loaded {len(df):,} rows")

    return df


def prepare_data(df):
    print("Preparing features...")

    df = df.sort_values(
        ["season", "player_id", "gameweek"]
    ).reset_index(drop=True)

    # Make home/away usable by the model
    df["was_home"] = df["was_home"].astype(float)

    # Position is categorical, so use one column for each position
    position = (
        df["position"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["position_GK"] = (position == "GK").astype(float)
    df["position_DEF"] = (position == "DEF").astype(float)
    df["position_MID"] = (position == "MID").astype(float)
    df["position_FWD"] = (position == "FWD").astype(float)

    numeric_columns = (
        SEQUENCE_FEATURES
        + CONTEXT_FEATURES
        + [TARGET]
    )

    # Remove duplicates because total_points is also the target
    numeric_columns = list(dict.fromkeys(numeric_columns))

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Match stats missing for a player are treated as zero
    df[SEQUENCE_FEATURES] = (
        df[SEQUENCE_FEATURES].fillna(0)
    )

    # Context uses the median when a numeric value is unavailable
    for column in CONTEXT_FEATURES:
        if df[column].isna().any():
            median = df[column].median()

            if pd.isna(median):
                median = 0

            df[column] = df[column].fillna(median)

    df = df.dropna(subset=[TARGET])

    print(f"Rows after preparation: {len(df):,}")

    return df


def build_samples(df):
    print(
        f"Building {SEQUENCE_LENGTH}-match "
        "sequence + context samples..."
    )

    sequences = []
    contexts = []
    targets = []
    metadata = []

    grouped = df.groupby(
        ["season", "player_id"],
        sort=False
    )

    for (season, player_id), player_df in grouped:
        player_df = player_df.sort_values(
            ["gameweek", "fixture_id"]
        ).reset_index(drop=True)

        if len(player_df) <= SEQUENCE_LENGTH:
            continue

        sequence_values = player_df[
            SEQUENCE_FEATURES
        ].to_numpy(dtype="float32")

        context_values = player_df[
            CONTEXT_FEATURES
        ].to_numpy(dtype="float32")

        target_values = player_df[
            TARGET
        ].to_numpy(dtype="float32")

        for i in range(
            SEQUENCE_LENGTH,
            len(player_df)
        ):
            # Only previous matches go into the LSTM
            sequence = sequence_values[
                i - SEQUENCE_LENGTH:i
            ]

            # Context comes from the fixture being predicted
            context = context_values[i]

            target = target_values[i]

            sequences.append(sequence)
            contexts.append(context)
            targets.append(target)

            metadata.append({
                "season": season,
                "player_id": player_id,
                "fixture_id": player_df.iloc[i]["fixture_id"],
                "gameweek": player_df.iloc[i]["gameweek"],
                "team_id": player_df.iloc[i]["team_id"],
                "opponent_team_id": player_df.iloc[i]["opponent_team_id"],
            })

    return sequences, contexts, targets, metadata


def save_dataset(
    sequences,
    contexts,
    targets,
    metadata
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X_sequence = np.asarray(
        sequences,
        dtype="float32"
    )

    X_context = np.asarray(
        contexts,
        dtype="float32"
    )

    y = np.asarray(
        targets,
        dtype="float32"
    )

    metadata_df = pd.DataFrame(metadata)

    np.save(
        OUTPUT_DIR / "X_sequence.npy",
        X_sequence
    )

    np.save(
        OUTPUT_DIR / "X_context.npy",
        X_context
    )

    np.save(
        OUTPUT_DIR / "y.npy",
        y
    )

    metadata_df.to_csv(
        OUTPUT_DIR / "metadata.csv",
        index=False
    )

    # Save feature names so we always know what the model used
    pd.Series(
        SEQUENCE_FEATURES,
        name="sequence_feature"
    ).to_csv(
        OUTPUT_DIR / "sequence_features.csv",
        index=False
    )

    pd.Series(
        CONTEXT_FEATURES,
        name="context_feature"
    ).to_csv(
        OUTPUT_DIR / "context_features.csv",
        index=False
    )

    print()
    print("Final dataset created")
    print("Sequence:", X_sequence.shape)
    print("Context: ", X_context.shape)
    print("Target:  ", y.shape)
    print("Metadata:", metadata_df.shape)

    print()
    print(
        f"Historical features: {len(SEQUENCE_FEATURES)}"
    )
    print(
        f"Context features: {len(CONTEXT_FEATURES)}"
    )

    print()
    print("Saved in pipeline/final/data/")


def main():
    df = load_data()
    df = prepare_data(df)

    sequences, contexts, targets, metadata = (
        build_samples(df)
    )

    if not sequences:
        raise RuntimeError(
            "No final samples were generated"
        )

    save_dataset(
        sequences,
        contexts,
        targets,
        metadata
    )


if __name__ == "__main__":
    main()