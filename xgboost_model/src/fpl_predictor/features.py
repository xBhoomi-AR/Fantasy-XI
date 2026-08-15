from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import PROCESSED_DIR, RAW_DIR, ensure_dirs


WINDOWS = [1, 3, 5, 10, 38]

POSITION_MAP = {
    "1": "GK",
    "2": "DEF",
    "3": "MID",
    "4": "FWD",
    "5": "AM",
    "Goalkeeper": "GK",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Forward": "FWD",
    "GK": "GK",
    "DEF": "DEF",
    "MID": "MID",
    "FWD": "FWD",
}

SEASON_ORDER = {
    "2016-17": 2016,
    "2017-18": 2017,
    "2018-19": 2018,
    "2019-20": 2019,
    "2020-21": 2020,
    "2021-22": 2021,
    "2022-23": 2022,
    "2023-24": 2023,
    "2024-25": 2024,
    "2025-26": 2025,
}


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", **kwargs)


def normalise_position(value) -> str:
    if pd.isna(value):
        return "UNK"
    return POSITION_MAP.get(str(value), str(value).upper())


def add_shifted_rolling(
    frame: pd.DataFrame,
    group_cols: list[str],
    order_cols: list[str],
    value_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    frame = frame.sort_values(group_cols + order_cols).copy()
    grouped = frame.groupby(group_cols, sort=False)
    for col in value_cols:
        shifted = grouped[col].shift(1)
        frame[f"{prefix}_{col}_last1"] = shifted
        for window in WINDOWS[1:]:
            frame[f"{prefix}_{col}_avg{window}"] = shifted.groupby([frame[c] for c in group_cols]).rolling(window, min_periods=1).mean().reset_index(level=list(range(len(group_cols))), drop=True)
    return frame


def build_team_fixture_history(fixtures: pd.DataFrame) -> pd.DataFrame:
    home = fixtures.rename(
        columns={
            "home_team_id": "team_id",
            "away_team_id": "opponent_team_id",
            "team_h_score": "team_goals_for",
            "team_a_score": "team_goals_against",
            "team_h_difficulty": "fixture_difficulty",
        }
    ).assign(was_home=True)
    away = fixtures.rename(
        columns={
            "away_team_id": "team_id",
            "home_team_id": "opponent_team_id",
            "team_a_score": "team_goals_for",
            "team_h_score": "team_goals_against",
            "team_a_difficulty": "fixture_difficulty",
        }
    ).assign(was_home=False)
    cols = [
        "season",
        "season_order",
        "fixture_id",
        "gameweek",
        "team_id",
        "opponent_team_id",
        "was_home",
        "fixture_difficulty",
        "team_goals_for",
        "team_goals_against",
    ]
    team = pd.concat([home[cols], away[cols]], ignore_index=True)
    team["team_points_result"] = np.select(
        [
            team["team_goals_for"] > team["team_goals_against"],
            team["team_goals_for"] == team["team_goals_against"],
        ],
        [3, 1],
        default=0,
    )
    value_cols = ["team_goals_for", "team_goals_against", "team_points_result"]
    team = add_shifted_rolling(team, ["team_id"], ["season_order", "gameweek", "fixture_id"], value_cols, "team")
    opponent_features = team[
        [
            "season",
            "fixture_id",
            "team_id",
            "team_team_goals_for_last1",
            "team_team_goals_for_avg3",
            "team_team_goals_for_avg5",
            "team_team_goals_against_last1",
            "team_team_goals_against_avg3",
            "team_team_goals_against_avg5",
            "team_team_points_result_avg5",
        ]
    ].rename(
        columns={
            "team_id": "opponent_team_id",
            "team_team_goals_for_last1": "opp_goals_for_last1",
            "team_team_goals_for_avg3": "opp_goals_for_avg3",
            "team_team_goals_for_avg5": "opp_goals_for_avg5",
            "team_team_goals_against_last1": "opp_goals_against_last1",
            "team_team_goals_against_avg3": "opp_goals_against_avg3",
            "team_team_goals_against_avg5": "opp_goals_against_avg5",
            "team_team_points_result_avg5": "opp_points_result_avg5",
        }
    )
    return team, opponent_features


def build_features() -> pd.DataFrame:
    ensure_dirs()
    match = read_csv(RAW_DIR / "player_match_stats.csv")
    fixtures = read_csv(RAW_DIR / "fixtures.csv")
    market = read_csv(RAW_DIR / "player_market_history.csv")
    players = read_csv(RAW_DIR / "players.csv")

    match = match[match["season"].isin(SEASON_ORDER)].copy()
    fixtures = fixtures[fixtures["season"].isin(SEASON_ORDER)].copy()
    market = market[market["season"].isin(SEASON_ORDER)].copy()

    match["season_order"] = match["season"].map(SEASON_ORDER)
    fixtures["season_order"] = fixtures["season"].map(SEASON_ORDER)
    market["season_order"] = market["season"].map(SEASON_ORDER)
    match["position_norm"] = match["position"].map(normalise_position)
    match = match[match["position_norm"].isin(["GK", "DEF", "MID", "FWD"])].copy()

    numeric_cols = [
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
    for col in numeric_cols:
        match[col] = pd.to_numeric(match[col], errors="coerce").fillna(0)

    match = match.drop_duplicates(["season", "fixture_id", "player_id"], keep="last")
    match = match.sort_values(["player_id", "season_order", "gameweek", "fixture_id"]).copy()
    match["played_last_match"] = (match.groupby("player_id")["minutes"].shift(1).fillna(0) > 0).astype(int)
    match["started_last_match"] = match.groupby("player_id")["started"].shift(1).map({"t": 1, "f": 0, True: 1, False: 0}).fillna(0)
    match["high_score_flag"] = (match["total_points"] >= 6).astype(int)

    player_roll_cols = [
        "total_points",
        "minutes",
        "goals_scored",
        "assists",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "clean_sheets",
        "goals_conceded",
        "expected_goals_conceded",
        "saves",
        "recoveries",
        "tackles",
        "defensive_contribution",
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
        "high_score_flag",
    ]
    match = add_shifted_rolling(match, ["player_id"], ["season_order", "gameweek", "fixture_id"], player_roll_cols, "player")

    market = market.drop_duplicates(["season", "fixture_id", "player_id"], keep="last")
    market_features = market[
        ["season", "fixture_id", "player_id", "value", "transfers_in", "transfers_out", "transfers_balance"]
    ].copy()
    for col in ["value", "transfers_in", "transfers_out", "transfers_balance"]:
        market_features[col] = pd.to_numeric(market_features[col], errors="coerce")

    fixture_side, opponent_features = build_team_fixture_history(fixtures)
    fixture_features = fixture_side[
        [
            "season",
            "fixture_id",
            "team_id",
            "opponent_team_id",
            "fixture_difficulty",
            "team_team_goals_for_last1",
            "team_team_goals_for_avg3",
            "team_team_goals_for_avg5",
            "team_team_goals_against_last1",
            "team_team_goals_against_avg3",
            "team_team_goals_against_avg5",
            "team_team_points_result_avg5",
        ]
    ]

    player_context = players[
        [
            "player_id",
            "status",
            "chance_of_playing_next_round",
            "chance_of_playing_this_round",
            "position_code",
        ]
    ].copy()
    player_context["status_code"] = player_context["status"].fillna("a").map({"a": 1.0, "d": 0.75, "i": 0.0, "s": 0.0, "u": 0.0}).fillna(0.5)
    for col in ["chance_of_playing_next_round", "chance_of_playing_this_round", "position_code"]:
        player_context[col] = pd.to_numeric(player_context[col], errors="coerce")
    player_context = player_context.drop(columns=["status"])

    features = match.merge(market_features, on=["season", "fixture_id", "player_id"], how="left")
    features = features.merge(fixture_features, on=["season", "fixture_id", "team_id", "opponent_team_id"], how="left")
    features = features.merge(opponent_features, on=["season", "fixture_id", "opponent_team_id"], how="left")
    features = features.merge(player_context, on="player_id", how="left")

    features["was_home_int"] = features["was_home"].map({"t": 1, "f": 0, True: 1, False: 0}).fillna(0).astype(int)
    features["target_points"] = features["total_points"].astype(float)
    features["position"] = features["position_norm"]

    features = features.sort_values(["season_order", "gameweek", "fixture_id", "player_id"]).reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED_DIR / "model_features.csv", index=False)
    return features


def feature_columns(frame: pd.DataFrame) -> list[str]:
    exclude = {
        "target_points",
        "total_points",
        "position",
        "position_norm",
        "started",
        "understat_position",
        "season",
        "fixture_id",
    }
    numeric = frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cols = [c for c in numeric if c not in exclude]
    leakage_patterns = re.compile(r"^(goals_scored|assists|expected_|shots|key_passes|clean_sheets|goals_conceded|saves|recoveries|tackles|bonus|bps|influence|creativity|threat|ict_index|npxg|xg|xa|xgi|minutes|high_score_flag)$")
    return [c for c in cols if not leakage_patterns.match(c)]
