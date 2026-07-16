import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import psycopg2
from psycopg2.extras import execute_values

CACHE_DIR = "cache"
BOOTSTRAP_CACHE = os.path.join(CACHE_DIR, "bootstrap_static.json")

BASE_URL = "https://fantasy.premierleague.com/api"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

PLAYER_HISTORY_CACHE = os.path.join(CACHE_DIR, "player_histories.pkl")

DB_CONFIG = {
    "dbname": "Fantasy XI",
    "user": "postgres",
    "password": "pokefan22#",
    "host": "localhost",
    "port": "5432"
}

def download_bootstrap(force_download=False):

    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(BOOTSTRAP_CACHE) and not force_download:

        print("Loading bootstrap-static from cache...")

        with open(BOOTSTRAP_CACHE, "r", encoding="utf-8") as file:
            return json.load(file)

    print("Downloading bootstrap-static...")

    response = requests.get(
        f"{BASE_URL}/bootstrap-static/",
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    bootstrap = response.json()

    with open(BOOTSTRAP_CACHE, "w", encoding="utf-8") as file:
        json.dump(bootstrap, file, indent=4)

    print("bootstrap-static cached successfully.")

    return bootstrap

############################################################
# LOOKUP TABLES
############################################################

def create_lookup_tables(bootstrap):

    players = bootstrap["elements"]

    player_team = {}

    player_position = {}

    position_map = {

        1: "GK",

        2: "DEF",

        3: "MID",

        4: "FWD"

    }

    for player in players:

        player_team[player["id"]] = player["team"]

        player_position[player["id"]] = position_map[
            player["element_type"]
        ]

    print(f"Lookup tables created for {len(players)} players.")

    return player_team, player_position

############################################################
# FETCH SINGLE PLAYER HISTORY
############################################################

def fetch_player_history(player_id):

    try:

        response = requests.get(
            f"{BASE_URL}/element-summary/{player_id}/",
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        summary = response.json()

        return player_id, summary["history"]

    except requests.exceptions.RequestException as e:

        print(f"Failed to download player {player_id}: {e}")

        return player_id, []
    
    ############################################################
# DOWNLOAD ALL PLAYER HISTORIES
############################################################

def download_all_histories(players, force_download=False):

    if os.path.exists(PLAYER_HISTORY_CACHE) and not force_download:

        print("Loading player histories from cache...")

        with open(PLAYER_HISTORY_CACHE, "rb") as file:

            return pickle.load(file)

    print("Downloading player histories...")

    all_histories = {}

    with ThreadPoolExecutor(max_workers=20) as executor:

        futures = {

            executor.submit(fetch_player_history, player["id"]): player["id"]

            for player in players

        }

        completed = 0

        total = len(players)

        for future in as_completed(futures):

            player_id, history = future.result()

            all_histories[player_id] = history

            completed += 1

            if completed % 25 == 0 or completed == total:

                print(f"{completed}/{total} players downloaded")

    with open(PLAYER_HISTORY_CACHE, "wb") as file:

        pickle.dump(all_histories, file)

    print("Player histories cached successfully.")

    return all_histories
############################################################
# BUILD PLAYER MATCH STAT ROWS
############################################################

CURRENT_SEASON = "2025/26"

def build_rows(all_histories, player_team, player_position):

    rows = []

    print("Building rows...")

    for player_id, history in all_histories.items():

        for match in history:

            rows.append(

                (

                    player_id,

                    match["fixture"],

                    player_team[player_id],

                    match["opponent_team"],

                    match["round"],

                    CURRENT_SEASON,

                    match["was_home"],

                    player_position[player_id],

                    bool(match["starts"]),

                    match["minutes"],

                    match["total_points"],

                    match["goals_scored"],

                    match["assists"],

                    float(match["expected_goals"]),

                    float(match["expected_assists"]),

                    float(match["expected_goal_involvements"]),

                    None,      # shots (FBRef later)

                    None,      # key_passes (FBRef later)

                    match["clean_sheets"],

                    match["goals_conceded"],

                    float(match["expected_goals_conceded"]),

                    match["saves"],

                    match["recoveries"],

                    match["tackles"],

                    match["clearances_blocks_interceptions"],

                    match["defensive_contribution"],

                    match["yellow_cards"],

                    match["red_cards"],

                    match["own_goals"],

                    match["penalties_saved"],

                    match["penalties_missed"],

                    match["bonus"],

                    match["bps"],

                    float(match["influence"]),

                    float(match["creativity"]),

                    float(match["threat"]),

                    float(match["ict_index"])

                )

            )

    print(f"Rows built: {len(rows)}")

    return rows

############################################################
# INSERT ROWS INTO POSTGRESQL
############################################################

def insert_rows(rows):

    if not rows:
        print("No rows to insert.")
        return

    print("Connecting to PostgreSQL...")

    connection = None

    try:

        connection = psycopg2.connect(**DB_CONFIG)

        cursor = connection.cursor()

        # Count rows before insert
        cursor.execute("""
            SELECT COUNT(*)
            FROM processed.player_match_stats;
        """)

        rows_before = cursor.fetchone()[0]

        query = """
        INSERT INTO processed.player_match_stats (

            player_id,
            fixture_id,
            team_id,
            opponent_team_id,
            gameweek,
            season,

            was_home,
            position,
            started,

            minutes,
            total_points,

            goals_scored,
            assists,

            expected_goals,
            expected_assists,
            expected_goal_involvements,

            shots,
            key_passes,

            clean_sheets,
            goals_conceded,
            expected_goals_conceded,

            saves,
            recoveries,
            tackles,

            clearances_blocks_interceptions,
            defensive_contribution,

            yellow_cards,
            red_cards,

            own_goals,

            penalties_saved,
            penalties_missed,

            bonus,
            bps,

            influence,
            creativity,
            threat,
            ict_index

        )

        VALUES %s

        ON CONFLICT (player_id, fixture_id) DO NOTHING;
        """

        execute_values(
            cursor,
            query,
            rows,
            page_size=1000
        )

        connection.commit()

        # Count rows after insert
        cursor.execute("""
            SELECT COUNT(*)
            FROM processed.player_match_stats;
        """)

        rows_after = cursor.fetchone()[0]

        print("\n========== INSERT SUMMARY ==========")
        print(f"Rows before : {rows_before}")
        print(f"Rows after  : {rows_after}")
        print(f"Inserted    : {rows_after - rows_before}")
        print("====================================")

    except Exception as e:

        print("\nDatabase Error")
        print(e)

        if connection:
            connection.rollback()

    finally:

        if connection:
            cursor.close()
            connection.close()


    ############################################################
# BUILD PLAYER MARKET HISTORY
############################################################

CURRENT_SEASON = "2025/26"

def build_market_rows(all_histories):

    rows = []

    print("Building player market history...")

    for player_id, history in all_histories.items():

        for match in history:

            rows.append(

                (

                    player_id,

                    match["fixture"],

                    match["round"],

                    CURRENT_SEASON,

                    match["value"],

                    match["transfers_in"],

                    match["transfers_out"],

                    match["transfers_balance"]

                )

            )

    print(f"Market rows built: {len(rows)}")

    return rows

############################################################
# INSERT PLAYER MARKET HISTORY
############################################################

def insert_market_rows(rows):

    if not rows:
        print("No rows to insert.")
        return

    print("Connecting to PostgreSQL...")

    connection = None

    try:

        connection = psycopg2.connect(**DB_CONFIG)

        cursor = connection.cursor()

        cursor.execute("""
            SELECT COUNT(*)
            FROM processed.player_market_history;
        """)

        before = cursor.fetchone()[0]

        query = """
        INSERT INTO processed.player_market_history (

            player_id,
            fixture_id,
            gameweek,
            season,
            value,
            transfers_in,
            transfers_out,
            transfers_balance

        )

        VALUES %s

        ON CONFLICT (player_id, fixture_id)
        DO NOTHING;
        """

        execute_values(
            cursor,
            query,
            rows,
            page_size=1000
        )

        connection.commit()

        cursor.execute("""
            SELECT COUNT(*)
            FROM processed.player_market_history;
        """)

        after = cursor.fetchone()[0]

        print("\n========== INSERT SUMMARY ==========")
        print(f"Rows before : {before}")
        print(f"Rows after  : {after}")
        print(f"Inserted    : {after-before}")
        print("====================================")

    except Exception as e:

        print(e)

        if connection:
            connection.rollback()

    finally:

        if connection:
            cursor.close()
            connection.close()

def main():

    print("\n===== PLAYER MARKET HISTORY ETL =====\n")

    bootstrap = download_bootstrap()

    players = bootstrap["elements"]

    histories = download_all_histories(players)

    market_rows = build_market_rows(histories)

    insert_market_rows(market_rows)

    print("\nETL Completed Successfully.")


if __name__ == "__main__":
    main()