import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

conn = psycopg.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
)

cur = conn.cursor()

folder = Path("data/raw/fpl/element_summary")

inserted = 0

for file in folder.glob("*.json"):

    player_id = int(file.stem)

    with open(file, encoding="utf-8") as f:
        data = json.load(f)

    history = data["history"]

    for match in history:

        cur.execute("""
        INSERT INTO raw.player_history(
            player_id,
            fixture_id,
            gameweek,
            opponent_team,
            kickoff_time,
            was_home,
            minutes,
            goals_scored,
            assists,
            clean_sheets,
            goals_conceded,
            own_goals,
            penalties_saved,
            penalties_missed,
            yellow_cards,
            red_cards,
            saves,
            bonus,
            bps,
            influence,
            creativity,
            threat,
            ict_index,
            expected_goals,
            expected_assists,
            expected_goal_involvements,
            expected_goals_conceded,
            total_points,
            value
        )
        VALUES (
            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,%s,%s,%s,%s
        );
        """, (

            player_id,
            match["fixture"],
            match["round"],
            match["opponent_team"],
            match["kickoff_time"],
            match["was_home"],
            match["minutes"],
            match["goals_scored"],
            match["assists"],
            match["clean_sheets"],
            match["goals_conceded"],
            match["own_goals"],
            match["penalties_saved"],
            match["penalties_missed"],
            match["yellow_cards"],
            match["red_cards"],
            match["saves"],
            match["bonus"],
            match["bps"],
            float(match["influence"]),
            float(match["creativity"]),
            float(match["threat"]),
            float(match["ict_index"]),
            float(match["expected_goals"]),
            float(match["expected_assists"]),
            float(match["expected_goal_involvements"]),
            float(match["expected_goals_conceded"]),
            match["total_points"],
            match["value"]

        ))

        inserted += 1

conn.commit()

print(f"Inserted {inserted} history rows")

cur.close()
conn.close()