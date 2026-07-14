import json
from pathlib import Path

import psycopg

# ---------- Load JSON ----------

json_path = Path("data/raw/fpl/bootstrap_static.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

players = data["elements"]

# ---------- Connect PostgreSQL ----------

conn = psycopg.connect(
    host="localhost",
    dbname="FantasyXI",
    user="postgres",
    password="@fantasy123@",
    port=5432
)

cur = conn.cursor()

# ---------- Insert ----------

for p in players:

    cur.execute(
        """
        INSERT INTO raw.players (
            id,
            code,
            first_name,
            second_name,
            web_name,
            team,
            team_code,
            element_type,
            now_cost,
            total_points,
            points_per_game,
            minutes,
            goals_scored,
            assists,
            clean_sheets,
            goals_conceded,
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
            selected_by_percent,
            transfers_in,
            transfers_out,
            status,
            chance_of_playing_this_round,
            chance_of_playing_next_round,
            news,
            photo,
            team_join_date
        )
        VALUES (
            %(id)s,
            %(code)s,
            %(first_name)s,
            %(second_name)s,
            %(web_name)s,
            %(team)s,
            %(team_code)s,
            %(element_type)s,
            %(now_cost)s,
            %(total_points)s,
            %(points_per_game)s,
            %(minutes)s,
            %(goals_scored)s,
            %(assists)s,
            %(clean_sheets)s,
            %(goals_conceded)s,
            %(saves)s,
            %(bonus)s,
            %(bps)s,
            %(influence)s,
            %(creativity)s,
            %(threat)s,
            %(ict_index)s,
            %(expected_goals)s,
            %(expected_assists)s,
            %(expected_goal_involvements)s,
            %(expected_goals_conceded)s,
            %(selected_by_percent)s,
            %(transfers_in)s,
            %(transfers_out)s,
            %(status)s,
            %(chance_of_playing_this_round)s,
            %(chance_of_playing_next_round)s,
            %(news)s,
            %(photo)s,
            %(team_join_date)s
        )
        ON CONFLICT (id) DO NOTHING;
        """,
        p,
    )

conn.commit()

cur.close()
conn.close()

print(f"Inserted {len(players)} players.")