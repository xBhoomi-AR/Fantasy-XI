import soccerdata as sd
import pandas as pd
import psycopg2


# 1. Fetch Understat data again
understat = sd.Understat(
    leagues="ENG-Premier League",
    seasons="2526"
)

players = understat.read_player_season_stats()

print(players.head())
print(players.columns)


# 2. Reset index because player name is currently in index
players = players.reset_index()

understat_sql = players.copy()

understat_sql = understat_sql.rename(columns={
    "matches": "matches_played",
    "np_goals": "npg",
    "np_xg": "npxg"
})


understat_sql = understat_sql[
    [
        "player_id",
        "season",
        "matches_played",
        "minutes",
        "goals",
        "shots",
        "xg",
        "assists",
        "key_passes",
        "xa",
        "xg_chain",
        "xg_buildup",
        "npg",
        "npxg"
    ]
]


print(understat_sql.head())
print(understat_sql.shape)

print(players.columns)
print(players[['player_id','season','player','xg','xa']].head())


conn = psycopg2.connect(
    host="localhost",
    database="Fantasy XI",
    user="postgres",
    password="pokefan22#",
    port="5432"
)

cur = conn.cursor()


insert_query = """
INSERT INTO processed.player_understat_season_stats
(
    player_id,
    season,
    matches_played,
    minutes,
    goals,
    shots,
    xg,
    assists,
    key_passes,
    xa,
    xg_chain,
    xg_buildup,
    npg,
    npxg
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (player_id, season)
DO UPDATE SET
    matches_played = EXCLUDED.matches_played,
    minutes = EXCLUDED.minutes,
    goals = EXCLUDED.goals,
    shots = EXCLUDED.shots,
    xg = EXCLUDED.xg,
    assists = EXCLUDED.assists,
    key_passes = EXCLUDED.key_passes,
    xa = EXCLUDED.xa,
    xg_chain = EXCLUDED.xg_chain,
    xg_buildup = EXCLUDED.xg_buildup,
    npg = EXCLUDED.npg,
    npxg = EXCLUDED.npxg;
"""


for _, row in understat_sql.iterrows():
    cur.execute(insert_query, tuple(row))


conn.commit()

cur.close()
conn.close()

print("Understat season stats populated successfully!")