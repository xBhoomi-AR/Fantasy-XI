import soccerdata as sd
import pandas as pd
import psycopg2
import unidecode


# -------------------------
# Get Understat players
# -------------------------

understat = sd.Understat(
    leagues="ENG-Premier League",
    seasons="2526"
)

df = understat.read_player_season_stats().reset_index()

understat_players = df[
    ["player_id", "player"]
].drop_duplicates()


# -------------------------
# SQL connection
# -------------------------

conn = psycopg2.connect(
    host="localhost",
    database="Fantasy XI",
    user="postgres",
    password="pokefan22#",
    port="5432"
)

cur = conn.cursor()


cur.execute("""
SELECT
    player_id,
    player_name,
    understat_id
FROM processed.players;
""")


sql_players = pd.DataFrame(
    cur.fetchall(),
    columns=[
        "player_id",
        "player_name",
        "understat_id"
    ]
)


# -------------------------
# Normalize names
# -------------------------

def clean(name):

    name = unidecode.unidecode(str(name))

    name = (
        name.lower()
        .replace("-", " ")
        .replace(".", "")
        .replace("'", "")
        .strip()
    )

    return name


understat_players["clean_name"] = (
    understat_players["player"]
    .apply(clean)
)


sql_players["clean_name"] = (
    sql_players["player_name"]
    .apply(clean)
)


# -------------------------
# Fuzzy Match Understat -> SQL players
# -------------------------

from rapidfuzz import process, fuzz


matches = []


sql_names = sql_players["player_name"].tolist()


for _, row in understat_players.iterrows():

    result = process.extractOne(
        row["player"],
        sql_names,
        scorer=fuzz.token_set_ratio
    )

    if result:

        matched_name = result[0]
        score = result[1]


        # accept only good matches
        if score >= 75:

            sql_row = sql_players[
                sql_players["player_name"] == matched_name
            ].iloc[0]


            matches.append(
                {
                    "understat_id": int(row["player_id"]),
                    "player": row["player"],
                    "internal_id": int(sql_row["player_id"]),
                    "player_name": matched_name,
                    "score": score
                }
            )


matches = pd.DataFrame(matches)


print(matches.head(30))

print(
    "Mapped:",
    len(matches)
)


print(
    "Lowest scores:"
)

print(
    matches.sort_values("score").head(20)
)

# ==============================
# Manual mappings for remaining players
# ==============================

manual_map = {
    12168: 713,   # Alejandro Jiménez
    11231: 391,   # Ben Doak
    12032: 67,    # Djordje Petrovic
    6986: 93,     # Hamed Traore
    9501: 123,    # Fabio Carvalho
    9451: 676,    # Lesley Ugochukwu
    9024: 712,    # Yeremi Pino
    9983: 311,    # Beto
    7230: 325,    # Emile Smith Rowe
    12203: 396,   # Trey Nyoni
    8094: 417,    # Mathis Cherki
    2496: 421,    # Rodri
    2248: 457,    # Casemiro
    12766: 518,   # Jota Silva
    13068: 511,   # Morato
    7365: 612,    # Lucas Paqueta
    13200: 645,   # Fernando Lopez
    14030: 731,   # Kevin Santos
    9156: 120    # Kevin Schade

}


for understat_id, internal_id in manual_map.items():

    cur.execute(
        """
        UPDATE processed.players
        SET understat_id = %s
        WHERE player_id = %s;
        """,
        (understat_id, internal_id)
    )


conn.commit()

print("Manual mappings inserted!")

cur.execute("""
SELECT COUNT(*)
FROM processed.players
WHERE understat_id IS NOT NULL;
""")

print("Total mapped:", cur.fetchone()[0])

# -------------------------
# Update processed.players
# -------------------------

update_query = """
UPDATE processed.players
SET understat_id = %s
WHERE player_id = %s;
"""


for _, row in matches.iterrows():

    try:

        cur.execute(
            update_query,
            (
                row["understat_id"],
                row["internal_id"]
            )
        )

    except psycopg2.errors.UniqueViolation:

        conn.rollback()

        print(
            "Duplicate skipped:",
            row["player"],
            "->",
            row["player_name"],
            row["understat_id"]
        )

conn.commit()


print("Understat mappings updated!")

# -------------------------
# Check remaining unmapped from DATABASE
# -------------------------

conn.commit()   # make sure all updates are saved first

cur.execute("""
SELECT understat_id
FROM processed.players
WHERE understat_id IS NOT NULL;
""")

db_mapped_ids = {
    int(row[0])
    for row in cur.fetchall()
    if row[0] is not None
}

missing = understat_players[
    ~understat_players["player_id"].astype(int).isin(db_mapped_ids)
]

print("\nStill missing:")

if missing.empty:
    print("None 🎉")
else:
    print(
        missing[
            ["player_id", "player"]
        ].sort_values("player").to_string(index=False)
    )

print("\nMissing count:", len(missing))

# =====================================================
# Insert Understat season stats into PostgreSQL
# =====================================================

stats = df.merge(
    sql_players[["player_id", "understat_id"]],
    left_on="player_id",
    right_on="understat_id",
    how="inner"
)

print("Players with stats:", stats["player_id_y"].nunique())
insert_query = """
INSERT INTO processed.player_understat_season_stats (
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
VALUES (
    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
)
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

for _, row in stats.iterrows():

    cur.execute(
    insert_query,
    (
        int(row["player_id_y"]),
        str(row["season"]),
        int(row["matches"]),
        int(row["minutes"]),
        int(row["goals"]),
        int(row["shots"]),
        float(row["xg"]),
        int(row["assists"]),
        int(row["key_passes"]),
        float(row["xa"]),
        float(row["xg_chain"]),
        float(row["xg_buildup"]),
        int(row["np_goals"]),
        float(row["np_xg"]),
    )
)

conn.commit()

print(f"Inserted/Updated {len(stats)} season stat rows.")

