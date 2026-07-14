import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

with open("data/raw/fpl/bootstrap_static.json", encoding="utf8") as f:
    data = json.load(f)

load_dotenv()

conn = psycopg.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
)

cur = conn.cursor()

# ---------- Teams ----------

for t in data["teams"]:

    cur.execute("""
    INSERT INTO raw.teams VALUES (
    %(id)s,
    %(code)s,
    %(name)s,
    %(short_name)s,
    %(strength)s,
    %(strength_overall_home)s,
    %(strength_overall_away)s,
    %(strength_attack_home)s,
    %(strength_attack_away)s,
    %(strength_defence_home)s,
    %(strength_defence_away)s
    )
    ON CONFLICT (id) DO NOTHING;
    """, t)

# ---------- Events ----------

for e in data["events"]:

    cur.execute("""
    INSERT INTO raw.events(
        id,
        name,
        deadline_time,
        finished,
        is_current,
        is_next,
        is_previous
    )
    VALUES(
        %(id)s,
        %(name)s,
        %(deadline_time)s,
        %(finished)s,
        %(is_current)s,
        %(is_next)s,
        %(is_previous)s
    )
    ON CONFLICT(id) DO NOTHING;
    """, e)

# ---------- Positions ----------

for p in data["element_types"]:

    cur.execute("""
    INSERT INTO raw.element_types(
        id,
        plural_name,
        singular_name,
        singular_name_short,
        squad_select,
        squad_min_play,
        squad_max_play
    )
    VALUES(
        %(id)s,
        %(plural_name)s,
        %(singular_name)s,
        %(singular_name_short)s,
        %(squad_select)s,
        %(squad_min_play)s,
        %(squad_max_play)s
    )
    ON CONFLICT(id) DO NOTHING;
    """, p)

conn.commit()

cur.close()
conn.close()

print("Metadata inserted successfully!")