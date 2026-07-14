import json
from pathlib import Path

import psycopg

with open("data/raw/fpl/bootstrap_static.json", encoding="utf8") as f:
    data = json.load(f)

conn = psycopg.connect(
    host="localhost",
    dbname="FantasyXI",
    user="postgres",
    password="@fantasy123@",
    port=5432
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