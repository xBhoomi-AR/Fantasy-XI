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

with open("data/raw/fpl/fixtures.json", encoding="utf-8") as f:
    fixtures = json.load(f)

for fixture in fixtures:

    cur.execute("""
        INSERT INTO raw.fixtures
        VALUES (
            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
        )
        ON CONFLICT (id)
        DO NOTHING
    """, (

        fixture["id"],
        fixture["code"],
        fixture["pulse_id"],

        fixture["event"],

        fixture["kickoff_time"],

        fixture["finished"],
        fixture["started"],

        fixture["minutes"],

        fixture["team_h"],
        fixture["team_a"],

        fixture["team_h_score"],
        fixture["team_a_score"],

        fixture["team_h_difficulty"],
        fixture["team_a_difficulty"]

    ))

conn.commit()

print(f"Inserted {len(fixtures)} fixtures")

cur.close()
conn.close()