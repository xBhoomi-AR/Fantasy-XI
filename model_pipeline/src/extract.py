from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# Load .env from the main Fantasy-XI folder
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")

import os

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")


# Create connection to the PostgreSQL database
engine = create_engine(DATABASE_URL)


def extract_player_data():
    query = """
        SELECT *
        FROM processed.player_match_stats
        ORDER BY season, gameweek, player_id
    """

    print("Connecting to database...")
    
    with engine.connect() as connection:
        count = connection.execute(
            text("SELECT COUNT(*) FROM processed.player_match_stats")
        ).scalar()

    print(f"Rows available: {count}")

    print("Loading player match data...")

    df = pd.read_sql(query, engine)

    output_path = project_root / "model_pipeline" / "data" / "raw" / "player_match_stats.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Data extracted successfully.")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    extract_player_data()