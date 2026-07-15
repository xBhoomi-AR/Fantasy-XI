from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATABASE = {
    "host": "localhost",
    "port": 5432,
    "database": "fantasy_xi",
    "user": "postgres",
    "password": "pokefan22#"
}

FPL_BASE_URL = "https://fantasy.premierleague.com/api"
UNDERSTAT_BASE_URL = "https://understat.com"
FBREF_BASE_URL = "https://fbref.com"

REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": "FantasyXI/1.0"
}