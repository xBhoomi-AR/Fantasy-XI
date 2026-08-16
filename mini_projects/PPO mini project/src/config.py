from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "prices.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "features.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = RESULTS_DIR / "models" / "ppo_portfolio.zip"
FIGURE_PATH = RESULTS_DIR / "figures" / "portfolio_metrics.png"

STOCKS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "JPM", "KO", "PEP"]
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

WINDOW = 20
EPISODE_LENGTH = 252
INITIAL_CAPITAL = 100_000
TRANSACTION_COST = 0.001
NUMBER_OF_ENVIRONMENTS = 4
TOTAL_TIMESTEPS = 100_000
SEED = 42
