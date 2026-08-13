# Fantasy XI OpenFPL-Inspired Points Predictor

This project builds a standalone FPL player-points prediction layer using the already-audited Supabase `processed` tables as the primary data source.

## Data Route

Route A was selected: Supabase is sufficient. The local audit found historical player match stats, fixtures/difficulty fields, market/value transfer data, player status/availability, teams, identity history, and Understat-linked fields. The database was treated as read-only.

Local extracted raw data lives in `data/raw/`. The feature matrix is `data/processed/model_features.csv`.

## Model

- Chronological train/validation/test split:
  - train: seasons through `2023-24`
  - validation: `2024-25`
  - test: `2025-26`
- Position-specific models for `GK`, `DEF`, `MID`, `FWD`
- XGBoost candidate search with high-score-aware sample weighting
- Random Forest component per position
- Validation-selected XGBoost/RF blend per position
- Required target-range evaluation: `0-2`, `3-5`, `6-9`, `10+`

## Reproduce

Use the Python 3.12 training environment:

```powershell
py -3.12 -m venv .train_venv
.\.train_venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://pypi.org/simple
```

The Supabase audit and extraction were already completed. To rebuild features from existing local CSVs:

```powershell
.\.train_venv\Scripts\python.exe scripts\build_features.py
```

Train and evaluate:

```powershell
.\.train_venv\Scripts\python.exe scripts\train.py
.\.train_venv\Scripts\python.exe scripts\evaluate_components.py
```

## Outputs

- Model bundle: `models/position_ensemble.joblib`
- Training metadata: `models/training_metadata.json`
- Main evaluation: `reports/evaluation_2025_26.csv`
- Component comparison: `reports/evaluation_components_2025_26.csv`
- Full 2025-26 test predictions: `predictions/test_2025_26_predictions.csv`
- Latest available gameweek predictions: `predictions/final_predictions_latest_gameweek.csv`
