# XGBoost (Fallback)

Position-specific regressors, trained on engineered historical FPL/fixture/team/Understat features.

- Four separate models — one each for GK, DEF, MID, FWD.
- Trained on Supabase-hosted historical data.

## Evaluation (2025–26 test set)

| Metric | Overall | GK | DEF | MID | FWD |
|---|---|---|---|---|---|
| MAE | 1.2689 | — | — | — | — |
| RMSE | 2.1067 | — | — | — | — |
| Spearman (ranking) | 0.7947 | 0.7021 | 0.7881 | 0.8133 | 0.8115 |

> **TODO:** Insert `spearman_by_position.png` and `xgboost_range_mae_rmse.png` here.

## Training

```bash
python models/xgboost_model/scripts/train.py
```

Output: `models/xgboost_model/predictions/*.csv`
