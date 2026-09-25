# BiLSTM (Default)

A dual-expert gate architecture for predicting player points.

- A 2-layer BiLSTM with temporal attention builds a trajectory representation from each player's last 5 gameweeks.
- Routed to a **low-band expert** (baseline appearances) or **high-band expert** (scoring returns).
- A **haul-potential calibrator** handles explosive, double-digit gameweeks.

## Evaluation (2025–26 test set, 23,406 samples)

| Metric | Overall | 0–2 pts | 3–5 pts | 6–9 pts | 10+ pts |
|---|---|---|---|---|---|
| MAE | 2.2315 | 2.072 | 3.998 | 1.610 | 4.453 |
| RMSE | 3.6546 | 3.634 | 4.192 | 2.171 | 5.318 |
| Spearman | 0.7183 | — | — | — | — |

> **TODO:** Insert `models/BiLSTM_model/reports/bandwise_results.png` here.

## Training

```bash
python models/BiLSTM_model/gate_architecture/train.py
python models/BiLSTM_model/gate_architecture/predict.py
```

Output: `models/BiLSTM_model/predicted_points.csv`
