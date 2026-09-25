# Training & Retraining

The trained artifacts already in this repository (`ppo_fpl_v4.zip`, BiLSTM checkpoints, XGBoost models) are sufficient to run everything in [Running the Demo](usage.md) — none of this is required for the demo.

## Retrain BiLSTM

```bash
python models/BiLSTM_model/gate_architecture/train.py
python models/BiLSTM_model/gate_architecture/predict.py
```

## Retrain XGBoost

```bash
python models/xgboost_model/scripts/train.py
```

## Retrain PPO

```bash
python -m rl_decision_layer.ppo.train --timesteps 50000
```

!!! warning
    This always overwrites whatever file is at its save path (default `ppo_fpl_pure_rl.zip`). **Never pass `--save-name ppo_fpl_v4`** — that name is reserved for the frozen, evaluated model this documentation's results describe. A freshly retrained model will not reproduce the same behavior even with identical hyperparameters.

More detail on every command and test suite: `rl_decision_layer/README.md`, `models/xgboost_model/README.md`.
