# Installation

## Prerequisites

- Python 3.10+ (developed and tested on 3.12).
- No Node.js/npm/React — the frontend is plain HTML/CSS/JS served directly by the backend.

## Install dependencies

The whole project (both prediction models + the RL/MILP decision layer + backend):

```bash
pip install -r requirements.txt
```

Demo only (no retraining) — lighter, scoped install:

```bash
pip install -r rl_decision_layer/requirements.txt -r backend/requirements.txt
```

Each component also keeps its own minimal `requirements.txt` (`models/BiLSTM_model/`, `models/xgboost_model/`, `rl_decision_layer/`, `backend/`) if you only need one part of the project.

Next: [Running the Demo](usage.md)
