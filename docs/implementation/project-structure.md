# Project Structure

```text
FantasyXI/
├── models/
│   ├── BiLSTM_model/          # Dual-expert BiLSTM prediction model (default)
│   └── xgboost_model/         # Position-specific XGBoost model (fallback)
├── rl_decision_layer/
│   ├── predictions/           # Canonical prediction schema
│   ├── candidates/            # Per-position candidate shortlisting
│   ├── environment/           # Squad rules + historical gameweek stepping
│   ├── optimization/          # MILP squad selection, starting XI, scoring, chips
│   ├── ppo/                   # PPO observation/action/env, training, trained models
│   └── tests/                 # Test suite
├── backend/                   # FastAPI backend
├── frontend/                  # Plain HTML/CSS/JS demo UI
├── mini_projects/             # Exploratory RL projects
├── docs/                      # This documentation site (MkDocs)
├── requirements.txt           # Consolidated dependencies
└── README.md
```

Full file-by-file breakdown: `rl_decision_layer/README.md`.
