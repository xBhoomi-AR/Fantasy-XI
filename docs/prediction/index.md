# Prediction Layer

Estimates each player's expected FPL points for an upcoming gameweek, from historical FPL, fixture, team, and Understat data.

Both models optimize for **ranking players well**, not exact scorelines — FPL decisions care about who's likely to outscore whom, not an exact point value.

| Model | Role |
|---|---|
| [BiLSTM](bilstm.md) | Current default |
| [XGBoost](xgboost.md) | Fallback / alternative |

Both write to the same canonical schema (`rl_decision_layer/predictions/interface.py`), so either can drive the rest of the pipeline interchangeably.
