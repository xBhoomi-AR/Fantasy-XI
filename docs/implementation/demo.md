# Frontend & Demo

A local, single-process demo visualizes the pipeline as a clickable sequential season:

```
Start Season → Gameweek 1 → Next Gameweek → Gameweek 2 → … → Season Complete
```

- Squad, pitch-formation starting XI, captain/vice.
- Transfers, chips, bank/free transfers.

**No prediction, RL, or optimization logic lives in the frontend** — it's a view onto the same `PPOEnv`/`SeasonState` machinery the CLI tools use.

> **TODO:** Insert demo screenshot(s) here.

## How it's served

A FastAPI backend (`backend/app.py`, `backend/rl_bridge.py`) wraps the sequential pipeline; a plain HTML/CSS/JS frontend (no build step) renders whatever it returns.

See [Setup](../setup/installation.md) to run it yourself.
