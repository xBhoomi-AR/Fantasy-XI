# Future Scope

- **Live FPL data integration** — the system currently runs entirely on historical, already-collected data; there is no live price/fixture/gameweek fetching yet.
- **A more adaptive decision layer** — PPO's strategic action has not yet been shown to vary its choices situationally; further training and evaluation is needed to demonstrate genuine adaptiveness.
- **Forward-looking optimization** — the MILP is myopic (optimizes one gameweek at a time and always spends its full budget), which can cause late-season infeasibility; a budget-aware, multi-gameweek-lookahead MILP is a natural next step.
- **A validated baseline comparison** — see [Results & Evaluation](evaluation/results.md).
