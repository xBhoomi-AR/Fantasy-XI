# MILP — Squad Optimization

A mixed-integer linear program (PuLP) that turns PPO's strategy into a legal squad, maximizing total predicted points.

## Squad constraints

- Exactly 15 players: 2 GK, 5 DEF, 5 MID, 3 FWD.
- Max 3 players from any one club.
- Total cost within budget (bank + current squad value).
- Free transfers roll over (capped at 5); extra transfers cost 4 points each — only taken when worth it.

## Starting XI

A second MILP picks 11 starters (1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD) and a captain/vice from the highest predicted scorers among starters.

> **TODO:** Add a short worked example (one gameweek's input → output) here.
