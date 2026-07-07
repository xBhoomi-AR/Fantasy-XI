# Team Metrics
`https://understat.com/team/Arsenal/2025`
## Overview

This dataset contains Understat's team-level advanced statistics. Unlike player metrics, these statistics describe the overall attacking and defensive performance of an entire team under different contexts such as formations, game states, timings, shot locations, attack speed, and match outcomes.

---

# Common Metrics

The following metrics appear across almost every Team Metrics table.

| Field | Type | Description |
|------|------|-------------|
| Sh | Integer | Total shots taken |
| G | Integer | Goals scored |
| ShA | Integer | Shots allowed (opponent shots) |
| GA | Integer | Goals conceded |
| xG | Float | Expected Goals created |
| xGA | Float | Expected Goals Against |
| xGD | Float | Expected Goal Difference (xG − xGA) |
| xG/Sh | Float | Expected Goals per shot |
| xGA/Sh | Float | Expected Goals conceded per opponent shot |

---

# 1. Situation

Performance grouped by attacking situation.

| Field | Type | Description |
|------|------|-------------|
| Situation | String | Open play, Corner, Set Piece, Direct Free Kick, Penalty |
| Sh | Integer | Shots |
| G | Integer | Goals |
| ShA | Integer | Opponent Shots |
| GA | Integer | Goals Against |
| xG | Float | Expected Goals |
| xGA | Float | Expected Goals Against |
| xGD | Float | Expected Goal Difference |
| xG/Sh | Float | Expected Goals per Shot |
| xGA/Sh | Float | Expected Goals Against per Opponent Shot |

---

# 2. Formation

Statistics grouped by tactical formation.

| Field | Type | Description |
|------|------|-------------|
| Formation | String | Team Formation (4-3-3, 4-2-3-1, etc.) |
| Min | Integer | Minutes played using formation |
| Sh | Integer | Shots |
| G | Integer | Goals |
| ShA | Integer | Opponent Shots |
| GA | Integer | Goals Against |
| xG | Float | Expected Goals |
| xGA | Float | Expected Goals Against |
| xGD | Float | Expected Goal Difference |
| xG90 | Float | Expected Goals per 90 minutes |
| xGA90 | Float | Expected Goals Against per 90 minutes |

---

# 3. Game State

Performance depending on current scoreline.

| Field | Type | Description |
|------|------|-------------|
| Game State | String | Goal Difference (0, +1, >+1, -1, etc.) |
| Min | Integer | Minutes played in this state |
| Sh | Integer | Shots |
| G | Integer | Goals |
| ShA | Integer | Opponent Shots |
| GA | Integer | Goals Against |
| xG | Float | Expected Goals |
| xGA | Float | Expected Goals Against |
| xGD | Float | Expected Goal Difference |
| xG90 | Float | Expected Goals per 90 |
| xGA90 | Float | Expected Goals Against per 90 |

---

# 4. Timing

Statistics grouped by match time intervals.

| Field | Type | Description |
|------|------|-------------|
| Timing | String | Match Interval (1-15,16-30,31-45,46-60,61-75,76+) |
| Sh | Integer | Shots |
| G | Integer | Goals |
| ShA | Integer | Opponent Shots |
| GA | Integer | Goals Against |
| xG | Float | Expected Goals |
| xGA | Float | Expected Goals Against |
| xGD | Float | Expected Goal Difference |
| xG/Sh | Float | Expected Goals per Shot |
| xGA/Sh | Float | Expected Goals Against per Opponent Shot |

---

# 5. Shot Zones

Statistics grouped by shot location.

| Field | Type | Description |
|------|------|-------------|
| Shot Zone | String | Own Goal, Out of Box, Penalty Area, Six-yard Box |
| Sh | Integer | Shots |
| G | Integer | Goals |
| ShA | Integer | Opponent Shots |
| GA | Integer | Goals Against |
| xG | Float | Expected Goals |
| xGA | Float | Expected Goals Against |
| xGD | Float | Expected Goal Difference |
| xG/Sh | Float | Expected Goals per Shot |
| xGA/Sh | Float | Expected Goals Against per Opponent Shot |

---

# 6. Attack Speed

Performance grouped by attack build-up speed.

| Field | Type | Description |
|------|------|-------------|
| Attack Speed | String | Normal, Standard, Slow, Fast |
| Sh | Integer | Shots |
| G | Integer | Goals |
| ShA | Integer | Opponent Shots |
| GA | Integer | Goals Against |
| xG | Float | Expected Goals |
| xGA | Float | Expected Goals Against |
| xGD | Float | Expected Goal Difference |
| xG/Sh | Float | Expected Goals per Shot |
| xGA/Sh | Float | Expected Goals Against per Opponent Shot |

---

# 7. Result

Statistics grouped by shot outcome.

| Field | Type | Description |
|------|------|-------------|
| Result | String | Missed Shot, Blocked Shot, Saved Shot, Goal, Shot on Post |
| Sh | Integer | Shots |
| G | Integer | Goals |
| ShA | Integer | Opponent Shots |
| GA | Integer | Goals Against |
| xG | Float | Expected Goals |
| xGA | Float | Expected Goals Against |
| xGD | Float | Expected Goal Difference |
| xG/Sh | Float | Expected Goals per Shot |
| xGA/Sh | Float | Expected Goals Against per Opponent Shot |

---

# Why We Use Team Metrics

These statistics capture tactical and contextual information that individual player statistics cannot.

Examples include:

- Team attacking style
- Defensive stability
- Formation effectiveness
- Performance while leading or trailing
- Attack speed efficiency
- Shot quality by location
- Time-based performance trends

These metrics will later be merged with player-level statistics and FPL data during feature engineering to build the final Fantasy Premier League prediction dataset.