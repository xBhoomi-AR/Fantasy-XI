# Element Summary - Official FPL API

## Endpoint

`https://fantasy.premierleague.com/api/element-summary/{player_id}/`

## Overview

The `element-summary` endpoint provides detailed historical and upcoming information for an individual player. Unlike the `elements` endpoint, which stores season-level aggregated statistics, this endpoint contains Gameweek-wise records of player performance. It serves as the primary source for constructing player time-series data, making it one of the most important datasets for forecasting Fantasy Premier League points.

---

## Data Categories

- Player Identity
- Match Information
- Fantasy Performance
- Match Statistics
- Advanced Performance Statistics
- Market Information

## 1. Player Identity

This category uniquely identifies the player and links each historical record to the player database.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| element | Player ID | Keep | Foreign key linking historical records to the Players table. |
---

## 2. Match Information

This category contains fixture-specific information describing the match in which the player participated.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| fixture | Fixture ID | Keep | Links player performance to the Fixtures table. |
| opponent_team | Opponent team ID | Keep | Required for opponent strength analysis. |
| round | Gameweek number | Keep | Essential for chronological time-series construction. |
| kickoff_time | Match kickoff date and time | Keep | Used for temporal ordering and rest-day calculations. |
| was_home | Indicates whether the player played at home | Keep | Important feature for home/away performance modelling. |
| modified | Indicates whether fixture data was modified | Optional | Useful for data validation but not a predictive feature. |
| team_h_score | Home team goals | Keep | Useful for historical match analysis. |
| team_a_score | Away team goals | Keep | Useful for historical match analysis. |
---

## 3. Fantasy Performance

This category contains Fantasy Premier League scoring metrics for a single Gameweek.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| total_points | Fantasy points scored in the Gameweek | Keep | Primary prediction target for supervised learning. |
| value | Player price during the Gameweek | Keep | Required for value-based features and budget optimization. |
---

## 4. Match Statistics

These statistics describe the player's on-field contribution during a single fixture.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| minutes | Minutes played | Keep | Strong predictor of future fantasy returns. |
| starts | Match started indicator | Keep | Reflects expected playing time. |
| goals_scored | Goals scored | Keep | Primary attacking metric. |
| assists | Assists | Keep | Measures creative contribution. |
| clean_sheets | Clean sheets | Keep | Important for goalkeepers and defenders. |
| goals_conceded | Goals conceded | Keep | Defensive performance indicator. |
| own_goals | Own goals | Optional | Rare event with limited predictive value. |
| penalties_saved | Penalties saved | Keep | Goalkeeper-specific performance metric. |
| penalties_missed | Penalties missed | Keep | Useful attacking statistic. |
| yellow_cards | Yellow cards | Keep | Measures disciplinary risk. |
| red_cards | Red cards | Keep | Indicates suspension risk. |
| saves | Goalkeeper saves | Keep | Major fantasy scoring source for goalkeepers. |
| bonus | Bonus points | Keep | Captures overall match contribution. |
| bps | Bonus Point System score | Keep | Strong indicator of player influence. |
---

## 5. Advanced Performance Statistics

These metrics provide a more detailed analytical description of player performance during the fixture.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| influence | FPL Influence Index | Keep | Measures overall impact during the match. |
| creativity | FPL Creativity Index | Keep | Measures chance creation. |
| threat | FPL Threat Index | Keep | Measures goal-scoring threat. |
| ict_index | Influence-Creativity-Threat Index | Keep | Composite attacking performance metric. |
| expected_goals | Expected Goals (xG) | Keep | Better indicator of finishing quality than goals alone. |
| expected_assists | Expected Assists (xA) | Keep | Measures expected chance creation. |
| expected_goal_involvements | Expected Goal Involvement (xGI) | Keep | Combined attacking contribution metric. |
| expected_goals_conceded | Expected Goals Conceded (xGC) | Keep | Defensive performance indicator. |
| clearances_blocks_interceptions | Defensive actions | Keep | Important defensive metric. |
| recoveries | Ball recoveries | Keep | Defensive work rate indicator. |
| tackles | Successful tackles | Keep | Defensive contribution metric. |
| defensive_contribution | Overall defensive contribution | Keep | Composite defensive performance metric. |
---

## 6. Market Information

These fields describe Fantasy Premier League market behaviour during the corresponding Gameweek.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| selected | Number of managers owning the player | Keep | Measures ownership dynamics. |
| transfers_in | Transfers into the player during the Gameweek | Keep | Captures market sentiment. |
| transfers_out | Transfers out of the player during the Gameweek | Keep | Indicates declining popularity or injury concerns. |
| transfers_balance | Net transfers (Transfers In − Transfers Out) | Keep | Useful for modelling ownership momentum and market trends. |
---
