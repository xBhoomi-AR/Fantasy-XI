# Fixtures - Official FPL API

## Endpoint

`https://fantasy.premierleague.com/api/fixtures/`

## Overview

The `fixtures` endpoint contains match-level information for every Premier League fixture. It stores details such as participating teams, match schedule, final scores, fixture difficulty ratings, and match statistics. This endpoint forms the backbone of fixture analysis and is essential for modelling opponent strength, home/away advantage, fixture congestion, rest periods, and future schedule difficulty.

---

## Data Categories

- Fixture Identity
- Match Schedule
- Match Status
- Teams & Scores
- Fixture Difficulty
- Match Statistics

## 1. Fixture Identity

This category uniquely identifies every Premier League fixture and enables linking fixtures with teams, gameweeks, and player performances.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| id | Unique fixture identifier | Keep | Primary key for every match. |
| code | Internal fixture code | Keep | Useful for indexing and cross-referencing. |
| pulse_id | Official Premier League identifier | Keep | Useful for integration with external datasets. |
---

## 2. Match Schedule

These fields describe when the fixture takes place and which Gameweek it belongs to.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| event | Gameweek number | Keep | Essential for time-series modelling. |
| kickoff_time | Match kickoff date and time | Keep | Used for chronological ordering and rest-day calculations. |
| provisional_start_time | Indicates whether kickoff time is provisional | Optional | Useful for scheduling but not a predictive feature. |
| minutes | Match duration | Keep | Useful for identifying abandoned or shortened matches. |
---

## 3. Match Status

These fields indicate the current state of a fixture.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| started | Indicates whether the match has started | Keep | Useful during live updates. |
| finished | Indicates whether the match has finished | Keep | Required to distinguish historical and upcoming fixtures. |
| finished_provisional | Indicates whether the final result is provisional | Optional | Useful for live systems but less important for historical analysis. |
---

## 4. Teams & Match Result

This category records the participating teams and the final outcome of the fixture.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| team_h | Home team ID | Keep | Required for home/away analysis. |
| team_a | Away team ID | Keep | Required for opponent modelling. |
| team_h_score | Goals scored by the home team | Keep | Useful for historical analysis and model evaluation. |
| team_a_score | Goals scored by the away team | Keep | Useful for historical analysis and model evaluation. |
---

## 5. Fixture Difficulty

The Official FPL API assigns fixture difficulty ratings separately for both teams. These values are widely used in Fantasy Premier League strategy and provide a direct estimate of match difficulty.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| team_h_difficulty | Fixture difficulty for the home team | Keep | Strong predictive feature for expected player performance. |
| team_a_difficulty | Fixture difficulty for the away team | Keep | Strong predictive feature for expected player performance. |
---

## 6. Match Statistics

The `stats` field contains detailed player-level statistics recorded during the fixture. Rather than storing aggregated team information, it records individual player contributions such as goals, assists, cards, saves, bonus points, BPS, and defensive contributions.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| stats | Collection of all player-level match statistics | Keep | Core source for constructing player Gameweek records and historical performance data. |
---

