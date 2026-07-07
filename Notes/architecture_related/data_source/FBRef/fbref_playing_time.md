# FBRef - Player Playing Time
`https://fbref.com/en/comps/9/playingtime/Premier-League-Stats#all_stats_playing_time`
## Purpose

Contains player availability, minutes played, starting frequency, substitution patterns, and team performance while a player is on the pitch.

This dataset is used to estimate:

- Expected minutes
- Starting probability
- Substitution probability
- Player reliability
- Rotation risk
- Team dependency
- Fantasy captaincy safety
- Availability features for ML models

---

# Source

FBRef

Competition:
Premier League

Table:
Player Playing Time

---

# Primary Key

player_id + season

---

# Columns

| Column | Type | Description |
|---------|------|-------------|
| player | string | Player name |
| nation | string | Nationality |
| pos | string | Playing position |
| squad | string | Club |
| age | integer | Player age |
| born | integer | Birth year |
| mp | integer | Matches played |
| min | integer | Total minutes played |
| mn_per_mp | float | Average minutes per appearance |
| min_percent | float | Percentage of available minutes played |
| nineties | float | Equivalent full 90-minute matches |
| starts | integer | Matches started |
| mn_per_start | float | Average minutes per start |
| complete_matches | integer | Full matches completed |
| subs | integer | Matches entered as substitute |
| mn_per_sub | float | Average substitute minutes |
| unused_sub | integer | Times named on bench but unused |
| ppm | float | Points per match while on pitch |
| goals_for_on | integer | Team goals scored while player on field |
| goals_against_on | integer | Team goals conceded while player on field |
| plus_minus | integer | Goal difference while on field |
| plus_minus_per90 | float | Goal difference per 90 minutes |
| on_off | float | Team goal difference impact when player is on vs off the pitch |
| matches_link | string | FBRef match history page |

---

# Feature Importance

## Minutes Prediction

- min
- mn_per_mp
- mn_per_start
- starts
- subs

---

## Rotation Detection

- starts
- subs
- unused_sub
- complete_matches

---

## Availability Score

- min_percent
- nineties
- mp

---

## Team Impact

- ppm
- goals_for_on
- goals_against_on
- plus_minus
- plus_minus_per90
- on_off

---

# FantasyXI Usage

Used for:

- Expected Minutes Model
- Rotation Prediction
- Starting XI Prediction
- Injury/Bench Risk
- Captain Reliability
- Player Availability Score
- Feature Engineering

---

# Notes

This table complements the FPL API availability data.

Unlike FPL bootstrap data, FBRef provides detailed minute distributions, substitution behaviour, and on/off team impact, making it valuable for predictive modelling.