# FBRef - Player Goalkeeping Statistics
`https://fbref.com/en/comps/9/keepers/Premier-League-Stats#all_stats_keeper`
## Purpose

Contains goalkeeper-specific statistics including shot stopping, clean sheets, goals conceded, save percentage, and penalty performance.

This dataset is used to evaluate goalkeeper quality for FantasyXI and engineer goalkeeper-specific features that are unavailable in the FPL API.

---

# Source

FBRef

Competition:
Premier League

Table:
Player Goalkeeping Stats

---

# Primary Key

player_id + season

---

# Columns

| Column | Type | Description |
|---------|------|-------------|
| player | string | Goalkeeper name |
| nation | string | Nationality |
| pos | string | Position (GK) |
| squad | string | Club |
| age | integer | Player age |
| born | integer | Birth year |
| mp | integer | Matches played |
| starts | integer | Matches started |
| min | integer | Minutes played |
| nineties | float | Equivalent full 90-minute matches |
| goals_against | integer | Goals conceded |
| goals_against_per90 | float | Goals conceded per 90 minutes |
| shots_on_target_against | integer | Shots on target faced |
| saves | integer | Total saves made |
| save_percentage | float | Save percentage |
| wins | integer | Wins |
| draws | integer | Draws |
| losses | integer | Losses |
| clean_sheets | integer | Clean sheets |
| clean_sheet_percentage | float | Percentage of matches with clean sheet |
| penalties_faced | integer | Penalties faced |
| penalties_allowed | integer | Penalties conceded |
| penalties_saved | integer | Penalties saved |
| penalties_missed | integer | Penalties missed by opponent |
| penalty_save_percentage | float | Penalty save percentage |
| matches_link | string | FBRef match history page |

---

# Feature Importance

## Shot Stopping

- saves
- save_percentage
- shots_on_target_against

---

## Defensive Strength

- goals_against
- goals_against_per90
- clean_sheets
- clean_sheet_percentage

---

## Match Reliability

- mp
- starts
- min
- nineties

---

## Team Performance

- wins
- draws
- losses

---

## Penalty Ability

- penalties_faced
- penalties_saved
- penalty_save_percentage

---

# FantasyXI Usage

Used for:

- Goalkeeper Rating
- Save Prediction
- Clean Sheet Prediction
- Expected FPL Goalkeeper Points
- Goalkeeper Selection Model
- Feature Engineering

---

# Notes

This dataset is only applicable to goalkeepers.

The FPL API provides fantasy points and basic statistics, while FBRef supplies detailed goalkeeping performance metrics required for predictive modelling.