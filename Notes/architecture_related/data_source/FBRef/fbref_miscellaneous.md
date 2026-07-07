# FBRef - Player Miscellaneous Statistics
`https://fbref.com/en/comps/9/misc/Premier-League-Stats#all_stats_misc`
## Purpose

Contains miscellaneous player performance metrics that are not covered by standard attacking or playing time statistics.

These metrics help evaluate player discipline, defensive work rate, ball-winning ability, aerial contribution, and overall physical involvement.

---

# Source

FBRef

Competition:
Premier League

Table:
Player Miscellaneous Stats

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
| nineties | float | Equivalent full 90-minute matches |
| yellow_cards | integer | Yellow cards received |
| red_cards | integer | Red cards received |
| second_yellow_red | integer | Red cards resulting from second yellow |
| fouls_committed | integer | Fouls committed |
| fouls_drawn | integer | Fouls won from opponents |
| offsides | integer | Times caught offside |
| crosses | integer | Crosses attempted |
| interceptions | integer | Interceptions made |
| tackles_won | integer | Tackles won |
| penalties_won | integer | Penalties won |
| penalties_conceded | integer | Penalties conceded |
| own_goals | integer | Own goals scored |
| matches_link | string | FBRef match history page |

---

# Feature Importance

## Discipline

- yellow_cards
- red_cards
- second_yellow_red

---

## Defensive Contribution

- interceptions
- tackles_won

---

## Attacking Contribution

- fouls_drawn
- crosses

---

## Defensive Risk

- fouls_committed
- penalties_conceded
- own_goals

---

## Position Behaviour

- offsides
- crosses
- interceptions
- tackles_won

---

# FantasyXI Usage

Used for:

- Discipline Score
- Defensive Activity Score
- Player Aggression Index
- Position Behaviour Features
- Bonus Point Prediction
- Feature Engineering

---

# Notes

Unlike the FPL API, this dataset provides detailed disciplinary and defensive event statistics that improve player profiling and feature engineering for machine learning models.