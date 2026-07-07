# Player Standard Statistics
`https://fbref.com/en/comps/9/stats/Premier-League-Stats#all_stats_standard`
## Purpose

The Player Standard Statistics table contains the fundamental season-level statistics for every Premier League player.

Unlike the Official FPL API, FBRef provides additional football-specific statistics such as detailed playing time, disciplinary records, non-penalty goals, and per-90 metrics.

These statistics act as the base layer for many engineered features used in FantasyXI.

---

## Table Description

One row represents:

> One Player × One Season

Example:

Player = Declan Rice

Season = 2025/26

---

## Available Columns

| Column | Description | Usage in FantasyXI |
|----------|-------------|--------------------|
| Player | Player name | Player identification |
| Nation | Nationality | Metadata |
| Pos | Playing position | Position validation |
| Squad | Club | Team mapping |
| Age | Player age | Age-based features |
| Born | Birth year | Metadata |
| MP | Matches played | Availability |
| Starts | Matches started | Starting probability |
| Min | Minutes played | Expected minutes |
| 90s | Equivalent full matches | Per-90 calculations |
| Gls | Goals scored | Offensive output |
| Ast | Assists | Creativity |
| G+A | Goals + Assists | Total attacking contribution |
| G-PK | Non-penalty goals | Better attacking metric |
| PK | Penalty goals | Penalty analysis |
| PKatt | Penalty attempts | Penalty responsibility |
| CrdY | Yellow Cards | Discipline |
| CrdR | Red Cards | Discipline |
| Gls/90 | Goals per 90 | Rate statistic |
| Ast/90 | Assists per 90 | Rate statistic |
| G+A/90 | Goal Contributions per 90 | Attacking efficiency |
| G-PK/90 | Non-Penalty Goals per 90 | Goal scoring efficiency |
| G+A-PK/90 | Non-Penalty Goal Contributions per 90 | Advanced attacking metric |
| Matches | Link to detailed match logs | Navigation only |

---

## Important FantasyXI Features

Primary features extracted:

- Minutes Played
- Starts
- Goals
- Assists
- Goal Contributions
- Non-Penalty Goals
- Yellow Cards
- Red Cards
- Goals per 90
- Assists per 90
- Goal Contributions per 90

---

## Data Characteristics

Granularity:

Player × Season

Primary Key:

(Player, Season)

Source:

FBRef → Premier League → Standard Stats

Update Frequency:

After every Premier League match.

---

## Usage in FantasyXI

This dataset is primarily used for:

- Expected Minutes estimation
- Player availability
- Goal prediction
- Assist prediction
- Historical player profiling
- Feature Engineering
- Dataset validation with FPL API