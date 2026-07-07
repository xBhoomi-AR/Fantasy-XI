# Player Match History

## Overview

This dataset contains match-by-match performance statistics for an individual player across a season.

Each row represents **one player's performance in one match**.

Unlike `player_metrics.md`, which stores season-level aggregate statistics, this dataset preserves chronological match history and is the primary source for calculating rolling form, momentum, and other time-dependent features.

---

# Dataset Structure

**Granularity**

- One Row = One Player × One Match

Example:

| Player | Match |
|---------|-------|
| Declan Rice | Arsenal vs Chelsea |
| Declan Rice | Arsenal vs Fulham |
| Bukayo Saka | Arsenal vs Chelsea |

---

# Fields

| Field | Type | Description |
|------|------|-------------|
| Date | Date | Match Date |
| Home | String | Home Team |
| Score | String | Final Match Score |
| Away | String | Away Team |
| Pos | String | Player Position in that Match |
| Min | Integer | Minutes Played |
| Sh | Integer | Total Shots |
| G | Integer | Goals Scored |
| KP | Integer | Key Passes |
| A | Integer | Assists |
| xG | Float | Expected Goals |
| xA | Float | Expected Assists |

---

# Example Record

| Date | Home | Score | Away | Pos | Min | Sh | G | KP | A | xG | xA |
|------|------|-------|------|-----|-----|----|---|----|---|----|----|
| 2026-05-18 | Arsenal | 1-0 | Burnley | MC | 90 | 0 | 0 | 0 | 0 | 0.00 | 0.00 |

---

# Why FantasyXI Uses This Dataset

This dataset is one of the most important inputs for feature engineering.

It allows the model to compute:

- Last N Match Form
- Rolling xG
- Rolling xA
- Rolling Goals
- Rolling Assists
- Rolling Minutes
- Rolling Shots
- Rolling Key Passes
- Form Momentum
- Exponential Moving Average (EMA)
- Consistency Score
- Recent Performance Trend
- Player Availability

Unlike season totals, this dataset preserves chronological order, making it suitable for time-series feature engineering and sequential deep learning models.

---

# Relationship with Other Datasets

This dataset is linked with:

- Official FPL API (player id, fixture id)
- Understat Player Metrics
- Team Match History
- Match Metrics
- Final SQL Database

These relationships allow FantasyXI to combine historical player performance with fixture difficulty, opponent strength, and FPL scoring data.