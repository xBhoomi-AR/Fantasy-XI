# FBRef Data Source
`https://fbref.com/en/comps/9/Premier-League-Stats`
## Overview

FBRef provides advanced football statistics that complement the official Fantasy Premier League (FPL) API and Understat.

Unlike the FPL API, which focuses on fantasy-specific information, FBRef offers detailed player and goalkeeper performance statistics, including shooting efficiency, playing time, disciplinary records, and goalkeeping metrics.

Within the FantasyXI project, FBRef serves as one of the three primary external data sources used for feature engineering and machine learning.

---

# Purpose

The FBRef dataset is used to:

- Collect advanced player statistics
- Gather detailed goalkeeping metrics
- Measure playing time and availability
- Evaluate shooting efficiency
- Capture disciplinary and defensive actions
- Generate engineered features for FantasyXI prediction models

---

# Data Coverage

The current implementation includes the following datasets:

| File | Description |
|------|-------------|
| fbref_standard_stats.md | General player statistics including appearances, goals, assists and playing time |
| fbref_shooting.md | Shooting performance, shot accuracy and finishing statistics |
| fbref_playing_time.md | Minutes played, starts, substitutions and player availability |
| fbref_miscellaneous.md | Cards, fouls, interceptions, tackles and other miscellaneous statistics |
| fbref_goalkeeping.md | Goalkeeper-specific statistics including saves, clean sheets and penalty performance |

---

# Relationship with Other Data Sources

FantasyXI combines three primary datasets:

1. Official FPL API
2. Understat
3. FBRef

Each source contributes different information.

- Official FPL API provides fantasy-related information such as prices, ownership, fixtures and fantasy points.
- Understat provides expected goals (xG), expected assists (xA) and other expected performance metrics.
- FBRef provides advanced player performance statistics and detailed goalkeeper metrics.

After collection, these datasets are merged into a unified relational database before preprocessing and feature engineering.

---

# Data Collection

Data is collected season-wise for Premier League players.

The collection process consists of:

1. Retrieve player statistics from FBRef.
2. Standardize player and club names.
3. Match players with FPL API identifiers.
4. Merge with Understat expected metrics.
5. Store the integrated dataset inside the FantasyXI database.

---

# Usage in FantasyXI

FBRef data contributes to several stages of the pipeline:

- Feature Engineering
- Expected Minutes Prediction
- Goalkeeper Evaluation
- Shooting Efficiency Analysis
- Defensive Contribution Analysis
- Discipline Assessment
- Player Availability Modelling
- Machine Learning Dataset Construction

---

# Notes

FBRef does not provide fantasy points directly.

Instead, it supplies detailed football performance statistics that significantly enhance feature engineering and improve predictive performance when combined with FPL API and Understat data.