# Understat Data Source

## Overview

Understat provides advanced football analytics that are not available in the official FPL API.

It focuses on expected-goal based statistics and detailed player/team performance metrics.

These datasets will be combined with the Official FPL API and FBRef to construct the final FantasyXI machine learning dataset.

---

## Purpose

Understat is primarily used to obtain:

- Expected Goals (xG)
- Expected Assists (xA)
- Shot-based statistics
- Key passes
- Team attacking/defensive metrics
- Player match-level performance
- Advanced football analytics

---

## Files

| File | Purpose |
|------|---------|
| player_metrics.md | Season-level player statistics and advanced attacking metrics |
| team_metrics.md | Season-level team attacking and defensive statistics |
| player_match_history.md | Match-by-match player performance statistics |
| shot_maps.md | Planned spatial shot-event dataset (currently pending implementation) |

---

## Key Metrics

- xG
- xA
- xG90
- xA90
- Shots per 90
- Key Passes per 90
- Team xGD
- Team xGA
- Shot Location Categories
- Shot Situations
- Attack Speed
- Match State

---

## Role in FantasyXI

Understat provides advanced performance indicators that are unavailable in the Official FPL API.

These metrics will be merged with FPL player, fixture and historical data to improve feature engineering and model performance.

---

## Status

Documentation Complete.

Shot map raw-event extraction is pending and will be implemented during the data collection stage.