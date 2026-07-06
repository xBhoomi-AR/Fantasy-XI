# Events (Gameweeks) - Official FPL API

## Endpoint

`https://fantasy.premierleague.com/api/bootstrap-static/`

## Overview

The `events` section represents individual Fantasy Premier League Gameweeks. It contains scheduling information, deadlines, overall Gameweek statistics, chip usage, transfers, and other metadata describing each Gameweek. This endpoint is essential for organising historical data chronologically and building time-series datasets for player performance prediction.

---

## Data Categories

- Gameweek Identity
- Schedule & Timeline
- Gameweek Status
- Overall Gameweek Statistics
- Chip Usage
- Popularity & Transfers
- Top Performers

## 1. Gameweek Identity

This category uniquely identifies each Fantasy Premier League Gameweek.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| id | Unique Gameweek identifier | Keep | Primary key for Gameweek records. |
| name | Official Gameweek name | Keep | Useful for display and reporting. |
---

## 2. Schedule & Timeline

These fields describe when the Gameweek begins and other scheduling information.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| deadline_time | Official Gameweek deadline | Keep | Essential for transfer planning and chronological ordering. |
| deadline_time_epoch | Deadline in Unix timestamp format | Optional | Useful for efficient time computations. |
| deadline_time_game_offset | Deadline offset | Optional | Rarely required for analysis. |
| release_time | Official release time | Optional | Mainly metadata. |
---

## 3. Gameweek Status

These fields describe the current lifecycle of a Gameweek.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| finished | Indicates whether the Gameweek has finished | Keep | Required for distinguishing historical and future Gameweeks. |
| data_checked | Indicates whether FPL has verified the data | Keep | Ensures reliable historical records. |
| is_previous | Indicates previous Gameweek | Optional | Useful for live systems. |
| is_current | Indicates current active Gameweek | Keep | Required for live prediction systems. |
| is_next | Indicates upcoming Gameweek | Keep | Useful for forecasting next Gameweek. |
| released | Indicates whether Gameweek data has been released | Keep | Data availability check. |
| can_enter | Whether managers can enter the Gameweek | Discard | Website functionality only. |
| can_manage | Whether managers can manage their team | Discard | Website functionality only. |
| cup_leagues_created | Cup competitions created | Discard | Not relevant to FantasyXI. |
| h2h_ko_matches_created | Head-to-head knockout creation status | Discard | Not relevant to prediction. |
---

## 4. Overall Gameweek Statistics

These fields summarize the overall Fantasy Premier League activity during a Gameweek.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| average_entry_score | Average score across all managers | Keep | Useful benchmark for model evaluation. |
| highest_score | Highest score achieved during the Gameweek | Optional | Mainly descriptive statistic. |
| ranked_count | Number of active ranked managers | Optional | Useful for popularity analysis. |
---

## 4. Overall Gameweek Statistics

These fields summarize the overall Fantasy Premier League activity during a Gameweek.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| average_entry_score | Average score across all managers | Keep | Useful benchmark for model evaluation. |
| highest_score | Highest score achieved during the Gameweek | Optional | Mainly descriptive statistic. |
| ranked_count | Number of active ranked managers | Optional | Useful for popularity analysis. |
---

## 5. Chip Usage

This category records how Fantasy Premier League chips were used during a Gameweek.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| chip_plays | Number of Bench Boost, Triple Captain, Wildcard, Free Hit, etc. used | Keep | Useful for analysing manager behaviour and market dynamics. |
---

## 6. Popularity & Transfers

These fields capture overall manager behaviour during the Gameweek.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| transfers_made | Total transfers made during the Gameweek | Keep | Measures overall market activity. |
| most_selected | Most owned player | Optional | Useful for popularity analysis. |
| most_transferred_in | Most transferred-in player | Keep | Indicates market trends. |
| most_captained | Most captained player | Keep | Useful for captaincy behaviour analysis. |
| most_vice_captained | Most vice-captained player | Optional | Secondary manager behaviour metric. |
---

## 7. Top Performers

These fields identify the best-performing players during a Gameweek.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| top_element | Player ID of highest-scoring player | Keep | Useful for analysing exceptional performances. |
| top_element_info | Additional information about the top player | Keep | Stores points and related metadata. |
| highest_scoring_entry | Manager ID with highest score | Discard | Not relevant to FantasyXI. |
| overrides | Internal FPL configuration overrides | Discard | Website configuration only; no predictive value. |
---

