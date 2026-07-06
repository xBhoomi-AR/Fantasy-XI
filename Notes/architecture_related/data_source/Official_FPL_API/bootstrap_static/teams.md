# Teams - Official FPL API

## Endpoint

`https://fantasy.premierleague.com/api/bootstrap-static/`

## Overview

The `teams` section contains team-level information for all Premier League clubs. It provides metadata, league performance, and team strength ratings that can be used to model opponent quality, fixture difficulty, attacking strength, and defensive strength. These attributes play a crucial role in predicting player performance, since a player's expected fantasy points are highly influenced by the strength of both their own team and their opponents.

---

## Data Categories

- Team Identity
- League Performance
- Team Strength Ratings
- Home & Away Strength
- Team Availability

## 1. Team Identity

The Team Identity category contains the basic information required to uniquely identify each Premier League club and establish relationships with players and fixtures.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| id | Unique FPL team identifier | Keep | Primary key for joining player and fixture datasets. |
| code | Internal FPL team code | Keep | Useful for cross-referencing FPL data. |
| name | Full team name | Keep | Display and identification purposes. |
| short_name | Team abbreviation | Keep | Useful for dashboards and visualizations. |
| pulse_id | Official Premier League identifier | Keep | Helpful for integrating external football datasets. |
| link_url | Team webpage slug | Discard | No predictive or analytical value. |

## 2. League Performance

These fields summarize the team's overall league performance during the current season.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| played | Matches played | Keep | Required for season progress analysis. |
| win | Matches won | Keep | Measures overall team success. |
| draw | Matches drawn | Keep | Useful for team performance analysis. |
| loss | Matches lost | Keep | Indicates overall team weakness. |
| points | League points | Keep | Strong indicator of team quality. |
| position | Current league position | Keep | Useful indicator of overall team strength. |
| form | Recent team form | Keep | Important short-term performance metric (when available). |

## 3. Team Strength Ratings

The FPL API provides internally computed strength ratings that estimate the overall quality of each team. These ratings are valuable features for modelling fixture difficulty and opponent strength.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| strength | Overall team strength | Keep | General indicator of team quality. |
| strength_attack_home | Home attacking strength | Keep | Useful when predicting attacking returns in home fixtures. |
| strength_attack_away | Away attacking strength | Keep | Useful for away fixture analysis. |
| strength_defence_home | Home defensive strength | Keep | Helps estimate clean sheet probability. |
| strength_defence_away | Away defensive strength | Keep | Important for evaluating away defensive performance. |
| strength_overall_home | Overall home strength | Keep | Captures overall team quality in home matches. |
| strength_overall_away | Overall away strength | Keep | Captures overall team quality in away matches. |

## 4. Team Availability

These fields indicate whether a team is currently active within the Fantasy Premier League ecosystem.

| Field | Description | Decision | Reason |
|------|-------------|----------|--------|
| unavailable | Indicates whether the team is unavailable | Keep | Prevents invalid team selection if applicable. |
| team_division | Team division identifier | Discard | Not relevant for Premier League analysis as all teams belong to the same competition. |