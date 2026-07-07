# Player Shooting Statistics
`https://fbref.com/en/comps/9/shooting/Premier-League-Stats#all_stats_shooting`
## Purpose

The Player Shooting Statistics dataset contains detailed shooting metrics for every Premier League player.

This dataset extends the basic goal statistics available in the FPL API by providing information about shot volume, shooting accuracy, finishing efficiency, and penalty statistics.

These metrics are valuable for estimating attacking potential and expected fantasy returns.

---

## Table Description

One row represents:

> One Player × One Season

Example:

Player = Bukayo Saka

Season = 2025/26

---

## Available Columns

| Column | Description | Usage in FantasyXI |
|----------|-------------|--------------------|
| Player | Player name | Player identification |
| Nation | Nationality | Metadata |
| Pos | Playing position | Position validation |
| Squad | Club | Team mapping |
| Age | Player age | Metadata |
| Born | Birth year | Metadata |
| 90s | Full-match equivalents played | Per-90 normalization |
| Gls | Goals scored | Goal prediction |
| Sh | Total shots attempted | Shooting volume |
| SoT | Shots on target | Shot accuracy |
| SoT% | Percentage of shots on target | Finishing consistency |
| Sh/90 | Shots per 90 minutes | Attacking involvement |
| SoT/90 | Shots on target per 90 | Threat creation |
| G/Sh | Goals per shot | Finishing efficiency |
| G/SoT | Goals per shot on target | Clinical finishing |
| PK | Penalty goals | Penalty taker identification |
| PKatt | Penalty attempts | Penalty responsibility |
| Matches | Link to detailed match logs | Navigation only |

---

## Important FantasyXI Features

Primary features extracted:

- Total Shots
- Shots on Target
- Shot Accuracy
- Shots per 90
- Shots on Target per 90
- Goals per Shot
- Goals per Shot on Target
- Penalty Goals
- Penalty Attempts

---

## Data Characteristics

Granularity:

Player × Season

Primary Key:

(Player, Season)

Source:

FBRef → Premier League → Player Shooting

Update Frequency:

Updated after every Premier League match.

---

## Usage in FantasyXI

This dataset is used for:

- Goal prediction
- Expected attacking involvement
- Finishing efficiency estimation
- Captaincy evaluation
- Differential player discovery
- Feature Engineering
- Cross-validation with Understat xG statistics