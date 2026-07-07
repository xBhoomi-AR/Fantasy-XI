# Player Metrics - Understat
`https://understat.com/player/647`
## Overview

The Player Metrics dataset provides advanced player-level analytics from Understat. Unlike the Official FPL API, Understat focuses on expected statistics (xG, xA) and contextual attacking performance. Player metrics are organized into five analytical views.

---

# 1. Season Statistics

## Description

Provides season-wise player performance across different clubs and competitions.

### Available Fields

| Field | Description | Decision | Reason |
|--------|-------------|----------|--------|
| Season | Football season | Keep | Time-series indexing |
| Team | Club represented | Keep | Player-team mapping |
| Apps | Appearances | Keep | Player availability |
| Min | Minutes played | Keep | Expected minutes feature |
| G | Goals | Keep | Attacking output |
| A | Assists | Keep | Creative output |
| Sh90 | Shots per 90 minutes | Keep | Shooting frequency |
| KP90 | Key passes per 90 minutes | Keep | Creativity metric |
| xG | Expected Goals | Keep | Core predictive feature |
| xA | Expected Assists | Keep | Core predictive feature |
| xG90 | Expected Goals per 90 | Keep | Efficiency metric |
| xA90 | Expected Assists per 90 | Keep | Efficiency metric |

---

# 2. Position Statistics

## Description

Provides player performance grouped by playing position.

### Available Fields

| Field | Description | Decision | Reason |
|--------|-------------|----------|--------|
| Position | Playing position | Keep | Position-based analysis |
| Apps | Appearances | Keep | Position usage |
| Min | Minutes played | Keep | Playing time |
| G | Goals | Keep | Position-wise scoring |
| A | Assists | Keep | Position-wise creativity |
| Sh90 | Shots per 90 | Keep | Shooting frequency |
| KP90 | Key passes per 90 | Keep | Creativity |
| xG | Expected Goals | Keep | Position-wise attacking quality |
| xA | Expected Assists | Keep | Position-wise creative quality |
| xG90 | Expected Goals per 90 | Keep | Efficiency |
| xA90 | Expected Assists per 90 | Keep | Efficiency |

---

# 3. Situation Statistics

## Description

Provides player performance under different match situations.

### Available Fields

| Field | Description | Decision | Reason |
|--------|-------------|----------|--------|
| Situation | Match situation category | Keep | Contextual analysis |
| Sh | Total shots | Keep | Shot volume |
| G | Goals | Keep | Finishing output |
| KP | Key passes | Keep | Chance creation |
| A | Assists | Keep | Creative output |
| xG | Expected Goals | Keep | Chance quality |
| xA | Expected Assists | Keep | Expected creativity |
| xG90 | Expected Goals per 90 | Keep | Efficiency |
| xA90 | Expected Assists per 90 | Keep | Efficiency |
| xG/Sh | Expected Goals per Shot | Keep | Shot quality |
| xA/KP | Expected Assists per Key Pass | Keep | Pass quality |

### Available Situations

- Open Play
- From Corner
- Set Piece
- Direct Free Kick
- Penalty

---

# 4. Shot Zone Statistics

## Description

Provides player performance based on shot location.

### Available Fields

| Field | Description | Decision | Reason |
|--------|-------------|----------|--------|
| Shot Zones | Shot location category | Keep | Spatial analysis |
| Sh | Total shots | Keep | Shot volume |
| G | Goals | Keep | Finishing output |
| KP | Key passes | Keep | Chance creation |
| A | Assists | Keep | Creativity |
| xG | Expected Goals | Keep | Chance quality |
| xA | Expected Assists | Keep | Creativity quality |
| xG/Sh | Expected Goals per Shot | Keep | Shot efficiency |
| xA/KP | Expected Assists per Key Pass | Keep | Passing efficiency |

### Available Shot Zones

- Out of Box
- Penalty Area
- Six-yard Box

---

# 5. Shot Type Statistics

## Description

Provides player performance grouped by shot type.

### Available Fields

| Field | Description | Decision | Reason |
|--------|-------------|----------|--------|
| Shot Types | Shot category | Keep | Finishing style analysis |
| Sh | Total shots | Keep | Shot volume |
| G | Goals | Keep | Finishing output |
| KP | Key passes | Keep | Chance creation |
| A | Assists | Keep | Creative output |
| xG | Expected Goals | Keep | Chance quality |
| xA | Expected Assists | Keep | Creativity quality |
| xG/Sh | Expected Goals per Shot | Keep | Shot efficiency |
| xA/KP | Expected Assists per Key Pass | Keep | Passing efficiency |

### Available Shot Types

- Right Foot
- Left Foot
- Head
- Other Body Part

---

# Importance for FantasyXI

Understat complements the Official FPL API by providing advanced expected metrics and contextual attacking statistics that are not directly available in the FPL dataset. These features will be evaluated during Feature Engineering for their contribution toward Fantasy Premier League point prediction.