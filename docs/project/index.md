# Project Overview

FantasyXI turns historical FPL data into a gameweek-by-gameweek squad recommendation.

- **Predict** each player's expected points (BiLSTM / XGBoost).
- **Decide** a strategy for the gameweek — transfer aggressiveness, budget, chips (PPO).
- **Optimize** that strategy into an actual legal squad and starting XI (MILP).
- **Carry state forward** — each gameweek builds on the previous one's real result.

## Project team

**Mentees:** Darshan Mahale · Bhoomi Vaity
**Mentors:** Ojas Alai · Kavish Nasta

See [Contributors & Mentors](../contributors.md) for full acknowledgements.

## Next

- [Problem & Motivation](motivation.md) — why this needed automating
- [Architecture](architecture.md) — how the layers connect
