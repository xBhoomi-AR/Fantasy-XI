# FantasyXI — FPL Points Prediction

## Overview

FantasyXI uses machine learning to predict FPL player points
for upcoming gameweeks.

The prediction layer is designed primarily to rank players
rather than perfectly predict the exact number of points.

## Model

We use position-specific XGBoost regressors for:

- Goalkeepers
- Defenders
- Midfielders
- Forwards

Historical FPL, player, fixture, team and Understat-derived
information is transformed into engineered features.

## Pipeline

Supabase / historical data
        ↓
Data extraction
        ↓
Feature engineering
        ↓
Position-specific models
        ↓
XGBoost prediction
        ↓
Player ranking
        ↓
Downstream FantasyXI decision/RL layer

## Evaluation

Test season: 2025–26

Overall:
- MAE: 1.2689
- RMSE: 2.1067
- Spearman correlation: 0.7947

Position-wise ranking:
- GK: 0.7021
- DEF: 0.7881
- MID: 0.8133
- FWD: 0.8115

The model performs substantially better at ranking players
than at reproducing extreme 10+ point hauls.

## Limitations

The dataset is heavily imbalanced toward low-point
player-gameweeks. High-scoring performances are relatively
rare, resulting in higher errors for the 10+ point range.

The model is therefore intended as a player-ranking /
prediction layer for the downstream FantasyXI system rather
than a perfect point forecasting system.