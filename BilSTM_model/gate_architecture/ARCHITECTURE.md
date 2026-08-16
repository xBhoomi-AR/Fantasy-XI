# Dual-Expert Gate Architecture — Technical Overview

An end-to-end deep learning framework for Fantasy Premier League (FPL) player point prediction, featuring a **Gated Mixture of Experts (MoE)** architecture, multi-task temporal BiLSTM networks, position-weighted loss functions, auxiliary minutes regression, and extended multi-window rolling feature engineering.

---

## Repository Structure

```
gate_architecture/
├── __init__.py                        # Package exports & module initialization
├── shared_config.py                   # Data loaders, 140+ feature pipeline, PositionWeightedLoss & Neural Architectures
├── expert2_bilstm.py                  # Stage 1: Upgraded Expert 2 Training (0–2 Pts) with Aux Minutes Head
├── expert2_predictions.py             # Stage 1: Expert 2 Inference Runner
├── expert1_low_band_bilstm.py         # Stage 2: Expert 1 Low Band Training (3–6 Pts)
├── expert1_low_band_predictions.py    # Stage 2: Expert 1 Low Band Inference Runner
├── expert1_high_band_bilstm.py        # Stage 3: Expert 1 High Band Training (3–4, 5–6, 7–10+ Pts)
├── expert1_high_band_predictions.py   # Stage 3: Expert 1 High Band Inference Runner
├── combined_predictions.py            # Final Unified System Evaluator (Clean Text Report)
└── kaggle_dual_expert_full.py         # Standalone production master script for Kaggle/Colab execution
```

---

## Key Architecture Highlights

### 1. Gate Probability Routing
* **Expert 2 Routing ($\text{Gate Prob} < 0.50$)**: Handles squad players, bench risks, and low-minute substitute appearances ($0\text{--}2$ points).
* **Expert 1 Routing ($\text{Gate Prob} \ge 0.50$)**: Handles regular starters and high-scoring point returns ($3\text{--}4$, $5\text{--}6$, $7\text{--}10+$ points).

### 2. Position-Weighted Regression Loss (`PositionWeightedSmoothL1Loss`)
FPL scoring varies dramatically by player position due to discrete clean-sheet rewards ($+4$ pts) and goal bonuses. To address target variance:
* **Defenders (DEF, Code 2)**: Weight = `1.35` (Heaviest penalty on defensive clean-sheet variance).
* **Goalkeepers (GK, Code 1)**: Weight = `1.25`.
* **Midfielders (MID, Code 3)**: Weight = `1.15`.
* **Forwards (FWD, Code 4)**: Weight = `1.10`.

### 3. Auxiliary Minutes Regression ($\hat{M}$) in Expert 2
Points in the $0\text{--}2$ band are strongly correlated with playing time. Expert 2 uses a multi-task head predicting both expected points ($\hat{y}$) and expected minutes ($\hat{M}$):
* $\hat{M} < 1.0 \implies 0.0$ pts (Unused bench / DNP)
* $1.0 \le \hat{M} < 59.5 \implies 1.0$ pt (Substitute appearance)
* $\hat{M} \ge 59.5 \implies 2.0$ pts (Starter baseline, with defensive xGC penalty gating)

### 4. Extended 140+ Feature Pipeline
* **Multi-Window Rolling Averages (`avg3`, `avg5`, `avg10`)**: Calculated for player points, minutes, xG, xA, clean sheets, BPS, ICT Index, team goals scored/conceded, and opponent goals scored/conceded.
* **Exponentially Weighted Moving Averages (EWMA)**: Calculated across past 5 gameweeks with recency weights `[0.05, 0.10, 0.15, 0.30, 0.40]`.
* **Rotation & Availability Signals**: `sub_app_flag`, `unused_bench_flag`, `xg_momentum`, and `fdr_decay_form`.

---

##  Execution Guide

### Option A: Running Individual Modular Pipeline Stages
Execute each stage sequentially from the repository root:

```bash
# 1. Train Stage 1: Upgraded Expert 2 (0-2 Pts)
python -m gate_architecture.expert2_bilstm

# 2. Train Stage 2: Expert 1 Low Band (3-6 Pts)
python -m gate_architecture.expert1_low_band_bilstm

# 3. Train Stage 3: Expert 1 High Band (3-4, 5-6, 7-10+ Pts)
python -m gate_architecture.expert1_high_band_bilstm

# 4. Run Combined Final System Evaluation
python -m gate_architecture.combined_predictions
```

### Option B: Running Standalone Single-Cell Master Script
For execution in cloud GPU environments (Kaggle Notebooks or Google Colab):
1. Copy the contents of `kaggle_dual_expert_full.py`.
2. Paste into a single GPU notebook cell and execute.

---

## Benchmark Evaluation Summary (2025–26 Test Set)

* **Total Test Samples Evaluated**: 23,406
* **Cumulative MAE**: **1.5401**
* **Cumulative RMSE**: **2.6978**
* **Spearman Rank Correlation ($\rho$)**: **0.7175**
* **Cumulative Classification Accuracy**: **65.37%**
* **Macro F1 Score**: **0.3582**

