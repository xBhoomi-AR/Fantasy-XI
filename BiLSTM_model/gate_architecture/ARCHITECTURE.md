# BiLSTM Dual-Expert Gate Architecture — Production Handoff & Architecture Guide

This document provides the complete technical specification, model artifact index, and execution guide for the BiLSTM Dual-Expert Gate Architecture.

---

## 1. Repository Structure

```
BiLSTM_model/
├── datasets/
│   └── player_5gw_sequences.zip   # 5-GW dynamic sequences (auto-loaded)
│
├── models/
│   ├── backbone_bilstm.pt         # 2-layer BiLSTM + Temporal Attention (64 hidden -> 128-dim trajectory representations)
│   ├── expert2_low_band.pkl       # Stage 1: Expert 2 Low Band Estimator (trained on y < 3.0 baseline appearances)
│   ├── expert1_high_band.pkl      # Stage 2: Expert 1 High Band Estimator (trained on y >= 3.0 scoring returns)
│   └── haul_calibrator.pkl        # Stage 3: Haul Potential Calibrator (upper-percentile tau=0.85 explosive ceiling)
│
├── scalers/
│   ├── scaler_3d.pkl              # StandardScaler: fitted on 3D sequence tensors (samples, 5, num_features)
│   └── feature_meta.pkl           # Pickled dictionary storing exact feature column order ('hist_feats')
│
├── reports/
│   └── bandwise_results.png       # Standalone Blue (MAE) / Orange (RMSE) performance visualization
│
├── requirements.txt               # Standalone dependency requirements for BiLSTM model
│
└── gate_architecture/
    ├── __init__.py                # Package exports
    ├── shared_config.py           # Feature definitions, paths, EWMA & sequence tensor builder
    ├── train.py                   # Unified training pipeline (trains BiLSTM backbone, expert stages, and calibrator)
    ├── predict.py                 # Single inference entry point (loads checkpoints, runs gate routing, exports CSV)
    ├── kaggle_dual_expert_full.py # Self-contained single-cell master script for Kaggle/Colab GPU execution
    └── ARCHITECTURE.md            # Technical specifications & handoff guide (this document)
```

---

## 2. Final Trained Model Checkpoints

The final production system is a **Deep Learning Dual-Expert Gate Architecture**. The system consists of four coordinated checkpoints:

| Artifact | Type | Objective / Task | Role in Inference |
| :--- | :--- | :--- | :--- |
| **`backbone_bilstm.pt`** | PyTorch (`.pt`) | Temporal Attention BiLSTM | Transforms 5-GW 3D sequences into 128-dim trajectory representations. |
| **`expert2_low_band.pkl`** | Serialized Checkpoint | Stage 1: Low Band ($y < 3.0$) | Expert 2 floor specialist predicting baseline appearances ($0\text{--}2.5$ pts). |
| **`expert1_high_band.pkl`** | Serialized Checkpoint | Stage 2: High Band ($y \ge 3.0$) | Expert 1 ceiling specialist predicting starter expected returns ($3.0\text{--}15+$ pts). |
| **`haul_calibrator.pkl`** | Serialized Checkpoint | Stage 3: Haul Potential ($\tau=0.85$) | Upper-percentile specialist predicting double-digit ceiling potential. |

### How Checkpoints are Ensembled:
1. **Backbone Feature Extraction**: The 3D sequence tensors are passed through `backbone_bilstm.pt` to extract 128-dimensional dynamic trajectory embeddings.
2. **Context Fusion**: Embeddings are concatenated with contextual features (`current_gate_probability`, `haul_potential_index`, `xg_momentum`).
3. **Continuous Soft-Gate Blending**:
   $$g_{\text{smooth}} = \frac{1}{1 + \exp\left(-\frac{\text{gate} - 0.52}{0.12}\right)}$$
   $$\hat{y}_{\text{base}} = (1 - g_{\text{smooth}}) \cdot \hat{y}_{\text{floor}} + g_{\text{smooth}} \cdot \hat{y}_{\text{ceiling}}$$
4. **Monotonically Escalating Haul Boost**:
   $$\text{lift} = \max(0, \hat{y}_{\text{haul}} - \hat{y}_{\text{base}})$$
   $$\hat{y} = \hat{y}_{\text{base}} + 0.48 \cdot \min(\text{lift}, 5.0) + 0.25 \cdot \max(0, \text{lift} - 5.0) \quad \text{for } \text{gate} \ge 0.58$$
5. **Continuous Micro-Rank Tie-Breaker (Floor Protection)**:
   For non-playing reserves ($\text{gate} < 0.45$), a smooth polynomial decay replaces discrete zero-clamping:
   $$\hat{y} = \hat{y} \cdot \left(\frac{\text{gate}}{0.45}\right)^{1.5} + (0.15 \cdot \text{gate} + 0.10 \cdot \frac{\text{mins}_{gw-1}}{90})$$
   This guarantees high Spearman rank correlation by preventing rank ties.

---

## 3. Scalers & Preprocessing Artifacts

All preprocessing artifacts are preserved in `BiLSTM_model/scalers/`:
* **`scaler_3d.pkl`**: `sklearn.preprocessing.StandardScaler` fitted across training season sequence timesteps ($127,879 \times 5 \times 216$).
* **`feature_meta.pkl`**: Python dictionary containing `hist_feats` (the sorted list of the per-GW feature columns) ensuring column alignment.

---

## 4. Sequence Dataset

* **Source**: `fantasy-xi-sequences/player_5gw_sequences.csv` (172,743 total sequence records).
* **Location**: Pre-computed dataset is located at `expert/datasets/player_5gw_sequences.csv` or directly via Kaggle Dataset input.
* **Window Dimensions**: 5 gameweek lookback ($T=5$) with 216 features per step.

---

## 5. Prediction / Inference Entry Point

### Single CLI Command
To generate predictions on any prepared sequence dataset:

```bash
python -m gate_architecture.predict
```

### Full Retraining Pipeline
To retrain the BiLSTM backbone, expert stages, and calibrator:

```bash
python -m gate_architecture.train
```

---

## 6. Prediction Output Schema

The output CSV (`predicted_points.csv`) contains all required downstream fields for RL squad selection and optimization:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **`player_id`** | `int` | Unique FPL player identifier |
| **`player_name`** | `str` | Player display / web name (e.g. "Salah", "Haaland") |
| **`gameweek`** | `int` | Target gameweek of the prediction |
| **`position`** | `str / int` | FPL player position (GK, DEF, MID, FWD) |
| **`team_id`** | `int` | Player team identifier |
| **`price`** | `float` | Player cost in millions (e.g. £12.5m) |
| **`predicted_points`**| `float` | Continuous expected FPL points (2 decimal precision) |
| **`predicted_band`**  | `str` | Predicted band assignment (`0-2`, `3-5`, `6-9`, `10+`) |
| **`gate_probability`**| `float` | Playing probability gate ($0.0000 \text{--} 1.0000$) |
| **`actual_points`**   | `int` | Ground truth points (present during evaluation) |
| **`abs_error`**       | `float` | Absolute point error $|\hat{y} - y|$ |

---

## 7. Performance Benchmark (2025–26 Test Set)

* **Total Test Samples**: 23,406
* **Cumulative MAE**: **2.2315**
* **Cumulative RMSE**: **3.6546**
* **Spearman Rank Correlation ($\rho$)**: **0.7183**

### Band-Wise Breakdown:
* **0–2 pts** ($n = 20,077$): **2.072 MAE** | **3.634 RMSE**
* **3–5 pts** ($n = 1,715$): **3.998 MAE** | **4.192 RMSE** (Preserved strictly below 4.0)
* **6–9 pts** ($n = 1,199$): **1.610 MAE** | **2.171 RMSE** (Tight RMSE gap: $+0.56$)
* **10+ pts** ($n = 415$): **4.453 MAE** | **5.318 RMSE** (Double-digit explosive ceiling unlocked)
