import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Set random seeds for consistent results
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SIZE = 5
BATCH_SIZE = 256
TARGET = "total_points"
GATE_COL = "current_gate_probability"

TRAIN_SEASONS = ["2016-17","2017-18","2018-19","2019-20","2020-21","2021-22","2022-23","2023-24"]
VAL_SEASONS = ["2024-25"]
TEST_SEASONS = ["2025-26"]

POS_WEIGHT_MAP = {1: 1.35, 2: 1.30, 3: 1.15, 4: 1.10}
BAND_NAMES = ["0-2", "3-5", "6-9", "10+"]

def get_paths():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    bilstm_dir = os.path.dirname(cur_dir)
    project_root = os.path.dirname(bilstm_dir)

    data_candidates = [
        os.path.join(bilstm_dir, "datasets", "player_5gw_sequences.csv"),
        os.path.join(bilstm_dir, "datasets", "player_5gw_sequences.zip"),
        os.path.join(project_root, "expert", "datasets", "player_5gw_sequences.csv"),
        os.path.join(project_root, "expert", "datasets", "player_5gw_sequences.zip"),
        os.path.join(project_root, "Datasets", "player_5gw_sequences.csv"),
        os.path.join(project_root, "data", "player_5gw_sequences.csv"),
        "/kaggle/input/datasets/xbhoomi/fantasy-xi-sequences/player_5gw_sequences.csv",
        "/kaggle/input/fantasy-xi-sequences/player_5gw_sequences.csv",
        "player_5gw_sequences.csv",
        "player_5gw_sequences.zip",
    ]

    data_path = None
    for candidate in data_candidates:
        if os.path.exists(candidate):
            data_path = candidate
            break

    if data_path is None:
        import glob
        matches = glob.glob("/kaggle/input/**/player_5gw_sequences.csv", recursive=True) + glob.glob("/kaggle/input/**/player_5gw_sequences.zip", recursive=True)
        data_path = matches[0] if matches else "player_5gw_sequences.csv"

    model_dir = os.path.join(bilstm_dir, "models")
    scaler_dir = os.path.join(bilstm_dir, "scalers")
    reports_dir = os.path.join(bilstm_dir, "reports")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    return data_path, model_dir, scaler_dir, reports_dir

def engineer_features(df):
    # Weight recent games higher when computing moving averages
    w = np.array([0.05, 0.10, 0.15, 0.30, 0.40])
    for metric in ["expected_goals", "expected_assists", "bps", "ict_index", "minutes"]:
        cols = [f"gw_minus_{s}_{metric}" for s in range(5, 0, -1) if f"gw_minus_{s}_{metric}" in df.columns]
        if len(cols) == 5:
            arr = np.column_stack([pd.to_numeric(df[c], errors="coerce").fillna(0).values for c in cols])
            df[f"ewma_{metric}"] = np.dot(arr, w).astype(np.float32)
        else:
            df[f"ewma_{metric}"] = 0.0

    df["haul_potential_index"] = (
        df["ewma_expected_goals"] * 4.0 +
        df["ewma_expected_assists"] * 3.0 +
        df["ewma_ict_index"] * 0.1
    ).astype(np.float32)

    gw1_xg = pd.to_numeric(df.get("gw_minus_1_expected_goals", 0), errors="coerce").fillna(0)
    rest_xg = [df.get(f"gw_minus_{s}_expected_goals", 0) for s in range(2, 6)]
    avg_rest = np.mean([pd.to_numeric(c, errors="coerce").fillna(0) for c in rest_xg], axis=0) if rest_xg else 0.0
    df["xg_momentum"] = (gw1_xg - avg_rest).astype(np.float32)

    fdr = pd.to_numeric(df.get("current_team_fdr", 3.0), errors="coerce").fillna(3.0)
    form = pd.to_numeric(df.get("gw_minus_1_form_last4", 0.0), errors="coerce").fillna(0.0)
    df["fdr_decay_form"] = (form / (fdr + 0.1)).astype(np.float32)

    return df

def get_feature_list(df):
    hist_feats = sorted(list(set([
        c.split("gw_minus_1_")[-1] for c in df.columns if c.startswith("gw_minus_1_")
    ])))
    return hist_feats

def build_3d_tensor(data, hist_feats, window_size=WINDOW_SIZE):
    n = len(data)
    X_3d = np.zeros((n, window_size, len(hist_feats)), dtype=np.float32)
    for step in range(1, window_size + 1):
        t = window_size - step
        for fi, feat in enumerate(hist_feats):
            col = f"gw_minus_{step}_{feat}"
            if col in data.columns:
                X_3d[:, t, fi] = pd.to_numeric(data[col], errors="coerce").fillna(0).values
    return X_3d

def build_hybrid_matrix(data, embeddings):
    # Combine neural embeddings with key match context
    wide_cols = ["current_gate_probability", "haul_potential_index", "xg_momentum"]
    wide_arr = np.column_stack([
        pd.to_numeric(data.get(c, 0), errors="coerce").fillna(0).values for c in wide_cols
    ])
    return np.hstack([embeddings, wide_arr]).astype(np.float32)

class TemporalAttention(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.w = nn.Sequential(
            nn.Linear(d_in, d_in // 2),
            nn.Tanh(),
            nn.Linear(d_in // 2, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.w(x), dim=1)
        return (x * weights).sum(dim=1)

class BiLSTMBackbone(nn.Module):
    def __init__(self, in_dim, h=64):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, h, num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.20
        )
        self.attn = TemporalAttention(h * 2)
        self.head = nn.Linear(h * 2, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        emb = self.attn(o)
        pred = self.head(emb).squeeze(-1)
        return pred, emb
