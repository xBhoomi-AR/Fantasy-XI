import os, sys, random, pickle, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, confusion_matrix, classification_report
)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count()
print(f"Device: {device} | GPUs: {N_GPUS}")

if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.set_device(0)
    _ = torch.empty(1, device=device)

CSV_FILENAME = "player_5gw_sequences.csv"
matches = glob.glob(f"/kaggle/input/**/{CSV_FILENAME}", recursive=True)
if matches:
    DATA_PATH = matches[0]
elif os.path.exists(f"/content/drive/MyDrive/Fantasy_XI/expert/datasets/{CSV_FILENAME}"):
    DATA_PATH = f"/content/drive/MyDrive/Fantasy_XI/expert/datasets/{CSV_FILENAME}"
else:
    raise FileNotFoundError(CSV_FILENAME)
print(f"Found CSV: {DATA_PATH}")

OUT_BASE  = "/kaggle/working" if os.path.exists("/kaggle/input") else "/content"
MODEL_DIR = os.path.join(OUT_BASE, "models", "gate_arch_full_features")
os.makedirs(MODEL_DIR, exist_ok=True)

WINDOW_SIZE   = 5
BATCH_SIZE    = 128
TARGET        = "total_points"
TRAIN_SEASONS = ["2016-17","2017-18","2018-19","2019-20","2020-21","2021-22","2022-23","2023-24"]
VAL_SEASONS   = ["2024-25"]
TEST_SEASONS  = ["2025-26"]

POS_WEIGHT_MAP = {1: 1.35, 2: 1.30, 3: 1.15, 4: 1.10}

BAND_NAMES_E1  = ["3-5", "6-9", "10+"]
BAND_NAMES_ALL = ["0-2 pts", "3-5 pts", "6-9 pts", "10+ pts"]

RAW_HIST = [
    "position","position_code","position_norm","understat_position","started","minutes","season_order",
    "goals_scored","assists","expected_goals","expected_assists","expected_goal_involvements",
    "shots","key_passes","clean_sheets","goals_conceded","expected_goals_conceded",
    "saves","recoveries","tackles","clearances_blocks_interceptions","defensive_contribution",
    "yellow_cards","red_cards","own_goals","penalties_saved","penalties_missed",
    "bonus","bps","influence","creativity","threat","ict_index","npxg","xg","xa","xgi",
    "team_id","opponent_team_id","played_last_match","started_last_match","high_score_flag",
    "chance_of_playing_next_round","chance_of_playing_this_round","status_code",
    "avg_minutes_last3","avg_minutes_last5","minutes_volatility_last5",
    "appearance_rate_last5","starts_rate_last5","games_played_season",
    "transfers_in","transfers_out","transfers_balance","value",
    "form_last4","xG_last4","xA_last4","team_fdr","was_home","was_home_int","days_rest",
    "minutes_last_gw","started_last_gw","appearances_last3","appearances_last5",
    "starts_last3","starts_last5","zero_minutes_last3","zero_minutes_last5",
    "minutes_trend","appearance_rate_last3","appearance_trend","consecutive_starts",
    "has_fixture","fixture_difficulty","consecutive_appearances",
    "player_total_points_last1","player_total_points_avg3","player_total_points_avg5","player_total_points_avg10","player_total_points_avg38",
    "player_minutes_last1","player_minutes_avg3","player_minutes_avg5","player_minutes_avg10","player_minutes_avg38",
    "player_goals_scored_last1","player_goals_scored_avg3","player_goals_scored_avg5","player_goals_scored_avg10","player_goals_scored_avg38",
    "player_assists_last1","player_assists_avg3","player_assists_avg5","player_assists_avg10","player_assists_avg38",
    "player_expected_goals_last1","player_expected_goals_avg3","player_expected_goals_avg5","player_expected_goals_avg10","player_expected_goals_avg38",
    "player_expected_assists_last1","player_expected_assists_avg3","player_expected_assists_avg5","player_expected_assists_avg10","player_expected_assists_avg38",
    "player_expected_goal_involvements_last1","player_expected_goal_involvements_avg3","player_expected_goal_involvements_avg5","player_expected_goal_involvements_avg10","player_expected_goal_involvements_avg38",
    "player_clean_sheets_last1","player_clean_sheets_avg3","player_clean_sheets_avg5","player_clean_sheets_avg10","player_clean_sheets_avg38",
    "player_goals_conceded_last1","player_goals_conceded_avg3","player_goals_conceded_avg5","player_goals_conceded_avg10","player_goals_conceded_avg38",
    "player_expected_goals_conceded_last1","player_expected_goals_conceded_avg3","player_expected_goals_conceded_avg5","player_expected_goals_conceded_avg10","player_expected_goals_conceded_avg38",
    "player_saves_last1","player_saves_avg3","player_saves_avg5","player_saves_avg10","player_saves_avg38",
    "player_recoveries_last1","player_recoveries_avg3","player_recoveries_avg5","player_recoveries_avg10","player_recoveries_avg38",
    "player_tackles_last1","player_tackles_avg3","player_tackles_avg5","player_tackles_avg10","player_tackles_avg38",
    "player_defensive_contribution_last1","player_defensive_contribution_avg3","player_defensive_contribution_avg5","player_defensive_contribution_avg10","player_defensive_contribution_avg38",
    "player_bonus_last1","player_bonus_avg3","player_bonus_avg5","player_bonus_avg10","player_bonus_avg38",
    "player_bps_last1","player_bps_avg3","player_bps_avg5","player_bps_avg10","player_bps_avg38",
    "player_influence_last1","player_influence_avg3","player_influence_avg5","player_influence_avg10","player_influence_avg38",
    "player_creativity_last1","player_creativity_avg3","player_creativity_avg5","player_creativity_avg10","player_creativity_avg38",
    "player_threat_last1","player_threat_avg3","player_threat_avg5","player_threat_avg10","player_threat_avg38",
    "player_ict_index_last1","player_ict_index_avg3","player_ict_index_avg5","player_ict_index_avg10","player_ict_index_avg38",
    "player_npxg_last1","player_npxg_avg3","player_npxg_avg5","player_npxg_avg10","player_npxg_avg38",
    "player_xg_last1","player_xg_avg3","player_xg_avg5","player_xg_avg10","player_xg_avg38",
    "player_xa_last1","player_xa_avg3","player_xa_avg5","player_xa_avg10","player_xa_avg38",
    "player_xgi_last1","player_xgi_avg3","player_xgi_avg5","player_xgi_avg10","player_xgi_avg38",
    "player_high_score_flag_last1","player_high_score_flag_avg3","player_high_score_flag_avg5","player_high_score_flag_avg10","player_high_score_flag_avg38",
    "team_team_goals_for_last1","team_team_goals_for_avg3","team_team_goals_for_avg5",
    "team_team_goals_against_last1","team_team_goals_against_avg3","team_team_goals_against_avg5",
    "team_team_points_result_avg5",
    "opp_goals_for_last1","opp_goals_for_avg3","opp_goals_for_avg5",
    "opp_goals_against_last1","opp_goals_against_avg3","opp_goals_against_avg5",
    "opp_points_result_avg5"
]

ENG_PER_STEP = ["pos_weight","xg_per90","xa_per90","bps_per90","def_cs_mult","sub_app_flag","unused_bench_flag","goal_threat_index","offensive_gi","haul_potential_index"]
HIST_FEATS   = RAW_HIST + ENG_PER_STEP

RAW_CTX = ["current_team_fdr","current_was_home","current_has_fixture","current_days_rest","current_gate_probability"]

EXTENDED_GLOBAL = [
    "ewma_expected_goals","ewma_expected_assists","ewma_bps","ewma_ict_index","ewma_minutes",
    "xg_momentum","fdr_decay_form","haul_potential_index","captaincy_score",
    "player_total_points_avg3","player_total_points_avg5","player_total_points_avg10",
    "player_minutes_avg3","player_minutes_avg5","player_minutes_avg10",
    "player_expected_goals_avg3","player_expected_goals_avg5","player_expected_goals_avg10",
    "player_expected_assists_avg3","player_expected_assists_avg5","player_expected_assists_avg10",
    "player_clean_sheets_avg3","player_clean_sheets_avg5","player_clean_sheets_avg10",
    "player_goals_conceded_avg3","player_goals_conceded_avg5","player_goals_conceded_avg10",
    "player_bps_avg3","player_bps_avg5","player_bps_avg10",
    "player_ict_index_avg3","player_ict_index_avg5","player_ict_index_avg10",
    "team_goals_for_avg3","team_goals_for_avg5","team_goals_against_avg3","team_goals_against_avg5",
    "opp_goals_for_avg3","opp_goals_for_avg5","opp_goals_against_avg3","opp_goals_against_avg5"
]

CTX_FEATS = RAW_CTX + ["xg_momentum","fdr_decay_form","haul_potential_index","captaincy_score"]

def engineer_features(df):
    for step in range(1, WINDOW_SIZE+1):
        mins_col  = f"gw_minus_{step}_minutes"
        mins      = pd.to_numeric(df[mins_col],  errors="coerce").fillna(0) if mins_col in df.columns else pd.Series(0, index=df.index)
        
        start_col = f"gw_minus_{step}_started"
        started   = pd.to_numeric(df[start_col], errors="coerce").fillna(0) if start_col in df.columns else pd.Series(0, index=df.index)
        
        ms        = np.where(mins > 0, mins, 90.0) / 90.0
        
        xg_col    = f"gw_minus_{step}_expected_goals"
        uxg_col   = f"gw_minus_{step}_xg"
        xg        = pd.to_numeric(df[xg_col], errors="coerce").fillna(0) if xg_col in df.columns else pd.Series(0, index=df.index)
        if uxg_col in df.columns:
            uxg   = pd.to_numeric(df[uxg_col], errors="coerce").fillna(0)
            xg    = pd.Series(np.where(xg > 0, xg, uxg), index=df.index)
            
        xa_col    = f"gw_minus_{step}_expected_assists"
        uxa_col   = f"gw_minus_{step}_xa"
        xa        = pd.to_numeric(df[xa_col], errors="coerce").fillna(0) if xa_col in df.columns else pd.Series(0, index=df.index)
        if uxa_col in df.columns:
            uxa   = pd.to_numeric(df[uxa_col], errors="coerce").fillna(0)
            xa    = pd.Series(np.where(xa > 0, xa, uxa), index=df.index)
            
        bps_col   = f"gw_minus_{step}_bps"
        bps       = pd.to_numeric(df[bps_col], errors="coerce").fillna(0) if bps_col in df.columns else pd.Series(0, index=df.index)
        
        cs_col    = f"gw_minus_{step}_clean_sheets"
        cs        = pd.to_numeric(df[cs_col], errors="coerce").fillna(0) if cs_col in df.columns else pd.Series(0, index=df.index)

        shots_col = f"gw_minus_{step}_shots"
        shots     = pd.to_numeric(df[shots_col], errors="coerce").fillna(0) if shots_col in df.columns else pd.Series(0, index=df.index)
        
        ict_col   = f"gw_minus_{step}_ict_index"
        ict       = pd.to_numeric(df[ict_col], errors="coerce").fillna(0) if ict_col in df.columns else pd.Series(0, index=df.index)

        pos_col   = f"gw_minus_{step}_position_code"
        pos_col2  = f"gw_minus_{step}_position"
        if pos_col in df.columns:
            pos   = pd.to_numeric(df[pos_col], errors="coerce").fillna(3)
        elif pos_col2 in df.columns:
            pos   = pd.to_numeric(df[pos_col2], errors="coerce").fillna(3)
        else:
            pos   = pd.Series(3, index=df.index)

        pw = pos.map(POS_WEIGHT_MAP).fillna(1.15).astype(np.float32)

        xg_per90  = ((xg / ms) * pw).astype(np.float32)
        xa_per90  = ((xa / ms) * pw).astype(np.float32)
        bps_per90 = ((bps / ms) * pw).astype(np.float32)
        hpi       = (xg * 4.0 + xa * 3.0 + ict * 0.1).astype(np.float32)

        df[f"gw_minus_{step}_pos_weight"]        = pw
        df[f"gw_minus_{step}_xg_per90"]          = xg_per90
        df[f"gw_minus_{step}_xa_per90"]          = xa_per90
        df[f"gw_minus_{step}_bps_per90"]         = bps_per90
        df[f"gw_minus_{step}_def_cs_mult"]       = (cs * ((pos == 1) | (pos == 2)) * pw).astype(np.float32)
        df[f"gw_minus_{step}_sub_app_flag"]      = ((mins > 0) & (started == 0)).astype(np.float32)
        df[f"gw_minus_{step}_unused_bench_flag"] = (mins == 0).astype(np.float32)
        df[f"gw_minus_{step}_goal_threat_index"] = (shots * xg_per90 * pw).astype(np.float32)
        df[f"gw_minus_{step}_offensive_gi"]      = (((xg + xa) / np.where(mins > 0, mins, 1) * 90) * pw).astype(np.float32)
        df[f"gw_minus_{step}_haul_potential_index"] = hpi

    w = np.array([0.05, 0.10, 0.15, 0.30, 0.40])
    for metric in ["expected_goals","expected_assists","bps","ict_index","minutes"]:
        cols_avail = []
        for s in range(5, 0, -1):
            cname = f"gw_minus_{s}_{metric}"
            if metric == "expected_goals" and cname not in df.columns and f"gw_minus_{s}_xg" in df.columns:
                cname = f"gw_minus_{s}_xg"
            elif metric == "expected_assists" and cname not in df.columns and f"gw_minus_{s}_xa" in df.columns:
                cname = f"gw_minus_{s}_xa"
            if cname in df.columns:
                cols_avail.append(cname)
        if len(cols_avail) == 5:
            arr = np.column_stack([pd.to_numeric(df[c], errors="coerce").fillna(0).values for c in cols_avail])
            df[f"ewma_{metric}"] = np.dot(arr, w).astype(np.float32)
        else:
            df[f"ewma_{metric}"] = 0.0

    df["haul_potential_index"] = (df["ewma_expected_goals"] * 4.0 + df["ewma_expected_assists"] * 3.0 + df["ewma_ict_index"] * 0.1).astype(np.float32)
    
    for metric in ["total_points","minutes","expected_goals","expected_assists","clean_sheets","goals_conceded","bps","ict_index"]:
        for win in [3, 5, 10]:
            col_name = f"player_{metric}_avg{win}"
            if col_name not in df.columns:
                cols_stack = []
                for s in range(min(win, 5), 0, -1):
                    cname = f"gw_minus_{s}_{metric}"
                    if cname in df.columns:
                        cols_stack.append(pd.to_numeric(df[cname], errors="coerce").fillna(0).values)
                    elif metric == "total_points" and f"gw_minus_{s}_points" in df.columns:
                        cols_stack.append(pd.to_numeric(df[f"gw_minus_{s}_points"], errors="coerce").fillna(0).values)
                if cols_stack:
                    df[col_name] = np.mean(np.column_stack(cols_stack), axis=1).astype(np.float32)
                else:
                    df[col_name] = 0.0

    for g_type in ["goals_scored", "goals_conceded"]:
        for win in [3, 5]:
            t_col = f"team_goals_for_avg{win}" if g_type == "goals_scored" else f"team_goals_against_avg{win}"
            o_col = f"opp_goals_for_avg{win}" if g_type == "goals_scored" else f"opp_goals_against_avg{win}"
            if t_col not in df.columns:
                cols_stack = [pd.to_numeric(df[f"gw_minus_{s}_{g_type}"], errors="coerce").fillna(0).values for s in range(min(win, 5), 0, -1) if f"gw_minus_{s}_{g_type}" in df.columns]
                if cols_stack:
                    avg_v = np.mean(np.column_stack(cols_stack), axis=1).astype(np.float32)
                    df[t_col] = avg_v
                    df[o_col] = avg_v
                else:
                    df[t_col] = 0.0
                    df[o_col] = 0.0

    gw1_xg_col  = "gw_minus_1_expected_goals"
    gw1_xg      = pd.to_numeric(df[gw1_xg_col], errors="coerce").fillna(0) if gw1_xg_col in df.columns \
                  else pd.to_numeric(df["gw_minus_1_xg"], errors="coerce").fillna(0) if "gw_minus_1_xg" in df.columns \
                  else pd.Series(0, index=df.index)
    rest_cols   = [f"gw_minus_{s}_expected_goals" if f"gw_minus_{s}_expected_goals" in df.columns else f"gw_minus_{s}_xg"
                   for s in range(2,6) if f"gw_minus_{s}_expected_goals" in df.columns or f"gw_minus_{s}_xg" in df.columns]
    avg_xg_rest = np.mean([pd.to_numeric(df[c], errors="coerce").fillna(0) for c in rest_cols], axis=0) if rest_cols else 0.0
    
    df["xg_momentum"]    = (gw1_xg - avg_xg_rest).astype(np.float32)
    fdr  = pd.to_numeric(df["current_team_fdr"], errors="coerce").fillna(3.0) if "current_team_fdr" in df.columns else 3.0
    
    form_col = "gw_minus_1_form_last4"
    form = pd.to_numeric(df[form_col], errors="coerce").fillna(0.0) if form_col in df.columns else 0.0
    df["fdr_decay_form"] = (form / (fdr + 0.1)).astype(np.float32)
    
    pts_avg3 = pd.to_numeric(df["player_total_points_avg3"], errors="coerce").fillna(0.0) if "player_total_points_avg3" in df.columns else 0.0
    df["captaincy_score"] = (pts_avg3 / (fdr + 0.1) * df["haul_potential_index"]).astype(np.float32)
    return df

def build_matrices(data, target_col):
    n = len(data)
    X_3d  = np.zeros((n, WINDOW_SIZE, len(HIST_FEATS)), dtype=np.float32)
    X_ctx = np.zeros((n, len(CTX_FEATS)),               dtype=np.float32)
    
    for step in range(1, WINDOW_SIZE+1):
        t = WINDOW_SIZE - step
        for fi, feat in enumerate(HIST_FEATS):
            col = f"gw_minus_{step}_{feat}"
            if col in data.columns:
                vals = pd.to_numeric(data[col], errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0)
                X_3d[:, t, fi] = vals.values
                
    for i, feat in enumerate(CTX_FEATS):
        if feat in data.columns:
            vals = pd.to_numeric(data[feat], errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0)
            X_ctx[:, i] = vals.values

    flat_cols  = [f"gw_minus_{s}_{f}" for s in range(1, WINDOW_SIZE+1) for f in HIST_FEATS]
    flat_cols += CTX_FEATS + EXTENDED_GLOBAL
    
    for col in flat_cols:
        if col not in data.columns:
            data[col] = 0.0
            
    X_flat = data[flat_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0).values.astype(np.float32)
    
    y      = data[target_col].values.astype(np.int64)
    y_reg  = data[TARGET].values.astype(np.float32)
    
    mins_col = "gw_minus_1_minutes"
    y_mins   = pd.to_numeric(data[mins_col], errors="coerce").fillna(0).values.astype(np.float32) if mins_col in data.columns else np.zeros(n, dtype=np.float32)
    
    pos_col = "gw_minus_1_position_code" if "gw_minus_1_position_code" in data.columns else "gw_minus_1_position"
    pos     = pd.to_numeric(data[pos_col], errors="coerce").fillna(3).values.astype(np.int64) if pos_col in data.columns else np.full(n, 3, dtype=np.int64)
    
    return X_3d, X_ctx, X_flat, y, y_reg, y_mins, pos

def sc3d(X, scaler):
    s = X.shape
    return scaler.transform(X.reshape(-1, len(HIST_FEATS))).reshape(s)

def compute_weights(targets, n_cls):
    cnts = np.bincount(targets, minlength=n_cls)
    w = 1.0 / np.maximum(cnts, 1)
    w = w / w.sum() * n_cls
    return torch.tensor(w, dtype=torch.float32).to(device)

def get_pos_w_tensor(bpos):
    w = torch.ones_like(bpos, dtype=torch.float32)
    for p_code, val in POS_WEIGHT_MAP.items():
        w[bpos == p_code] = val
    return w

class PositionWeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target, positions):
        err = (target - pred).abs()
        l1  = torch.where(err < self.beta, 0.5 * (err**2) / self.beta, err - 0.5 * self.beta)
        weights = get_pos_w_tensor(positions)
        return (weights * l1).mean()

class TAttn(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(h, h//2), nn.Tanh(), nn.Linear(h//2, 1))
    def forward(self, x):
        return (x * torch.softmax(self.a(x), dim=1)).sum(dim=1)

class Expert2BiLSTM(nn.Module):
    def __init__(self, in_sz, ctx_sz, n_cls=3, h=192, drop=0.25):
        super().__init__()
        self.lstm = nn.LSTM(in_sz, h, num_layers=2, batch_first=True, dropout=drop, bidirectional=True)
        d = h*2
        self.attn    = TAttn(d)
        self.ctx_net = nn.Sequential(nn.Linear(ctx_sz, 64), nn.ReLU(), nn.Dropout(drop))
        self.backbone = nn.Sequential(
            nn.Linear(d+64, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(drop)
        )
        self.cls_head  = nn.Linear(128, n_cls)
        self.reg_head  = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.mins_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, s, c):
        o,_ = self.lstm(s)
        feat = self.backbone(torch.cat([self.attn(o), self.ctx_net(c)], dim=1))
        return self.cls_head(feat), self.reg_head(feat).squeeze(-1), self.mins_head(feat).squeeze(-1)

class Expert2MLP(nn.Module):
    def __init__(self, in_sz, n_cls=3, drop=0.30):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_sz, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(drop),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(drop)
        )
        self.cls_head  = nn.Linear(128, n_cls)
        self.reg_head  = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.mins_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        feat = self.backbone(x)
        return self.cls_head(feat), self.reg_head(feat).squeeze(-1), self.mins_head(feat).squeeze(-1)

class MultiTaskBiLSTM(nn.Module):
    def __init__(self, in_sz, ctx_sz, n_cls, h=192, drop=0.25):
        super().__init__()
        self.lstm = nn.LSTM(in_sz, h, num_layers=2, batch_first=True, dropout=drop, bidirectional=True)
        d = h*2
        self.attn    = TAttn(d)
        self.ctx_net = nn.Sequential(nn.Linear(ctx_sz, 64), nn.ReLU(), nn.Dropout(drop))
        self.backbone = nn.Sequential(
            nn.Linear(d+64, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(drop)
        )
        self.cls_head = nn.Linear(128, n_cls)
        self.reg_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, s, c):
        o,_ = self.lstm(s)
        feat = self.backbone(torch.cat([self.attn(o), self.ctx_net(c)], dim=1))
        return self.cls_head(feat), self.reg_head(feat).squeeze(-1)

class MultiTaskMLP(nn.Module):
    def __init__(self, in_sz, n_cls, drop=0.30):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_sz, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(drop),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(drop)
        )
        self.cls_head = nn.Linear(128, n_cls)
        self.reg_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        feat = self.backbone(x)
        return self.cls_head(feat), self.reg_head(feat).squeeze(-1)

def wrap(m):
    if N_GPUS > 1: m = nn.DataParallel(m)
    return m.to(device)

def train_step_e2(model, loader, cls_crit, reg_crit, mins_crit, optimizer, mode="lstm"):
    model.train()
    tot_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        if mode == "lstm":
            bh, bc, by, byr, bym, bpos = batch
            bh, bc, by, byr, bym, bpos = bh.to(device), bc.to(device), by.to(device), byr.to(device), bym.to(device), bpos.to(device)
            logits, pts, mins          = model(bh, bc)
        else:
            bf, by, byr, bym, bpos = batch
            bf, by, byr, bym, bpos = bf.to(device), by.to(device), byr.to(device), bym.to(device), bpos.to(device)
            logits, pts, mins        = model(bf)
        
        pw = get_pos_w_tensor(bpos)
        loss_cls  = (pw * F.cross_entropy(logits, by, reduction="none")).mean()
        loss_reg  = reg_crit(pts, byr, bpos)
        loss_mins = mins_crit(mins, bym)
        loss      = loss_cls + 1.5 * loss_reg + 0.02 * loss_mins

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot_loss += loss.item() * by.size(0)
    return tot_loss / len(loader.dataset)

@torch.no_grad()
def eval_step_e2(model, loader, cls_crit, reg_crit, mins_crit, mode="lstm"):
    model.eval()
    tot_loss, preds, truths = 0.0, [], []
    for batch in loader:
        if mode == "lstm":
            bh, bc, by, byr, bym, bpos = batch
            bh, bc, by, byr, bym, bpos = bh.to(device), bc.to(device), by.to(device), byr.to(device), bym.to(device), bpos.to(device)
            logits, pts, mins          = model(bh, bc)
        else:
            bf, by, byr, bym, bpos = batch
            bf, by, byr, bym, bpos = bf.to(device), by.to(device), byr.to(device), bym.to(device), bpos.to(device)
            logits, pts, mins        = model(bf)
            
        pw = get_pos_w_tensor(bpos)
        loss_cls  = (pw * F.cross_entropy(logits, by, reduction="none")).mean()
        loss_reg  = reg_crit(pts, byr, bpos)
        loss_mins = mins_crit(mins, bym)
        loss      = loss_cls + 1.5 * loss_reg + 0.02 * loss_mins

        tot_loss += loss.item() * by.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        truths.extend(by.cpu().tolist())
    return tot_loss / len(loader.dataset), preds, truths

def fit_e2(model, tr_ld, va_ld, epochs, lr=5e-4, wd=1e-4, patience=8, cls_crit=None, mode="lstm"):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    reg_crit  = PositionWeightedSmoothL1Loss()
    mins_crit = nn.SmoothL1Loss()
    if cls_crit is None: cls_crit = nn.CrossEntropyLoss()
    
    best_f1, best_state, pat = -1.0, None, 0
    for ep in range(epochs):
        tr_loss = train_step_e2(model, tr_ld, cls_crit, reg_crit, mins_crit, opt, mode=mode)
        va_loss, vp, vt = eval_step_e2(model, va_ld, cls_crit, reg_crit, mins_crit, mode=mode)
        va_f1  = f1_score(vt, vp, average="macro", zero_division=0)
        va_acc = accuracy_score(vt, vp)
        sched.step()
        improved = va_f1 > best_f1
        if improved:
            best_f1    = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        lr_now = opt.param_groups[0]["lr"]
        print(f"  Ep {ep+1:02d}/{epochs} | TrLoss: {tr_loss:.4f} | VaLoss: {va_loss:.4f} | VaAcc: {va_acc*100:.2f}% | VaF1: {va_f1:.4f} | LR: {lr_now:.1e}{' *' if improved else ''}")
        if pat >= patience:
            print(f"  Early stop @ ep {ep+1} (best F1={best_f1:.4f})")
            break
    model.load_state_dict(best_state)
    return model

def train_step(model, loader, cls_crit, reg_crit, optimizer, mode="lstm", lambda_reg=2.0):
    model.train()
    tot_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        if mode == "lstm":
            bh, bc, by, byr, bym, bpos = batch
            bh, bc, by, byr, bym, bpos = bh.to(device), bc.to(device), by.to(device), byr.to(device), bym.to(device), bpos.to(device)
            logits, pts                = model(bh, bc)
        else:
            bf, by, byr, bym, bpos = batch
            bf, by, byr, bym, bpos = bf.to(device), by.to(device), byr.to(device), bym.to(device), bpos.to(device)
            logits, pts            = model(bf)
        
        pw = get_pos_w_tensor(bpos)
        loss_cls = (pw * F.cross_entropy(logits, by, reduction='none')).mean()
        loss_reg = reg_crit(pts, byr, bpos)
        loss     = loss_cls + lambda_reg * loss_reg

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot_loss += loss.item() * by.size(0)
    return tot_loss / len(loader.dataset)

@torch.no_grad()
def eval_step(model, loader, cls_crit, reg_crit, mode="lstm", lambda_reg=2.0):
    model.eval()
    tot_loss, preds, truths, pts_list = 0.0, [], [], []
    for batch in loader:
        if mode == "lstm":
            bh, bc, by, byr, bym, bpos = batch
            bh, bc, by, byr, bym, bpos = bh.to(device), bc.to(device), by.to(device), byr.to(device), bym.to(device), bpos.to(device)
            logits, pts                = model(bh, bc)
        else:
            bf, by, byr, bym, bpos = batch
            bf, by, byr, bym, bpos = bf.to(device), by.to(device), byr.to(device), bym.to(device), bpos.to(device)
            logits, pts            = model(bf)
            
        pw = get_pos_w_tensor(bpos)
        loss_cls = (pw * F.cross_entropy(logits, by, reduction='none')).mean()
        loss_reg = reg_crit(pts, byr, bpos)
        loss     = loss_cls + lambda_reg * loss_reg

        tot_loss += loss.item() * by.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        truths.extend(by.cpu().tolist())
        pts_list.extend(pts.cpu().tolist())
    return tot_loss / len(loader.dataset), preds, truths, np.array(pts_list)

def fit(model, tr_ld, va_ld, epochs, lr=5e-4, wd=1e-4, patience=8, cls_crit=None, mode="lstm"):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    reg_crit = PositionWeightedSmoothL1Loss()
    if cls_crit is None: cls_crit = nn.CrossEntropyLoss(reduction='none')
    
    best_f1, best_state, pat = -1.0, None, 0
    for ep in range(epochs):
        tr_loss = train_step(model, tr_ld, cls_crit, reg_crit, opt, mode=mode)
        va_loss, vp, vt, _ = eval_step(model, va_ld, cls_crit, reg_crit, mode=mode)
        va_f1  = f1_score(vt, vp, average="macro", zero_division=0)
        va_acc = accuracy_score(vt, vp)
        sched.step()
        improved = va_f1 > best_f1
        if improved:
            best_f1    = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        lr_now = opt.param_groups[0]["lr"]
        print(f"  Ep {ep+1:02d}/{epochs} | TrLoss: {tr_loss:.4f} | VaLoss: {va_loss:.4f} | VaAcc: {va_acc*100:.2f}% | VaF1: {va_f1:.4f} | LR: {lr_now:.1e}{' *' if improved else ''}")
        if pat >= patience:
            print(f"  Early stop @ ep {ep+1} (best F1={best_f1:.4f})")
            break
    model.load_state_dict(best_state)
    return model

@torch.no_grad()
def predict_expert2_lstm(model, X_3d, X_ctx, bs=256):
    model.eval()
    ds = TensorDataset(torch.tensor(X_3d), torch.tensor(X_ctx))
    ld = DataLoader(ds, batch_size=bs, shuffle=False)
    probs, pts_arr, mins_arr = [], [], []
    for bh, bc in ld:
        logits, pts, mins = model(bh.to(device), bc.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        pts_arr.extend(pts.cpu().numpy())
        mins_arr.extend(mins.cpu().numpy())
    return np.vstack(probs), np.array(pts_arr), np.array(mins_arr)

@torch.no_grad()
def predict_expert2_mlp(model, X_flat, bs=256):
    model.eval()
    ds = TensorDataset(torch.tensor(X_flat))
    ld = DataLoader(ds, batch_size=bs, shuffle=False)
    probs, pts_arr, mins_arr = [], [], []
    for (bf,) in ld:
        logits, pts, mins = model(bf.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        pts_arr.extend(pts.cpu().numpy())
        mins_arr.extend(mins.cpu().numpy())
    return np.vstack(probs), np.array(pts_arr), np.array(mins_arr)

@torch.no_grad()
def predict_lstm(model, X_3d, X_ctx, bs=256):
    model.eval()
    ds = TensorDataset(torch.tensor(X_3d), torch.tensor(X_ctx))
    ld = DataLoader(ds, batch_size=bs, shuffle=False)
    probs, pts_arr = [], []
    for bh, bc in ld:
        logits, pts = model(bh.to(device), bc.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        pts_arr.extend(pts.cpu().numpy())
    return np.vstack(probs), np.array(pts_arr)

@torch.no_grad()
def predict_mlp(model, X_flat, bs=256):
    model.eval()
    ds = TensorDataset(torch.tensor(X_flat))
    ld = DataLoader(ds, batch_size=bs, shuffle=False)
    probs, pts_arr = [], []
    for (bf,) in ld:
        logits, pts = model(bf.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        pts_arr.extend(pts.cpu().numpy())
    return np.vstack(probs), np.array(pts_arr)

print("\nLoading and engineering dataset with Extended Features & Position Weighting Across ALL Bands...")
df = pd.read_csv(DATA_PATH)
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()
df = engineer_features(df)

def exp2_band(p):
    if p <= 0: return 0
    elif p <= 1: return 1
    else: return 2

def exp1_band_3cls(p):
    if p <= 5: return 0
    elif p <= 9: return 1
    else: return 2

def overall_band_4cls(p):
    if p <= 2: return 0
    elif p <= 5: return 1
    elif p <= 9: return 2
    else: return 3

df["exp2_target"]     = df[TARGET].apply(exp2_band)
df["exp1_target"]     = df[TARGET].apply(exp1_band_3cls)
df["high_return"]     = (df[TARGET] >= 6).astype(int)
df["explosive_return"]= (df[TARGET] >= 10).astype(int)
df["overall_target"]  = df[TARGET].apply(overall_band_4cls)

train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
val_df   = df[df["season"].isin(VAL_SEASONS)].copy()
test_df  = df[df["season"].isin(TEST_SEASONS)].copy()

df_tr2  = train_df[train_df[TARGET] < 3].copy()
df_trh  = train_df[train_df[TARGET] >= 3].copy()

df_va2  = val_df[val_df["current_gate_probability"] < 0.50].copy()
df_vah  = val_df[(val_df["current_gate_probability"] >= 0.50) & (val_df[TARGET] >= 3)].copy()

print(f"Train rows: E2={len(df_tr2):,} | High={len(df_trh):,}")
print(f"Val   rows: E2={len(df_va2):,}   | High={len(df_vah):,}")

print("\n" + "="*80)
print("STAGE 1: TRAINING UPGRADED EXPERT 2 (POSITION WEIGHTED ALL LOSS & FEATS)")
print("="*80)

X_tr2_3d, X_tr2_ctx, X_tr2_flat, y_tr2, yr_tr2, ym_tr2, pos_tr2 = build_matrices(df_tr2, "exp2_target")
X_va2_3d, X_va2_ctx, X_va2_flat, y_va2, yr_va2, ym_va2, pos_va2 = build_matrices(df_va2,   "exp2_target")

sc2_3d   = StandardScaler().fit(X_tr2_3d.reshape(-1, len(HIST_FEATS)))
sc2_ctx  = StandardScaler().fit(X_tr2_ctx)
sc2_flat = RobustScaler().fit(X_tr2_flat)

X_tr2_3ds, X_va2_3ds = sc3d(X_tr2_3d, sc2_3d), sc3d(X_va2_3d, sc2_3d)
X_tr2_cs, X_va2_cs   = sc2_ctx.transform(X_tr2_ctx), sc2_ctx.transform(X_va2_ctx)
X_tr2_fs, X_va2_fs   = sc2_flat.transform(X_tr2_flat), sc2_flat.transform(X_va2_flat)

cw2 = compute_weights(y_tr2, 3)

tr2_ld = DataLoader(TensorDataset(torch.tensor(X_tr2_3ds), torch.tensor(X_tr2_cs), torch.tensor(y_tr2), torch.tensor(yr_tr2), torch.tensor(ym_tr2), torch.tensor(pos_tr2)), batch_size=BATCH_SIZE, shuffle=True)
va2_ld = DataLoader(TensorDataset(torch.tensor(X_va2_3ds), torch.tensor(X_va2_cs), torch.tensor(y_va2), torch.tensor(yr_va2), torch.tensor(ym_va2), torch.tensor(pos_va2)), batch_size=BATCH_SIZE, shuffle=False)

bilstm_e2 = wrap(Expert2BiLSTM(len(HIST_FEATS), len(CTX_FEATS), 3, h=192))
print("\n[BiLSTM Upgraded Expert 2 — Position Weighted All]")
bilstm_e2 = fit_e2(bilstm_e2, tr2_ld, va2_ld, epochs=30, cls_crit=nn.CrossEntropyLoss(weight=cw2, reduction='none'))

tr2_mlp_ld = DataLoader(TensorDataset(torch.tensor(X_tr2_fs), torch.tensor(y_tr2), torch.tensor(yr_tr2), torch.tensor(ym_tr2), torch.tensor(pos_tr2)), batch_size=BATCH_SIZE, shuffle=True)
va2_mlp_ld = DataLoader(TensorDataset(torch.tensor(X_va2_fs), torch.tensor(y_tr2), torch.tensor(yr_tr2), torch.tensor(ym_tr2), torch.tensor(pos_tr2)), batch_size=BATCH_SIZE, shuffle=False)

mlp_e2 = wrap(Expert2MLP(X_tr2_fs.shape[1], 3))
print("\n[MultiTask MLP Upgraded Expert 2 — Position Weighted All]")
mlp_e2 = fit_e2(mlp_e2, tr2_mlp_ld, va2_mlp_ld, epochs=25, cls_crit=nn.CrossEntropyLoss(weight=cw2, reduction='none'), mode="mlp")

print("\n" + "="*80)
print("STAGE 2: TRAINING EXPERT 1 HIGH BAND (BALANCED 3-5 & 6-9 LOSS WEIGHTING)")
print("="*80)

X_trh_3d, X_trh_ctx, X_trh_flat, y_trh_band, yr_trh, _, pos_trh = build_matrices(df_trh, "exp1_target")
X_vah_3d, X_vah_ctx, X_vah_flat, y_vah_band, yr_vah, _, pos_vah = build_matrices(df_vah,   "exp1_target")

y_trh_bin = df_trh["high_return"].values.astype(np.int64)
y_vah_bin = df_vah["high_return"].values.astype(np.int64)

sch_3d   = StandardScaler().fit(X_trh_3d.reshape(-1, len(HIST_FEATS)))
sch_ctx  = StandardScaler().fit(X_trh_ctx)
sch_flat = RobustScaler().fit(X_trh_flat)

X_trh_3ds, X_vah_3ds = sc3d(X_trh_3d, sch_3d), sc3d(X_vah_3d, sch_3d)
X_trh_cs, X_vah_cs   = sch_ctx.transform(X_trh_ctx), sch_ctx.transform(X_vah_ctx)
X_trh_fs, X_vah_fs   = sch_flat.transform(X_trh_flat), sch_flat.transform(X_vah_flat)

cw3 = compute_weights(y_trh_band, 3) * torch.tensor([1.10, 1.15, 1.30], device=device)
cw_bin = compute_weights(y_trh_bin, 2)

trh_ld3 = DataLoader(TensorDataset(torch.tensor(X_trh_3ds), torch.tensor(X_trh_cs), torch.tensor(y_trh_band), torch.tensor(yr_trh), torch.tensor(yr_trh), torch.tensor(pos_trh)), batch_size=BATCH_SIZE, shuffle=True)
vah_ld3 = DataLoader(TensorDataset(torch.tensor(X_vah_3ds), torch.tensor(X_vah_cs), torch.tensor(y_vah_band), torch.tensor(yr_vah), torch.tensor(yr_vah), torch.tensor(pos_vah)), batch_size=BATCH_SIZE, shuffle=False)

bilstm_3cls = wrap(MultiTaskBiLSTM(len(HIST_FEATS), len(CTX_FEATS), 3, h=256))
print("\n[Strategy 1] MultiTask 3-Class BiLSTM — Balanced Loss")
bilstm_3cls = fit(bilstm_3cls, trh_ld3, vah_ld3, epochs=35, lr=5e-4, cls_crit=nn.CrossEntropyLoss(weight=cw3, label_smoothing=0.02, reduction='none'))

trh_ldb = DataLoader(TensorDataset(torch.tensor(X_trh_3ds), torch.tensor(X_trh_cs), torch.tensor(y_trh_bin), torch.tensor(yr_trh), torch.tensor(yr_trh), torch.tensor(pos_trh)), batch_size=BATCH_SIZE, shuffle=True)
vah_ldb = DataLoader(TensorDataset(torch.tensor(X_vah_3ds), torch.tensor(X_vah_cs), torch.tensor(y_vah_bin), torch.tensor(yr_trh), torch.tensor(yr_trh), torch.tensor(pos_trh)), batch_size=BATCH_SIZE, shuffle=False)

bilstm_bin = wrap(MultiTaskBiLSTM(len(HIST_FEATS), len(CTX_FEATS), 2, h=128))
print("\n[Strategy 2] Binary BiLSTM — Position Weighted All")
bilstm_bin = fit(bilstm_bin, trh_ldb, vah_ldb, epochs=30, cls_crit=nn.CrossEntropyLoss(weight=cw_bin, reduction='none'))

trh_mlpb = DataLoader(TensorDataset(torch.tensor(X_trh_fs), torch.tensor(y_trh_bin), torch.tensor(yr_trh), torch.tensor(yr_trh), torch.tensor(pos_trh)), batch_size=BATCH_SIZE, shuffle=True)
vah_mlpb = DataLoader(TensorDataset(torch.tensor(X_vah_fs), torch.tensor(y_trh_bin), torch.tensor(yr_trh), torch.tensor(yr_trh), torch.tensor(pos_trh)), batch_size=BATCH_SIZE, shuffle=False)

mlp_bin = wrap(MultiTaskMLP(X_trh_fs.shape[1], 2))
print("\n[Strategy 2] Binary MLP — Position Weighted All")
mlp_bin = fit(mlp_bin, trh_mlpb, vah_mlpb, epochs=25, cls_crit=nn.CrossEntropyLoss(weight=cw_bin, reduction='none'), mode="mlp")

print("\n" + "="*95)
print("FINAL UNIFIED SYSTEM EVALUATION REPORT — 2025-26 TEST SET")
print("="*95)

X_te_3d, X_te_ctx, X_te_flat, y_te_overall, yr_te, ym_te, pos_te = build_matrices(test_df, "overall_target")
gate_prob = test_df["current_gate_probability"].values

p2_lstm, pts2_l, mins2_l = predict_expert2_lstm(bilstm_e2, sc3d(X_te_3d, sc2_3d), sc2_ctx.transform(X_te_ctx))
p2_mlp,  pts2_m, mins2_m = predict_expert2_mlp(mlp_e2, sc2_flat.transform(X_te_flat))

p2_bl    = 0.55 * p2_lstm + 0.45 * p2_mlp
pts2     = 0.55 * pts2_l  + 0.45 * pts2_m
pred_m2  = 0.55 * mins2_l + 0.45 * mins2_m

p_3cls, pts_3c   = predict_lstm(bilstm_3cls, sc3d(X_te_3d, sch_3d), sch_ctx.transform(X_te_ctx))

gw1_mins_col = "gw_minus_1_minutes"
gw1_mins     = pd.to_numeric(test_df[gw1_mins_col], errors="coerce").fillna(0).values if gw1_mins_col in test_df.columns else np.zeros(len(test_df))

gw1_start_col = "gw_minus_1_started"
gw1_start     = pd.to_numeric(test_df[gw1_start_col], errors="coerce").fillna(0).values if gw1_start_col in test_df.columns else np.zeros(len(test_df))

xgc_col   = "gw_minus_1_expected_goals_conceded"
xgc_vals  = pd.to_numeric(test_df[xgc_col], errors="coerce").fillna(0).values if xgc_col in test_df.columns else np.zeros(len(test_df))

combined_preds_band = np.zeros(len(test_df), dtype=int)
combined_pred_pts   = np.zeros(len(test_df), dtype=np.float32)

anchors = np.array([3.8, 7.2, 11.5], dtype=np.float32)

for i in range(len(test_df)):
    if gate_prob[i] < 0.50:
        combined_preds_band[i] = 0
        if pred_m2[i] < 1.0 or (gw1_mins[i] == 0 and gw1_start[i] == 0):
            combined_pred_pts[i] = 0.0
        elif pred_m2[i] < 59.5:
            combined_pred_pts[i] = 1.0
        else:
            base_p = 2.0
            if pos_te[i] in [1, 2] and xgc_vals[i] >= 1.8:
                base_p = 1.0
            combined_pred_pts[i] = 0.70 * base_p + 0.30 * np.clip(pts2[i], 0.0, 2.4)
    else:
        probs = p_3cls[i]
        top_cls = probs.argmax()
        combined_preds_band[i] = top_cls + 1
        soft_val = np.sum(probs * anchors)
        combined_pred_pts[i] = 0.65 * soft_val + 0.35 * max(pts_3c[i], 3.0)

cum_mae  = mean_absolute_error(yr_te, combined_pred_pts)
cum_rmse = np.sqrt(mean_squared_error(yr_te, combined_pred_pts))
cum_acc  = accuracy_score(y_te_overall, combined_preds_band) * 100
cum_prec = precision_score(y_te_overall, combined_preds_band, average="macro", zero_division=0) * 100
cum_rec  = recall_score(y_te_overall, combined_preds_band, average="macro", zero_division=0) * 100
cum_f1   = f1_score(y_te_overall, combined_preds_band, average="macro", zero_division=0)
rho, _   = spearmanr(yr_te, combined_pred_pts)

diff = np.abs(yr_te - combined_pred_pts)
pct_pm1 = (diff <= 1.0).mean() * 100
pct_pm2 = (diff <= 2.0).mean() * 100

print("\n" + "="*80)
print("1. OVERALL SYSTEM CUMULATIVE METRICS (ENTIRE DATASET — 0 to 10+ POINTS)")
print("="*80)
print(f"Total Test Samples Evaluated:        {len(test_df):,}")
print(f"--------------------------------------------------")
print(f"  Cumulative MAE:                   {cum_mae:.4f}  (XGBoost Benchmark: 1.269)")
print(f"  Cumulative RMSE:                  {cum_rmse:.4f}  (XGBoost Benchmark: 2.107)")
print(f"  Spearman Rank Correlation:        {rho:.4f}  (XGBoost Benchmark: 0.795)")
print(f"  Cumulative Classification Acc:    {cum_acc:.2f}%")
print(f"  Cumulative Macro Precision:       {cum_prec:.2f}%")
print(f"  Cumulative Macro Recall:          {cum_rec:.2f}%")
print(f"  Cumulative Macro F1 Score:        {cum_f1:.4f}")
print(f"--------------------------------------------------")
print(f"  Predictions within ±1 FPL point:  {pct_pm1:.2f}%")
print(f"  Predictions within ±2 FPL points: {pct_pm2:.2f}%")

print("\n" + "="*80)
print("2. INDIVIDUAL SCORE BAND DETAILED BREAKDOWN (4 BANDS: 0-2, 3-5, 6-9, 10+)")
print("="*80)
print(f"{'Band':<10} | {'Samples':<8} | {'MAE':<8} | {'RMSE':<8} | {'Accuracy (%)':<14} | {'Precision (%)':<15} | {'Recall (%)':<12} | {'F1':<8}")
print("-" * 95)

for b_idx, b_name in enumerate(BAND_NAMES_ALL):
    mask_b = (y_te_overall == b_idx)
    cnt_b  = mask_b.sum()
    if cnt_b == 0: continue
    
    y_actual_b = yr_te[mask_b]
    y_pred_b   = combined_pred_pts[mask_b]
    mae_b      = mean_absolute_error(y_actual_b, y_pred_b)
    rmse_b     = np.sqrt(mean_squared_error(y_actual_b, y_pred_b))
    
    acc_b  = (combined_preds_band[mask_b] == b_idx).mean() * 100
    prec_b = precision_score(y_te_overall == b_idx, combined_preds_band == b_idx, zero_division=0) * 100
    rec_b  = recall_score(y_te_overall == b_idx, combined_preds_band == b_idx, zero_division=0) * 100
    f1_b   = f1_score(y_te_overall == b_idx, combined_preds_band == b_idx, zero_division=0)

    print(f"{b_name:<10} | {cnt_b:<8,} | {mae_b:<8.4f} | {rmse_b:<8.4f} | {acc_b:<14.2f}% | {prec_b:<15.2f}% | {rec_b:<12.2f}% | {f1_b:<8.4f}")

print("\n" + "="*80)
print("3. EXPERT STANDALONE METRICS & CONFIDENCE TABLES")
print("="*80)

mask_e2  = (test_df[TARGET] < 3).values
e2_acc   = accuracy_score(test_df.loc[mask_e2, "exp2_target"], p2_bl[mask_e2].argmax(1)) * 100
print(f"\n[Upgraded Expert 2 Standalone (0-2 pts)]  Accuracy: {e2_acc:.2f}%  | Samples: {mask_e2.sum():,}")

print("\n--- Confidence Precision & Recall Tables ---")
mask_e1 = (test_df[TARGET] >= 3).values
y_te_e1 = test_df.loc[mask_e1, "exp1_target"].values

for b_idx in range(3):
    b_name = BAND_NAMES_E1[b_idx]
    print(f"\n[Score Band: {b_name}]")
    print(f"{'Min Threshold':<15} | {'Predictions':<13} | {'Correct':<10} | {'Precision (%)':<15} | {'Recall (%)':<12}")
    print("-" * 70)
    actual  = (y_te_e1 == b_idx)
    act_cnt = actual.sum()
    for t in [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        pred  = (p_3cls[mask_e1, b_idx] >= t)
        cnt   = pred.sum()
        corr  = (actual & pred).sum()
        prec_t= (corr / cnt * 100) if cnt > 0 else 0.0
        rec_t = (corr / act_cnt * 100) if act_cnt > 0 else 0.0
        print(f"P >= {t:<10.2f} | {cnt:<13} | {corr:<10} | {prec_t:<15.2f}% | {rec_t:<12.2f}%")

print("\n" + "="*95)
print("FINAL EXTENDED FEATURE & POSITION-WEIGHTED 4-BAND EVALUATION COMPLETE")
print("="*95)