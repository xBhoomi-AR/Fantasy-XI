import os
import sys
import random
import pickle
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
    mean_absolute_error, mean_squared_error
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count()

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

def get_paths(csv_filename="player_5gw_sequences.csv"):
    in_kaggle = os.path.exists("/kaggle/input")
    in_colab  = "google.colab" in sys.modules or os.path.exists("/content")

    if in_kaggle:
        import glob
        matches = glob.glob(f"/kaggle/input/**/{csv_filename}", recursive=True)
        if not matches:
            raise FileNotFoundError(f"Cannot find {csv_filename} in /kaggle/input")
        data_path = matches[0]
        out_base  = "/kaggle/working"
    elif in_colab:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        data_path = f"/content/drive/MyDrive/Fantasy_XI/expert/datasets/{csv_filename}"
        out_base  = "/content/drive/MyDrive/Fantasy_XI/expert"
    else:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(root, "expert", "datasets", csv_filename)
        out_base  = os.path.join(root, "expert")

    model_dir  = os.path.join(out_base, "models",  "gate_architecture")
    scaler_dir = os.path.join(out_base, "scalers", "gate_architecture")
    for d in [model_dir, scaler_dir]:
        os.makedirs(d, exist_ok=True)

    return data_path, out_base, model_dir, scaler_dir

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
        xa        = pd.to_numeric(df[xa_col], errors="coerce").fillna(0) if xxa_col in df.columns else pd.Series(0, index=df.index) if 'xxa_col' in locals() else pd.to_numeric(df[xa_col], errors="coerce").fillna(0) if xa_col in df.columns else pd.Series(0, index=df.index)
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
