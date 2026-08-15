"""
shared_config.py
================
Shared configuration, feature lists, multi-stage multi-task neural network architectures,
data preprocessing functions, and training/evaluation utilities for the Dual Expert Gate Architecture.
"""

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

# ── Reproducibility & Device ──────────────────────────────────────────
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

# ── Configuration Constants ───────────────────────────────────────────
WINDOW_SIZE   = 5
BATCH_SIZE    = 128
TARGET        = "total_points"
TRAIN_SEASONS = ["2016-17","2017-18","2018-19","2019-20","2020-21","2021-22","2022-23","2023-24"]
VAL_SEASONS   = ["2024-25"]
TEST_SEASONS  = ["2025-26"]

POS_WEIGHT_MAP = {1: 1.35, 2: 1.30, 3: 1.15, 4: 1.10}

BAND_NAMES_E1  = ["3-5", "6-9", "10+"]
BAND_NAMES_ALL = ["0-2 pts", "3-5 pts", "6-9 pts", "10+ pts"]

# Full Per-Step Features (216 per GW step + Engineered)
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

# ── Paths ─────────────────────────────────────────────────────────────
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
