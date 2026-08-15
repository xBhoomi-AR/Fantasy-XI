import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error
)

from shared_config import (
    device, TEST_SEASONS, TARGET, HIST_FEATS, CTX_FEATS,
    get_paths, engineer_features, build_matrices, sc3d,
    Expert2BiLSTM, Expert2MLP, MultiTaskBiLSTM, MultiTaskMLP,
    predict_expert2_lstm, predict_expert2_mlp, predict_lstm, predict_mlp
)

def main():
    print("\n" + "="*95)
    print("FINAL UNIFIED SYSTEM EVALUATION REPORT — 2025-26 TEST SET")
    print("="*95)

    DATA_PATH, OUT_BASE, MODEL_DIR, SCALER_DIR = get_paths()
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df = engineer_features(df)

    def exp2_band(p):
        if p <= 0: return 0
        elif p <= 1: return 1
        else: return 2

    def low_band(p): return 0 if p <= 4 else 1

    def exp1_band_unified(p):
        if p <= 4: return 0      # 3-4 pts
        elif p <= 6: return 1    # 5-6 pts
        else: return 2           # 7-10+ pts High Return Band

    def overall_band_unified(p):
        if p <= 2: return 0      # 0-2 pts (Expert 2)
        elif p <= 4: return 1    # 3-4 pts
        elif p <= 6: return 2    # 5-6 pts
        else: return 3           # 7-10+ pts

    df["exp2_target"]    = df[TARGET].apply(exp2_band)
    df["low_band_target"]= df[TARGET].apply(low_band)
    df["exp1_target"]    = df[TARGET].apply(exp1_band_unified)
    df["high_return"]    = (df[TARGET] >= 7).astype(int)
    df["overall_target"] = df[TARGET].apply(overall_band_unified)

    test_df  = df[df["season"].isin(TEST_SEASONS)].copy()

    # Load scalers
    with open(os.path.join(SCALER_DIR, "expert2_scalers.pkl"), "rb") as f:
        sc2 = pickle.load(f)
    with open(os.path.join(SCALER_DIR, "expert1_low_scalers.pkl"), "rb") as f:
        scl = pickle.load(f)
    with open(os.path.join(SCALER_DIR, "expert1_high_scalers.pkl"), "rb") as f:
        sch = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "expert1_high_thresh.pkl"), "rb") as f:
        opt_thresh = pickle.load(f)["opt_thresh"]

    X_te_3d, X_te_ctx, X_te_flat, y_te_overall, yr_te, ym_te, pos_te = build_matrices(test_df, "overall_target")
    gate_prob = test_df["current_gate_probability"].values

    # Load models
    bilstm_e2 = Expert2BiLSTM(len(HIST_FEATS), len(CTX_FEATS), 3, h=192).to(device)
    bilstm_e2.load_state_dict(torch.load(os.path.join(MODEL_DIR, "expert2_bilstm.pt"), map_location=device))

    mlp_e2 = Expert2MLP(X_te_flat.shape[1], 3).to(device)
    mlp_e2.load_state_dict(torch.load(os.path.join(MODEL_DIR, "expert2_mlp.pt"), map_location=device))

    bilstm_low = MultiTaskBiLSTM(len(HIST_FEATS), len(CTX_FEATS), 2, h=192).to(device)
    bilstm_low.load_state_dict(torch.load(os.path.join(MODEL_DIR, "expert1_low_bilstm.pt"), map_location=device))

    bilstm_3cls = MultiTaskBiLSTM(len(HIST_FEATS), len(CTX_FEATS), 3, h=192).to(device)
    bilstm_3cls.load_state_dict(torch.load(os.path.join(MODEL_DIR, "expert1_high_3cls.pt"), map_location=device))

    bilstm_bin = MultiTaskBiLSTM(len(HIST_FEATS), len(CTX_FEATS), 2, h=128).to(device)
    bilstm_bin.load_state_dict(torch.load(os.path.join(MODEL_DIR, "expert1_high_bbin.pt"), map_location=device))

    mlp_bin = MultiTaskMLP(X_te_flat.shape[1], 2).to(device)
    mlp_bin.load_state_dict(torch.load(os.path.join(MODEL_DIR, "expert1_high_mbin.pt"), map_location=device))

    # Inference predictions
    p2_lstm, pts2_l, mins2_l = predict_expert2_lstm(bilstm_e2, sc3d(X_te_3d, sc2["scaler_3d"]), sc2["scaler_ctx"].transform(X_te_ctx))
    p2_mlp,  pts2_m, mins2_m = predict_expert2_mlp(mlp_e2, sc2["scaler_flat"].transform(X_te_flat))

    p2_bl    = 0.55 * p2_lstm + 0.45 * p2_mlp
    pts2     = 0.55 * pts2_l  + 0.45 * pts2_m
    pred_m2  = 0.55 * mins2_l + 0.45 * mins2_m

    p_low_bl, pts_low = predict_lstm(bilstm_low, sc3d(X_te_3d, scl["scaler_3d"]), scl["scaler_ctx"].transform(X_te_ctx))
    p_3cls, pts_3c    = predict_lstm(bilstm_3cls, sc3d(X_te_3d, sch["scaler_3d"]), sch["scaler_ctx"].transform(X_te_ctx))

    gw1_mins_col = "gw_minus_1_minutes"
    gw1_mins     = pd.to_numeric(test_df[gw1_mins_col], errors="coerce").fillna(0).values if gw1_mins_col in test_df.columns else np.zeros(len(test_df))

    gw1_start_col = "gw_minus_1_started"
    gw1_start     = pd.to_numeric(test_df[gw1_start_col], errors="coerce").fillna(0).values if gw1_start_col in test_df.columns else np.zeros(len(test_df))

    xgc_col   = "gw_minus_1_expected_goals_conceded"
    xgc_vals  = pd.to_numeric(test_df[xgc_col], errors="coerce").fillna(0).values if xgc_col in test_df.columns else np.zeros(len(test_df))

    combined_preds_band = np.zeros(len(test_df), dtype=int)
    combined_pred_pts   = np.zeros(len(test_df), dtype=np.float32)

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
            e1_cls = p_3cls[i].argmax()
            combined_preds_band[i] = e1_cls + 1
            if e1_cls == 0:
                combined_pred_pts[i] = np.clip(pts_3c[i], 3.0, 4.4)
            elif e1_cls == 1:
                combined_pred_pts[i] = np.clip(pts_3c[i], 5.0, 6.4)
            else:
                combined_pred_pts[i] = max(pts_3c[i], 7.0)

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
    print("2. INDIVIDUAL SCORE BAND DETAILED BREAKDOWN")
    print("="*80)
    print(f"{'Band':<10} | {'Samples':<8} | {'MAE':<8} | {'RMSE':<8} | {'Accuracy (%)':<14} | {'Precision (%)':<15} | {'Recall (%)':<12} | {'F1':<8}")
    print("-" * 95)

    BAND_NAMES_ALL = ["0-2 pts", "3-4 pts", "5-6 pts", "7-10+ pts"]

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

    mask_low = ((test_df[TARGET] >= 3) & (test_df[TARGET] <= 6)).values
    low_acc  = accuracy_score(test_df.loc[mask_low, "low_band_target"], p_low_bl[mask_low].argmax(1)) * 100
    print(f"[Expert 1 Low Standalone (3-6 pts)] Accuracy: {low_acc:.2f}%  | Samples: {mask_low.sum():,}")

    print("\n--- Confidence Precision & Recall Tables ---")
    BAND_NAMES_E1 = ["3-4", "5-6", "7-10+"]
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
    print("FINAL EVALUATION COMPLETE")
    print("="*95)

if __name__ == "__main__":
    main()
