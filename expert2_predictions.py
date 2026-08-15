
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

from shared_config import (
    device, TEST_SEASONS, TARGET, HIST_FEATS, CTX_FEATS,
    get_paths, engineer_features, build_matrices, sc3d,
    Expert2BiLSTM, Expert2MLP, predict_expert2_lstm, predict_expert2_mlp
)

def main():
    print("=" * 80)
    print("EXPERT 2 — INFERENCE & PREDICTIONS RUNNER")
    print("=" * 80)

    DATA_PATH, OUT_BASE, MODEL_DIR, SCALER_DIR = get_paths()
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df = engineer_features(df)

    def exp2_band(p):
        if p <= 0: return 0
        elif p <= 1: return 1
        else: return 2

    df["exp2_target"] = df[TARGET].apply(exp2_band)
    test_df = df[df["season"].isin(TEST_SEASONS)].copy()

    with open(os.path.join(SCALER_DIR, "expert2_scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)

    X_te_3d, X_te_ctx, X_te_flat, y_te, yr_te, ym_te, pos_te = build_matrices(test_df, "exp2_target")

    bilstm_e2 = Expert2BiLSTM(len(HIST_FEATS), len(CTX_FEATS), 3, h=192).to(device)
    bilstm_e2.load_state_dict(torch.load(os.path.join(MODEL_DIR, "expert2_bilstm.pt"), map_location=device))

    mlp_e2 = Expert2MLP(X_te_flat.shape[1], 3).to(device)
    mlp_e2.load_state_dict(torch.load(os.path.join(MODEL_DIR, "expert2_mlp.pt"), map_location=device))

    p2_lstm, pts2_l, mins2_l = predict_expert2_lstm(bilstm_e2, sc3d(X_te_3d, scalers["scaler_3d"]), scalers["scaler_ctx"].transform(X_te_ctx))
    p2_mlp,  pts2_m, mins2_m = predict_expert2_mlp(mlp_e2, scalers["scaler_flat"].transform(X_te_flat))

    p2_blend = 0.55 * p2_lstm + 0.45 * p2_mlp
    pts2     = 0.55 * pts2_l  + 0.45 * pts2_m
    mins2    = 0.55 * mins2_l + 0.45 * mins2_m

    test_df["pred_exp2_cls"]  = p2_blend.argmax(axis=1)
    test_df["pred_exp2_pts"]  = pts2
    test_df["pred_exp2_mins"] = mins2

    out_csv = os.path.join(OUT_BASE, "expert2_predictions_test.csv")
    test_df.to_csv(out_csv, index=False)
    print(f"Expert 2 predictions saved to: {out_csv}")

if __name__ == "__main__":
    main()
