import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

from shared_config import (
    device, TEST_SEASONS, TARGET, HIST_FEATS, CTX_FEATS,
    get_paths, engineer_features, build_matrices, sc3d,
    MultiTaskBiLSTM, MultiTaskMLP, predict_lstm, predict_mlp
)

def main():
    print("=" * 80)
    print("EXPERT 1 HIGH BAND — INFERENCE & PREDICTIONS RUNNER")
    print("=" * 80)

    DATA_PATH, OUT_BASE, MODEL_DIR, SCALER_DIR = get_paths()
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df = engineer_features(df)

    def exp1_band_unified(p):
        if p <= 4: return 0
        elif p <= 6: return 1
        else: return 2

    df["exp1_target"] = df[TARGET].apply(exp1_band_unified)
    test_df = df[df["season"].isin(TEST_SEASONS)].copy()

    with open(os.path.join(SCALER_DIR, "expert1_high_scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)

    X_te_3d, X_te_ctx, X_te_flat, y_te, yr_te, _, pos_te = build_matrices(test_df, "exp1_target")

    bilstm_3cls = MultiTaskBiLSTM(len(HIST_FEATS), len(CTX_FEATS), 3, h=192).to(device)
    bilstm_3cls.load_state_dict(torch.load(os.path.join(MODEL_DIR, "expert1_high_3cls.pt"), map_location=device))

    p_3cls, pts_3c = predict_lstm(bilstm_3cls, sc3d(X_te_3d, scalers["scaler_3d"]), scalers["scaler_ctx"].transform(X_te_ctx))

    test_df["pred_high_cls"] = p_3cls.argmax(axis=1)
    test_df["pred_high_pts"] = pts_3c

    out_csv = os.path.join(OUT_BASE, "expert1_high_predictions_test.csv")
    test_df.to_csv(out_csv, index=False)
    print(f"Expert 1 High predictions saved to: {out_csv}")

if __name__ == "__main__":
    main()
