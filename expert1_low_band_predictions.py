

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

from shared_config import (
    device, TEST_SEASONS, TARGET, HIST_FEATS, CTX_FEATS,
    get_paths, engineer_features, build_matrices, sc3d,
    MultiTaskBiLSTM, predict_lstm
)

def main():
    print("=" * 80)
    print("EXPERT 1 LOW BAND — INFERENCE & PREDICTIONS RUNNER")
    print("=" * 80)

    DATA_PATH, OUT_BASE, MODEL_DIR, SCALER_DIR = get_paths()
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df = engineer_features(df)

    def low_band(p): return 0 if p <= 4 else 1

    df["low_band_target"] = df[TARGET].apply(low_band)
    test_df = df[df["season"].isin(TEST_SEASONS)].copy()

    with open(os.path.join(SCALER_DIR, "expert1_low_scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)

    X_te_3d, X_te_ctx, X_te_flat, y_te, yr_te, _, pos_te = build_matrices(test_df, "low_band_target")

    bilstm_low = MultiTaskBiLSTM(len(HIST_FEATS), len(CTX_FEATS), 2, h=192).to(device)
    bilstm_low.load_state_dict(torch.load(os.path.join(MODEL_DIR, "expert1_low_bilstm.pt"), map_location=device))

    p_low_bl, pts_low = predict_lstm(bilstm_low, sc3d(X_te_3d, scalers["scaler_3d"]), scalers["scaler_ctx"].transform(X_te_ctx))

    test_df["pred_low_cls"] = p_low_bl.argmax(axis=1)
    test_df["pred_low_pts"] = pts_low

    out_csv = os.path.join(OUT_BASE, "expert1_low_predictions_test.csv")
    test_df.to_csv(out_csv, index=False)
    print(f"Expert 1 Low predictions saved to: {out_csv}")

if __name__ == "__main__":
    main()
