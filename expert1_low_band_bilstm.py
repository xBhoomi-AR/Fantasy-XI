import os
import sys
import pickle
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler

from shared_config import (
    device, TRAIN_SEASONS, VAL_SEASONS, TARGET,
    HIST_FEATS, CTX_FEATS, BATCH_SIZE, get_paths,
    engineer_features, build_matrices, sc3d, compute_weights,
    MultiTaskBiLSTM, fit, wrap
)

def main():
    print("=" * 80)
    print("STAGE 2: TRAINING EXPERT 1 LOW BAND (3-6 POINTS: POSITION WEIGHTED)")
    print("=" * 80)

    DATA_PATH, OUT_BASE, MODEL_DIR, SCALER_DIR = get_paths()
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df = engineer_features(df)

    def low_band(p):
        return 0 if p <= 4 else 1

    df["low_band_target"] = df[TARGET].apply(low_band)

    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val_df   = df[df["season"].isin(VAL_SEASONS)].copy()

    train_low = train_df[(train_df[TARGET] >= 3) & (train_df[TARGET] <= 6)].copy()
    val_low   = val_df[(val_df["current_gate_probability"] >= 0.50) & (val_df[TARGET] >= 3) & (val_df[TARGET] <= 6)].copy()

    print(f"Train Low rows: {len(train_low):,} | Val Low rows: {len(val_low):,}")

    X_trl_3d, X_trl_ctx, X_trl_flat, y_trl, yr_trl, _, pos_trl = build_matrices(train_low, "low_band_target")
    X_val_3d, X_val_ctx, X_val_flat, y_val, yr_val, _, pos_val  = build_matrices(val_low,   "low_band_target")

    scl_3d   = StandardScaler().fit(X_trl_3d.reshape(-1, len(HIST_FEATS)))
    scl_ctx  = StandardScaler().fit(X_trl_ctx)
    scl_flat = RobustScaler().fit(X_trl_flat)

    X_trl_3ds, X_val_3ds = sc3d(X_trl_3d, scl_3d), sc3d(X_val_3d, scl_3d)
    X_trl_cs, X_val_cs   = scl_ctx.transform(X_trl_ctx), scl_ctx.transform(X_val_ctx)
    X_trl_fs, X_val_fs   = scl_flat.transform(X_trl_flat), scl_flat.transform(X_val_flat)

    cwl = compute_weights(y_trl, 2)

    trl_ld = DataLoader(TensorDataset(torch.tensor(X_trl_3ds), torch.tensor(X_trl_cs), torch.tensor(y_trl), torch.tensor(yr_trl), torch.tensor(yr_trl), torch.tensor(pos_trl)), batch_size=BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(TensorDataset(torch.tensor(X_val_3ds), torch.tensor(X_val_cs), torch.tensor(y_val), torch.tensor(yr_val), torch.tensor(yr_val), torch.tensor(pos_val)), batch_size=BATCH_SIZE, shuffle=False)

    bilstm_low = wrap(MultiTaskBiLSTM(len(HIST_FEATS), len(CTX_FEATS), 2, h=192))
    print("\n[BiLSTM Expert 1 Low]")
    bilstm_low = fit(bilstm_low, trl_ld, val_ld, epochs=35, lr=5e-4, cls_crit=nn.CrossEntropyLoss(weight=cwl))

    # Save models and scalers
    raw_bilstm = bilstm_low.module if hasattr(bilstm_low, "module") else bilstm_low
    torch.save(raw_bilstm.state_dict(), os.path.join(MODEL_DIR, "expert1_low_bilstm.pt"))

    with open(os.path.join(SCALER_DIR, "expert1_low_scalers.pkl"), "wb") as f:
        pickle.dump({"scaler_3d": scl_3d, "scaler_ctx": scl_ctx, "scaler_flat": scl_flat}, f)

    print(f"Expert 1 Low model saved to: {MODEL_DIR}")

if __name__ == "__main__":
    main()
