import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler

from shared_config import (
    device, TRAIN_SEASONS, VAL_SEASONS, TARGET,
    HIST_FEATS, CTX_FEATS, BATCH_SIZE, get_paths,
    engineer_features, build_matrices, sc3d, compute_weights,
    MultiTaskBiLSTM, MultiTaskMLP, fit, wrap
)

def main():
    print("=" * 80)
    print("STAGE 2: TRAINING EXPERT 1 HIGH BAND (BALANCED 3-5 & 6-9 FOCUS)")
    print("=" * 80)

    DATA_PATH, OUT_BASE, MODEL_DIR, SCALER_DIR = get_paths()
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df = engineer_features(df)

    def exp1_band_3cls(p):
        if p <= 5: return 0
        elif p <= 9: return 1
        else: return 2

    df["exp1_target"] = df[TARGET].apply(exp1_band_3cls)
    df["high_return"] = (df[TARGET] >= 6).astype(int)

    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val_df   = df[df["season"].isin(VAL_SEASONS)].copy()

    df_trh = train_df[train_df[TARGET] >= 3].copy()
    df_vah = val_df[(val_df["current_gate_probability"] >= 0.50) & (val_df[TARGET] >= 3)].copy()

    print(f"Train High rows: {len(df_trh):,} | Val High rows: {len(df_vah):,}")

    X_trh_3d, X_trh_ctx, X_trh_flat, y_trh_band, yr_trh, _, pos_trh = build_matrices(df_trh, "exp1_target")
    X_vah_3d, X_vah_ctx, X_vah_flat, y_vah_band, yr_vah, _, pos_vah = build_matrices(df_vah, "exp1_target")

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
    vah_ldb = DataLoader(TensorDataset(torch.tensor(X_vah_3ds), torch.tensor(X_vah_cs), torch.tensor(y_vah_bin), torch.tensor(yr_vah), torch.tensor(yr_vah), torch.tensor(pos_vah)), batch_size=BATCH_SIZE, shuffle=False)

    bilstm_bin = wrap(MultiTaskBiLSTM(len(HIST_FEATS), len(CTX_FEATS), 2, h=128))
    print("\n[Strategy 2] Binary BiLSTM — Position Weighted All")
    bilstm_bin = fit(bilstm_bin, trh_ldb, vah_ldb, epochs=30, cls_crit=nn.CrossEntropyLoss(weight=cw_bin, reduction='none'))

    trh_mlpb = DataLoader(TensorDataset(torch.tensor(X_trh_fs), torch.tensor(y_trh_bin), torch.tensor(yr_trh), torch.tensor(yr_trh), torch.tensor(pos_trh)), batch_size=BATCH_SIZE, shuffle=True)
    vah_mlpb = DataLoader(TensorDataset(torch.tensor(X_vah_fs), torch.tensor(y_vah_bin), torch.tensor(yr_vah), torch.tensor(yr_vah), torch.tensor(pos_vah)), batch_size=BATCH_SIZE, shuffle=False)

    mlp_bin = wrap(MultiTaskMLP(X_trh_fs.shape[1], 2))
    print("\n[Strategy 2] Binary MLP — Position Weighted All")
    mlp_bin = fit(mlp_bin, trh_mlpb, vah_mlpb, epochs=25, cls_crit=nn.CrossEntropyLoss(weight=cw_bin, reduction='none'), mode="mlp")

    raw_3cls = bilstm_3cls.module if hasattr(bilstm_3cls, "module") else bilstm_3cls
    raw_bbin = bilstm_bin.module if hasattr(bilstm_bin, "module") else bilstm_bin
    raw_mbin = mlp_bin.module if hasattr(mlp_bin, "module") else mlp_bin

    torch.save(raw_3cls.state_dict(), os.path.join(MODEL_DIR, "expert1_3cls.pt"))
    torch.save(raw_bbin.state_dict(), os.path.join(MODEL_DIR, "expert1_bbin.pt"))
    torch.save(raw_mbin.state_dict(), os.path.join(MODEL_DIR, "expert1_mbin.pt"))

    with open(os.path.join(SCALER_DIR, "expert1_scalers.pkl"), "wb") as f:
        pickle.dump({"scaler_3d": sch_3d, "scaler_ctx": sch_ctx, "scaler_flat": sch_flat}, f)

    print(f"Expert 1 models saved to: {MODEL_DIR}")

if __name__ == "__main__":
    main()
