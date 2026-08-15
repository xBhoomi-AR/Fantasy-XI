import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score

from shared_config import (
    device, TRAIN_SEASONS, VAL_SEASONS, TARGET,
    HIST_FEATS, CTX_FEATS, BATCH_SIZE, get_paths,
    engineer_features, build_matrices, sc3d, compute_weights,
    MultiTaskBiLSTM, MultiTaskMLP, fit, predict_lstm, predict_mlp, wrap
)

def main():
    print("=" * 80)
    print("STAGE 3: TRAINING EXPERT 1 HIGH BAND (UNIFIED 7-10+ HIGH RETURN)")
    print("=" * 80)

    DATA_PATH, OUT_BASE, MODEL_DIR, SCALER_DIR = get_paths()
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df = engineer_features(df)

    def exp1_band_unified(p):
        if p <= 4: return 0      # 3-4 pts
        elif p <= 6: return 1    # 5-6 pts
        else: return 2           # 7-10+ pts High Return Band

    df["exp1_target"] = df[TARGET].apply(exp1_band_unified)
    df["high_return"] = (df[TARGET] >= 7).astype(int)

    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val_df   = df[df["season"].isin(VAL_SEASONS)].copy()

    train_e1h = train_df[train_df[TARGET] >= 3].copy()
    val_e1h   = val_df[(val_df["current_gate_probability"] >= 0.50) & (val_df[TARGET] >= 3)].copy()

    print(f"Train High rows: {len(train_e1h):,} | Val High rows: {len(val_e1h):,}")

    X_trh_3d, X_trh_ctx, X_trh_flat, y_trh_band, yr_trh, _, pos_trh = build_matrices(train_e1h, "exp1_target")
    X_vah_3d, X_vah_ctx, X_vah_flat, y_vah_band, yr_vah, _, pos_vah = build_matrices(val_e1h,   "exp1_target")
    y_trh_bin = train_e1h["high_return"].values.astype(np.int64)
    y_vah_bin = val_e1h["high_return"].values.astype(np.int64)

    sch_3d   = StandardScaler().fit(X_trh_3d.reshape(-1, len(HIST_FEATS)))
    sch_ctx  = StandardScaler().fit(X_trh_ctx)
    sch_flat = RobustScaler().fit(X_trh_flat)

    X_trh_3ds, X_vah_3ds = sc3d(X_trh_3d, sch_3d), sc3d(X_vah_3d, sch_3d)
    X_trh_cs, X_vah_cs   = sch_ctx.transform(X_trh_ctx), sch_ctx.transform(X_vah_ctx)
    X_trh_fs, X_vah_fs   = sch_flat.transform(X_trh_flat), sch_flat.transform(X_vah_flat)

    cw3 = compute_weights(y_trh_band, 3)
    cw_bin = compute_weights(y_trh_bin, 2)

    trh_ld3 = DataLoader(TensorDataset(torch.tensor(X_trh_3ds), torch.tensor(X_trh_cs), torch.tensor(y_trh_band), torch.tensor(yr_trh), torch.tensor(yr_trh), torch.tensor(pos_trh)), batch_size=BATCH_SIZE, shuffle=True)
    vah_ld3 = DataLoader(TensorDataset(torch.tensor(X_vah_3ds), torch.tensor(X_vah_cs), torch.tensor(y_vah_band), torch.tensor(yr_vah), torch.tensor(yr_vah), torch.tensor(pos_vah)), batch_size=BATCH_SIZE, shuffle=False)

    bilstm_3cls = wrap(MultiTaskBiLSTM(len(HIST_FEATS), len(CTX_FEATS), 3, h=192))
    print("\n[Strategy 1] Unified 3-Class BiLSTM (3-4, 5-6, 7-10+)")
    bilstm_3cls = fit(bilstm_3cls, trh_ld3, vah_ld3, epochs=35, cls_crit=nn.CrossEntropyLoss(weight=cw3))

    trh_ldb = DataLoader(TensorDataset(torch.tensor(X_trh_3ds), torch.tensor(X_trh_cs), torch.tensor(y_trh_bin), torch.tensor(yr_trh), torch.tensor(yr_trh), torch.tensor(pos_trh)), batch_size=BATCH_SIZE, shuffle=True)
    vah_ldb = DataLoader(TensorDataset(torch.tensor(X_vah_3ds), torch.tensor(X_vah_cs), torch.tensor(y_vah_bin), torch.tensor(yr_vah), torch.tensor(yr_vah), torch.tensor(pos_vah)), batch_size=BATCH_SIZE, shuffle=False)

    bilstm_bin = wrap(MultiTaskBiLSTM(len(HIST_FEATS), len(CTX_FEATS), 2, h=128))
    print("\n[Strategy 2] Binary BiLSTM (>=7 pts High Return)")
    bilstm_bin = fit(bilstm_bin, trh_ldb, vah_ldb, epochs=30, cls_crit=nn.CrossEntropyLoss(weight=cw_bin))

    trh_mlpb = DataLoader(TensorDataset(torch.tensor(X_trh_fs), torch.tensor(y_trh_bin), torch.tensor(yr_trh), torch.tensor(yr_trh), torch.tensor(pos_trh)), batch_size=BATCH_SIZE, shuffle=True)
    vah_mlpb = DataLoader(TensorDataset(torch.tensor(X_vah_fs), torch.tensor(y_vah_bin), torch.tensor(yr_vah), torch.tensor(yr_vah), torch.tensor(pos_vah)), batch_size=BATCH_SIZE, shuffle=False)

    mlp_bin = wrap(MultiTaskMLP(X_trh_fs.shape[1], 2))
    print("\n[Strategy 2] Binary MLP (>=7 pts High Return)")
    mlp_bin = fit(mlp_bin, trh_mlpb, vah_mlpb, epochs=25, cls_crit=nn.CrossEntropyLoss(weight=cw_bin), mode="mlp")

    # Optimal binary threshold search
    pv_bin_l, _ = predict_lstm(bilstm_bin, X_vah_3ds, X_vah_cs)
    pv_bin_m, _ = predict_mlp(mlp_bin, X_vah_fs)
    pv_bin = 0.60 * pv_bin_l + 0.40 * pv_bin_m
    best_t, best_acc_t = 0.50, 0.0
    for t in np.arange(0.30, 0.80, 0.01):
        acc = accuracy_score(y_vah_bin, (pv_bin[:,1] >= t).astype(int))
        if acc > best_acc_t:
            best_acc_t, best_t = acc, t
    print(f"\nOptimal binary threshold (val): {best_t:.2f}")

    # Save models and scalers
    raw_3cls = bilstm_3cls.module if hasattr(bilstm_3cls, "module") else bilstm_3cls
    raw_bbin = bilstm_bin.module if hasattr(bilstm_bin, "module") else bilstm_bin
    raw_mbin = mlp_bin.module if hasattr(mlp_bin, "module") else mlp_bin

    torch.save(raw_3cls.state_dict(), os.path.join(MODEL_DIR, "expert1_high_3cls.pt"))
    torch.save(raw_bbin.state_dict(), os.path.join(MODEL_DIR, "expert1_high_bbin.pt"))
    torch.save(raw_mbin.state_dict(), os.path.join(MODEL_DIR, "expert1_high_mbin.pt"))

    with open(os.path.join(MODEL_DIR, "expert1_high_thresh.pkl"), "wb") as f:
        pickle.dump({"opt_thresh": best_t}, f)

    with open(os.path.join(SCALER_DIR, "expert1_high_scalers.pkl"), "wb") as f:
        pickle.dump({"scaler_3d": sch_3d, "scaler_ctx": sch_ctx, "scaler_flat": sch_flat}, f)

    print(f"Expert 1 High models saved to: {MODEL_DIR}")

if __name__ == "__main__":
    main()
