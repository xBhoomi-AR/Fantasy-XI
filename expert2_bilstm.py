

import os
import sys
import pickle
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler

from shared_config import (
    device, N_GPUS, TRAIN_SEASONS, VAL_SEASONS, TARGET,
    HIST_FEATS, CTX_FEATS, BATCH_SIZE, get_paths,
    engineer_features, build_matrices, sc3d, compute_weights,
    Expert2BiLSTM, Expert2MLP, fit_e2, wrap
)

def main():
    print("=" * 80)
    print("STAGE 1: TRAINING UPGRADED EXPERT 2 (POSITION WEIGHTED + AUX MINS)")
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

    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val_df   = df[df["season"].isin(VAL_SEASONS)].copy()

    train_e2 = train_df[train_df[TARGET] < 3].copy()
    val_e2   = val_df[val_df["current_gate_probability"] < 0.50].copy()

    print(f"Train rows: E2={len(train_e2):,} | Val rows: E2={len(val_e2):,}")

    X_tr2_3d, X_tr2_ctx, X_tr2_flat, y_tr2, yr_tr2, ym_tr2, pos_tr2 = build_matrices(train_e2, "exp2_target")
    X_va2_3d, X_va2_ctx, X_va2_flat, y_va2, yr_va2, ym_va2, pos_va2 = build_matrices(val_e2,   "exp2_target")

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
    print("\n[BiLSTM Upgraded Expert 2]")
    bilstm_e2 = fit_e2(bilstm_e2, tr2_ld, va2_ld, epochs=30, cls_crit=nn.CrossEntropyLoss(weight=cw2))

    tr2_mlp_ld = DataLoader(TensorDataset(torch.tensor(X_tr2_fs), torch.tensor(y_tr2), torch.tensor(yr_tr2), torch.tensor(ym_tr2), torch.tensor(pos_tr2)), batch_size=BATCH_SIZE, shuffle=True)
    va2_mlp_ld = DataLoader(TensorDataset(torch.tensor(X_va2_fs), torch.tensor(y_va2), torch.tensor(yr_va2), torch.tensor(ym_va2), torch.tensor(pos_va2)), batch_size=BATCH_SIZE, shuffle=False)

    mlp_e2 = wrap(Expert2MLP(X_tr2_fs.shape[1], 3))
    print("\n[MultiTask MLP Upgraded Expert 2]")
    mlp_e2 = fit_e2(mlp_e2, tr2_mlp_ld, va2_mlp_ld, epochs=25, cls_crit=nn.CrossEntropyLoss(weight=cw2), mode="mlp")

    # Save models and scalers
    raw_bilstm = bilstm_e2.module if hasattr(bilstm_e2, "module") else bilstm_e2
    raw_mlp    = mlp_e2.module if hasattr(mlp_e2, "module") else mlp_e2

    torch.save(raw_bilstm.state_dict(), os.path.join(MODEL_DIR, "expert2_bilstm.pt"))
    torch.save(raw_mlp.state_dict(), os.path.join(MODEL_DIR, "expert2_mlp.pt"))

    with open(os.path.join(SCALER_DIR, "expert2_scalers.pkl"), "wb") as f:
        pickle.dump({"scaler_3d": sc2_3d, "scaler_ctx": sc2_ctx, "scaler_flat": sc2_flat}, f)

    print(f"Expert 2 models saved to: {MODEL_DIR}")

if __name__ == "__main__":
    main()
