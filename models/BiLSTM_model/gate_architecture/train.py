import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import lightgbm as lgb

try:
    from .shared_config import (
        device, SEED, BATCH_SIZE, TARGET, GATE_COL,
        TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS, BAND_NAMES,
        get_paths, engineer_features, get_feature_list,
        build_3d_tensor, build_hybrid_matrix, BiLSTMBackbone
    )
    from .predict import run_inference_engine
except ImportError:
    from shared_config import (
        device, SEED, BATCH_SIZE, TARGET, GATE_COL,
        TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS, BAND_NAMES,
        get_paths, engineer_features, get_feature_list,
        build_3d_tensor, build_hybrid_matrix, BiLSTMBackbone
    )
    from predict import run_inference_engine

def train(data_path=None, epochs_backbone=10):
    default_data, model_dir, scaler_dir, _ = get_paths()
    if data_path is None:
        data_path = default_data

    print(f"Training pipeline started using: {data_path}")

    df = pd.read_csv(data_path)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df = engineer_features(df)

    hist_feats = get_feature_list(df)

    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val_df   = df[df["season"].isin(VAL_SEASONS)].copy()
    test_df  = df[df["season"].isin(TEST_SEASONS)].copy()
    print(f"Train samples: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Scale 3D sequence features across the 5 gameweek lookback
    X_tr_3d = build_3d_tensor(train_df, hist_feats)
    X_va_3d = build_3d_tensor(val_df, hist_feats)
    X_te_3d = build_3d_tensor(test_df, hist_feats)

    scaler_3d = StandardScaler()
    X_tr_3d = scaler_3d.fit_transform(X_tr_3d.reshape(-1, len(hist_feats))).reshape(X_tr_3d.shape)
    X_va_3d = scaler_3d.transform(X_va_3d.reshape(-1, len(hist_feats))).reshape(X_va_3d.shape)
    X_te_3d = scaler_3d.transform(X_te_3d.reshape(-1, len(hist_feats))).reshape(X_te_3d.shape)

    with open(os.path.join(scaler_dir, "scaler_3d.pkl"), "wb") as f:
        pickle.dump(scaler_3d, f)
    with open(os.path.join(scaler_dir, "feature_meta.pkl"), "wb") as f:
        pickle.dump({"hist_feats": hist_feats}, f)

    y_train = train_df[TARGET].values.astype(np.float32)
    y_val   = val_df[TARGET].values.astype(np.float32)
    y_test  = test_df[TARGET].values.astype(np.float32)

    # Train BiLSTM to learn 128-dim representations from sequential match stats
    backbone = BiLSTMBackbone(len(hist_feats), h=64).to(device)
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    ds_tr = TensorDataset(torch.tensor(X_tr_3d), torch.tensor(y_train))
    ld_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)

    print("Training BiLSTM backbone...")
    for ep in range(1, epochs_backbone + 1):
        backbone.train()
        tot_loss = 0.0
        for bx, by in ld_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            p, _ = backbone(bx)
            loss = criterion(p, by)
            loss.backward()
            nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item() * bx.size(0)
        avg_loss = tot_loss / len(ds_tr)
        if ep % 2 == 0 or ep == epochs_backbone:
            print(f"  Epoch {ep:02d}/{epochs_backbone:02d} | Loss: {avg_loss:.4f}")

    torch.save(backbone.state_dict(), os.path.join(model_dir, "backbone_bilstm.pt"))

    @torch.no_grad()
    def get_embeddings(model, arr):
        model.eval()
        ld = DataLoader(TensorDataset(torch.tensor(arr)), batch_size=512, shuffle=False)
        embs = []
        for (bx,) in ld:
            _, emb = model(bx.to(device))
            embs.append(emb.cpu().numpy())
        return np.vstack(embs)

    emb_tr = get_embeddings(backbone, X_tr_3d)
    emb_va = get_embeddings(backbone, X_va_3d)
    emb_te = get_embeddings(backbone, X_te_3d)

    # Build hybrid features by combining neural embeddings with match context
    X_hyb_tr = build_hybrid_matrix(train_df, emb_tr)
    X_hyb_va = build_hybrid_matrix(val_df, emb_va)
    X_hyb_te = build_hybrid_matrix(test_df, emb_te)

    # Expert 1: Floor model for low-scoring appearance and bench players
    print("Training Stage 1: Expert 2 Low Band BiLSTM...")
    mask_floor = y_train < 3.0
    model_floor = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=600, learning_rate=0.04,
        num_leaves=31, random_state=SEED, n_jobs=-1, verbose=-1
    )
    model_floor.fit(X_hyb_tr[mask_floor], y_train[mask_floor])
    with open(os.path.join(model_dir, "expert2_low_band.pkl"), "wb") as f:
        pickle.dump(model_floor, f)

    # Expert 2: Ceiling model for active starters, with extra weight on big returns
    print("Training Stage 2: Expert 1 High Band BiLSTM...")
    mask_ceil = y_train >= 3.0
    weights_ceil = np.where(y_train[mask_ceil] >= 10.0, 4.0, 1.0)
    model_ceil = lgb.LGBMRegressor(
        objective="huber", alpha=0.90, n_estimators=800, learning_rate=0.03,
        num_leaves=45, random_state=SEED, n_jobs=-1, verbose=-1
    )
    model_ceil.fit(X_hyb_tr[mask_ceil], y_train[mask_ceil], sample_weight=weights_ceil)
    with open(os.path.join(model_dir, "expert1_high_band.pkl"), "wb") as f:
        pickle.dump(model_ceil, f)

    # Expert 3: Quantile specialist to estimate double-digit ceiling potential
    print("Training Stage 3: Haul Potential Calibrator...")
    mask_haul = y_train >= 4.0
    model_haul = lgb.LGBMRegressor(
        objective="quantile", alpha=0.85, n_estimators=700, learning_rate=0.03,
        num_leaves=31, random_state=SEED, n_jobs=-1, verbose=-1
    )
    model_haul.fit(X_hyb_tr[mask_haul], y_train[mask_haul])
    with open(os.path.join(model_dir, "haul_calibrator.pkl"), "wb") as f:
        pickle.dump(model_haul, f)

    # Quick test check
    test_preds = run_inference_engine(test_df, X_hyb_te, model_floor, model_ceil, model_haul)
    cum_mae = mean_absolute_error(y_test, test_preds)
    cum_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    rho, _ = spearmanr(y_test, test_preds)

    print(f"\nTraining complete.")
    print(f"Overall MAE: {cum_mae:.4f} | RMSE: {cum_rmse:.4f} | Spearman: {rho:.4f}")

if __name__ == "__main__":
    train()
