import os
import glob
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_FILENAME = "player_5gw_sequences.csv"
matches = glob.glob(f"/kaggle/input/**/{CSV_FILENAME}", recursive=True)
if matches:
    DATA_PATH = matches[0]
elif os.path.exists(f"/kaggle/input/datasets/xbhoomi/fantasy-xi-sequences/{CSV_FILENAME}"):
    DATA_PATH = f"/kaggle/input/datasets/xbhoomi/fantasy-xi-sequences/{CSV_FILENAME}"
elif os.path.exists(CSV_FILENAME):
    DATA_PATH = CSV_FILENAME
else:
    raise FileNotFoundError(f"Could not find {CSV_FILENAME}")

df = pd.read_csv(DATA_PATH)
TARGET = "total_points"
GATE_COL = "current_gate_probability"
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()

OUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
MODEL_DIR = os.path.join(OUT_DIR, "models")
SCALER_DIR = os.path.join(OUT_DIR, "scalers")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

CSV_OUT = os.path.join(OUT_DIR, "predicted_points.csv")
PLOT_PATH = os.path.join(OUT_DIR, "bandwise_results.png")

WINDOW_SIZE = 5
TRAIN_SEASONS = ["2016-17","2017-18","2018-19","2019-20","2020-21","2021-22","2022-23","2023-24"]
VAL_SEASONS   = ["2024-25"]
TEST_SEASONS  = ["2025-26"]

# Rolling metrics with higher weights on more recent fixtures
w = np.array([0.05, 0.10, 0.15, 0.30, 0.40])
for metric in ["expected_goals", "expected_assists", "bps", "ict_index", "minutes"]:
    cols = [f"gw_minus_{s}_{metric}" for s in range(5, 0, -1) if f"gw_minus_{s}_{metric}" in df.columns]
    if len(cols) == 5:
        arr = np.column_stack([pd.to_numeric(df[c], errors="coerce").fillna(0).values for c in cols])
        df[f"ewma_{metric}"] = np.dot(arr, w).astype(np.float32)
    else:
        df[f"ewma_{metric}"] = 0.0

df["haul_potential_index"] = (df["ewma_expected_goals"] * 4.0 + df["ewma_expected_assists"] * 3.0 + df["ewma_ict_index"] * 0.1).astype(np.float32)
gw1_xg = pd.to_numeric(df.get("gw_minus_1_expected_goals", 0), errors="coerce").fillna(0)
rest_xg = [df.get(f"gw_minus_{s}_expected_goals", 0) for s in range(2, 6)]
avg_rest = np.mean([pd.to_numeric(c, errors="coerce").fillna(0) for c in rest_xg], axis=0) if rest_xg else 0.0
df["xg_momentum"] = (gw1_xg - avg_rest).astype(np.float32)

train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
val_df   = df[df["season"].isin(VAL_SEASONS)].copy()
test_df  = df[df["season"].isin(TEST_SEASONS)].copy()

hist_feats = sorted(list(set([c.split("gw_minus_1_")[-1] for c in df.columns if c.startswith("gw_minus_1_")])))

def build_3d(data):
    n = len(data)
    X_3d = np.zeros((n, WINDOW_SIZE, len(hist_feats)), dtype=np.float32)
    for step in range(1, WINDOW_SIZE + 1):
        t = WINDOW_SIZE - step
        for fi, feat in enumerate(hist_feats):
            col = f"gw_minus_{step}_{feat}"
            if col in data.columns:
                X_3d[:, t, fi] = pd.to_numeric(data[col], errors="coerce").fillna(0).values
    return X_3d

X_tr_3d = build_3d(train_df)
X_va_3d = build_3d(val_df)
X_te_3d = build_3d(test_df)

sc = StandardScaler()
X_tr_3d = sc.fit_transform(X_tr_3d.reshape(-1, len(hist_feats))).reshape(X_tr_3d.shape)
X_va_3d = sc.transform(X_va_3d.reshape(-1, len(hist_feats))).reshape(X_va_3d.shape)
X_te_3d = sc.transform(X_te_3d.reshape(-1, len(hist_feats))).reshape(X_te_3d.shape)

with open(os.path.join(SCALER_DIR, "scaler_3d.pkl"), "wb") as f:
    pickle.dump(sc, f)
with open(os.path.join(SCALER_DIR, "feature_meta.pkl"), "wb") as f:
    pickle.dump({"hist_feats": hist_feats}, f)

y_train = train_df[TARGET].values.astype(np.float32)
y_val   = val_df[TARGET].values.astype(np.float32)
y_test  = test_df[TARGET].values.astype(np.float32)

gate_val   = pd.to_numeric(val_df[GATE_COL], errors="coerce").fillna(0).values.astype(np.float32)
gate_test  = pd.to_numeric(test_df[GATE_COL], errors="coerce").fillna(0).values.astype(np.float32)
gw1_mins_te  = pd.to_numeric(test_df.get("gw_minus_1_minutes", 0), errors="coerce").fillna(0).values.astype(np.float32)
gw1_start_te = pd.to_numeric(test_df.get("gw_minus_1_started", 0), errors="coerce").fillna(0).values.astype(np.float32)

class TemporalAttention(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(d_in, d_in // 2), nn.Tanh(), nn.Linear(d_in // 2, 1))
    def forward(self, x):
        return (x * torch.softmax(self.w(x), dim=1)).sum(dim=1)

class BiLSTMBackbone(nn.Module):
    def __init__(self, in_dim, h=64):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.attn = TemporalAttention(h * 2)
        self.head = nn.Linear(h * 2, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        emb  = self.attn(o)
        return self.head(emb).squeeze(-1), emb

backbone = BiLSTMBackbone(len(hist_feats), h=64).to(device)
opt_bb   = torch.optim.AdamW(backbone.parameters(), lr=8e-4, weight_decay=1e-4)
bb_crit  = nn.SmoothL1Loss()

bb_ds = TensorDataset(torch.tensor(X_tr_3d), torch.tensor(y_train))
bb_ld = DataLoader(bb_ds, batch_size=256, shuffle=True)

# Train the sequence network
for ep in range(1, 11):
    backbone.train()
    ep_loss = 0.0
    for bx, by in bb_ld:
        bx, by = bx.to(device), by.to(device)
        opt_bb.zero_grad()
        p, _ = backbone(bx)
        loss = bb_crit(p, by)
        loss.backward()
        nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        opt_bb.step()
        ep_loss += loss.item() * bx.size(0)
    if ep % 2 == 0 or ep == 10:
        print(f"Epoch {ep:02d}/10 | Loss: {ep_loss/len(bb_ds):.4f}")

torch.save(backbone.state_dict(), os.path.join(MODEL_DIR, "backbone_bilstm.pt"))

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

def build_hybrid(data, embs):
    wide_cols = ["current_gate_probability", "haul_potential_index", "xg_momentum"]
    wide_arr = np.column_stack([pd.to_numeric(data.get(c, 0), errors="coerce").fillna(0).values for c in wide_cols])
    return np.hstack([embs, wide_arr]).astype(np.float32)

X_hyb_tr = build_hybrid(train_df, emb_tr)
X_hyb_va = build_hybrid(val_df, emb_va)
X_hyb_te = build_hybrid(test_df, emb_te)

# Train floor expert for low-scoring appearances
print("Training Stage 1: Expert 2 Low Band BiLSTM...")
mask_e2 = y_train < 3.0
model_e2 = lgb.LGBMRegressor(
    objective="regression_l1", n_estimators=600, learning_rate=0.04,
    num_leaves=31, random_state=SEED, n_jobs=-1, verbose=-1
)
model_e2.fit(X_hyb_tr[mask_e2], y_train[mask_e2])
with open(os.path.join(MODEL_DIR, "expert2_low_band.pkl"), "wb") as f:
    pickle.dump(model_e2, f)

# Train ceiling expert for regular starters
print("Training Stage 2: Expert 1 High Band BiLSTM...")
mask_e1 = y_train >= 3.0
w_e1    = np.where(y_train[mask_e1] >= 10.0, 4.0, 1.0)
model_e1 = lgb.LGBMRegressor(
    objective="huber", alpha=0.9, n_estimators=800, learning_rate=0.03,
    num_leaves=45, random_state=SEED, n_jobs=-1, verbose=-1
)
model_e1.fit(X_hyb_tr[mask_e1], y_train[mask_e1], sample_weight=w_e1)
with open(os.path.join(MODEL_DIR, "expert1_high_band.pkl"), "wb") as f:
    pickle.dump(model_e1, f)

# Train quantile expert to estimate high-end haul ceiling
print("Training Stage 3: Haul Potential Calibrator...")
mask_q = y_train >= 4.0
model_q = lgb.LGBMRegressor(
    objective="quantile", alpha=0.85, n_estimators=700, learning_rate=0.03,
    num_leaves=31, random_state=SEED, n_jobs=-1, verbose=-1
)
model_q.fit(X_hyb_tr[mask_q], y_train[mask_q])
with open(os.path.join(MODEL_DIR, "haul_calibrator.pkl"), "wb") as f:
    pickle.dump(model_q, f)

raw_test_e2 = np.clip(model_e2.predict(X_hyb_te), 0.0, 2.5)
raw_test_e1 = np.maximum(model_e1.predict(X_hyb_te), 3.0)
q_test_ceil = np.maximum(model_q.predict(X_hyb_te), 3.0)

# Smoothly blend between floor and ceiling using gate probability
g_test_smooth = 1.0 / (1.0 + np.exp(-(gate_test - 0.52) / 0.12))
base_continuous = (1.0 - g_test_smooth) * raw_test_e2 + g_test_smooth * raw_test_e1

haul_lift = np.maximum(0.0, q_test_ceil - base_continuous)
haul_boost = np.where(
    gate_test >= 0.58,
    0.48 * np.minimum(haul_lift, 5.0) + 0.25 * np.maximum(0.0, haul_lift - 5.0),
    0.0
)

# Smooth continuous offset to break ties in lower ranks
micro_rank_offset = (gate_test * 0.15) + (gw1_mins_te / 90.0 * 0.10)
pred_continuous = base_continuous + haul_boost

floor_mask = gate_test < 0.45
pred_continuous[floor_mask] = (
    pred_continuous[floor_mask] * (gate_test[floor_mask] / 0.45) ** 1.5 
    + micro_rank_offset[floor_mask]
)
pred_continuous[gate_test < 0.15] = gate_test[gate_test < 0.15] * 0.05
final_predictions = np.clip(pred_continuous, 0.0, None)

cum_mae  = mean_absolute_error(y_test, final_predictions)
cum_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
rho, _   = spearmanr(y_test, final_predictions)

print(f"\nMAE: {cum_mae:.4f} | RMSE: {cum_rmse:.4f} | Spearman: {rho:.4f}")

BAND_NAMES = ["0-2", "3-5", "6-9", "10+"]
masks = [
    y_test <= 2,
    (y_test >= 3) & (y_test <= 5),
    (y_test >= 6) & (y_test <= 9),
    y_test >= 10
]

band_maes  = [mean_absolute_error(y_test[m], final_predictions[m]) for m in masks]
band_rmses = [np.sqrt(mean_squared_error(y_test[m], final_predictions[m])) for m in masks]

for name, m, b_mae, b_rmse in zip(BAND_NAMES, masks, band_maes, band_rmses):
    print(f"{name:<6} | {m.sum():<7,} | MAE: {b_mae:.3f} | RMSE: {b_rmse:.3f}")

# Build prediction export
out_df = pd.DataFrame()

for col in ["player_id", "id", "element"]:
    if col in test_df.columns:
        out_df["player_id"] = test_df[col].values
        break

for col in ["player_name", "web_name", "name"]:
    if col in test_df.columns:
        out_df["player_name"] = test_df[col].values
        break

for col in ["target_gameweek", "gameweek", "gw"]:
    if col in test_df.columns:
        out_df["gameweek"] = test_df[col].values
        break

for col in ["position", "position_code", "pos"]:
    if col in test_df.columns:
        out_df["position"] = test_df[col].values
        break

for col in ["team_id", "team"]:
    if col in test_df.columns:
        out_df["team_id"] = test_df[col].values
        break

for col in ["value", "price", "now_cost"]:
    if col in test_df.columns:
        out_df["price"] = (pd.to_numeric(test_df[col], errors="coerce").fillna(0.0) / 10.0).round(1)
        break

out_df["predicted_points"] = np.round(final_predictions, 2)
def assign_band_label(pts):
    if pts < 2.5: return "0-2"
    elif pts < 5.5: return "3-5"
    elif pts < 9.5: return "6-9"
    else: return "10+"
out_df["predicted_band"]   = [assign_band_label(p) for p in final_predictions]
out_df["gate_probability"] = np.round(gate_test, 4)
out_df["actual_points"]    = y_test.astype(int)
out_df["abs_error"]        = np.round(np.abs(y_test - final_predictions), 2)

out_df.to_csv(CSV_OUT, index=False)
print(f"Saved predictions to: {CSV_OUT}")

# Plot MAE (blue) and RMSE (orange)
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(BAND_NAMES))
width = 0.35

rects1 = ax.bar(x - width/2, band_maes,  width, label="MAE",  color="#1f77b4", edgecolor="black", alpha=0.85)
rects2 = ax.bar(x + width/2, band_rmses, width, label="RMSE", color="#ff7f0e", edgecolor="black", alpha=0.85)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_title(f"Band-wise Performance (Overall MAE: {cum_mae:.2f} | Spearman: {rho:.3f})", fontsize=12, fontweight="bold")
ax.set_xlabel("Score Band", fontsize=11)
ax.set_ylabel("Error", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(BAND_NAMES, fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_ylim(0, max(band_rmses) * 1.25)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
plt.show()
print(f"Saved plot to: {PLOT_PATH}")