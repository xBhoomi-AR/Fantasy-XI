import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

try:
    from .shared_config import (
        device, WINDOW_SIZE, TARGET, GATE_COL, BAND_NAMES, TEST_SEASONS,
        get_paths, engineer_features, build_3d_tensor,
        build_hybrid_matrix, BiLSTMBackbone
    )
except ImportError:
    from shared_config import (
        device, WINDOW_SIZE, TARGET, GATE_COL, BAND_NAMES, TEST_SEASONS,
        get_paths, engineer_features, build_3d_tensor,
        build_hybrid_matrix, BiLSTMBackbone
    )

def run_inference_engine(df, X_hybrid, model_floor, model_ceil, model_haul):
    gate_prob = pd.to_numeric(df.get(GATE_COL, 0.0), errors="coerce").fillna(0.0).values.astype(np.float32)
    mins_col = "gw_minus_1_minutes"
    gw1_mins = pd.to_numeric(df.get(mins_col, 0.0), errors="coerce").fillna(0.0).values.astype(np.float32)

    # Base raw outputs from the expert models
    raw_floor = np.clip(model_floor.predict(X_hybrid), 0.0, 2.5)
    raw_ceil  = np.maximum(model_ceil.predict(X_hybrid), 3.0)
    q_ceil    = np.maximum(model_haul.predict(X_hybrid), 3.0)

    # Smooth transition from floor to ceiling based on playing chance
    g_smooth = 1.0 / (1.0 + np.exp(-(gate_prob - 0.52) / 0.12))
    base_continuous = (1.0 - g_smooth) * raw_floor + g_smooth * raw_ceil

    # Add ceiling boost for likely starters with haul potential
    haul_lift = np.maximum(0.0, q_ceil - base_continuous)
    haul_boost = np.where(
        gate_prob >= 0.58,
        0.48 * np.minimum(haul_lift, 5.0) + 0.25 * np.maximum(0.0, haul_lift - 5.0),
        0.0
    )

    # Micro tie-breaker to prevent flat ranks for unplayed or low-minute players
    micro_rank_offset = (gate_prob * 0.15) + (gw1_mins / 90.0 * 0.10)
    pred_continuous = base_continuous + haul_boost

    floor_mask = gate_prob < 0.45
    pred_continuous[floor_mask] = (
        pred_continuous[floor_mask] * (gate_prob[floor_mask] / 0.45) ** 1.5 +
        micro_rank_offset[floor_mask]
    )
    pred_continuous[gate_prob < 0.15] = gate_prob[gate_prob < 0.15] * 0.05

    return np.clip(pred_continuous, 0.0, None)

def assign_band_label(pts):
    if pts < 2.5:
        return "0-2"
    elif pts < 5.5:
        return "3-5"
    elif pts < 9.5:
        return "6-9"
    else:
        return "10+"

def predict(data_path=None, output_path=None, eval_mode=True):
    default_data, model_dir, scaler_dir, _ = get_paths()
    if data_path is None:
        data_path = default_data
    if output_path is None:
        output_path = os.path.join(os.path.dirname(model_dir), "predicted_points.csv")

    scaler_path = os.path.join(scaler_dir, "scaler_3d.pkl")
    meta_path   = os.path.join(scaler_dir, "feature_meta.pkl")
    bb_ckpt     = os.path.join(model_dir, "backbone_bilstm.pt")
    floor_ckpt  = os.path.join(model_dir, "expert2_low_band.pkl")
    ceil_ckpt   = os.path.join(model_dir, "expert1_high_band.pkl")
    haul_ckpt   = os.path.join(model_dir, "haul_calibrator.pkl")

    for p in [scaler_path, meta_path, bb_ckpt, floor_ckpt, ceil_ckpt, haul_ckpt]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing artifact: {p}. Run train.py first.")

    with open(scaler_path, "rb") as f:
        scaler_3d = pickle.load(f)
    with open(meta_path, "rb") as f:
        hist_feats = pickle.load(f)["hist_feats"]
    with open(floor_ckpt, "rb") as f:
        model_floor = pickle.load(f)
    with open(ceil_ckpt, "rb") as f:
        model_ceil = pickle.load(f)
    with open(haul_ckpt, "rb") as f:
        model_haul = pickle.load(f)

    backbone = BiLSTMBackbone(len(hist_feats), h=64).to(device)
    backbone.load_state_dict(torch.load(bb_ckpt, map_location=device))
    backbone.eval()

    df = pd.read_csv(data_path)
    df = engineer_features(df)

    X_3d = build_3d_tensor(df, hist_feats)
    X_3d_scaled = scaler_3d.transform(X_3d.reshape(-1, len(hist_feats))).reshape(X_3d.shape)

    # Extract embeddings from the trained sequence backbone
    ld = DataLoader(TensorDataset(torch.tensor(X_3d_scaled)), batch_size=512, shuffle=False)
    embs = []
    with torch.no_grad():
        for (bx,) in ld:
            _, emb = backbone(bx.to(device))
            embs.append(emb.cpu().numpy())
    embeddings = np.vstack(embs)

    X_hybrid = build_hybrid_matrix(df, embeddings)
    final_points = run_inference_engine(df, X_hybrid, model_floor, model_ceil, model_haul)

    # Build clean output dataframe with key player information
    out_df = pd.DataFrame()

    for col in ["player_id", "id", "element"]:
        if col in df.columns:
            out_df["player_id"] = df[col].values
            break
    if "player_id" not in out_df.columns:
        out_df["player_id"] = np.arange(len(df))

    for col in ["player_name", "web_name", "name", "second_name"]:
        if col in df.columns:
            out_df["player_name"] = df[col].values
            break

    if "season" in df.columns:
        out_df["season"] = df["season"].values

    for col in ["target_gameweek", "gameweek", "gw", "event"]:
        if col in df.columns:
            out_df["gameweek"] = df[col].values
            break

    for col in ["position", "gw_minus_1_position", "position_code", "pos"]:
        if col in df.columns:
            out_df["position"] = df[col].values
            break

    for col in ["team_id", "gw_minus_1_team_id", "team", "current_team_id"]:
        if col in df.columns:
            out_df["team_id"] = df[col].values
            break

    for col in ["value", "gw_minus_1_value", "price", "now_cost", "current_cost"]:
        if col in df.columns:
            # Keep FPL's standard tenths-of-a-million integer/float price (e.g. 55 = £5.5m)
            out_df["price"] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            break

    out_df["predicted_points"] = np.round(final_points, 2)
    out_df["predicted_band"]   = [assign_band_label(p) for p in final_points]

    if GATE_COL in df.columns:
        out_df["gate_probability"] = np.round(pd.to_numeric(df[GATE_COL], errors="coerce").fillna(0.0).values, 4)

    # Optional evaluation against actual points if available
    if TARGET in df.columns and eval_mode:
        actuals = pd.to_numeric(df[TARGET], errors="coerce")
        if actuals.notna().sum() > 0:
            out_df["actual_points"] = actuals.fillna(0.0).astype(int)
            out_df["abs_error"] = np.round(np.abs(out_df["actual_points"] - final_points), 2)

            # Evaluate on the held-out test season (2025-26) for true benchmark metrics
            eval_mask = df["season"].isin(TEST_SEASONS).values if "season" in df.columns else np.ones(len(df), dtype=bool)
            test_actuals = out_df.loc[eval_mask, "actual_points"].values
            test_preds = final_points[eval_mask]

            cum_mae  = mean_absolute_error(test_actuals, test_preds)
            cum_rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
            rho, _   = spearmanr(test_actuals, test_preds)

            print(f"\nTest Benchmark (2025-26 Season | {len(test_actuals):,} samples):")
            print(f"Overall MAE: {cum_mae:.4f} | RMSE: {cum_rmse:.4f} | Spearman: {rho:.4f}\n")

            # Generate band-wise metrics and plot in reports/
            _, _, _, reports_dir = get_paths()
            masks = [
                test_actuals <= 2,
                (test_actuals >= 3) & (test_actuals <= 5),
                (test_actuals >= 6) & (test_actuals <= 9),
                test_actuals >= 10
            ]
            band_maes  = [mean_absolute_error(test_actuals[m], test_preds[m]) for m in masks]
            band_rmses = [np.sqrt(mean_squared_error(test_actuals[m], test_preds[m])) for m in masks]

            for name, m, b_mae, b_rmse in zip(BAND_NAMES, masks, band_maes, band_rmses):
                print(f"Band {name:<5} | Count: {m.sum():<7,} | MAE: {b_mae:.3f} | RMSE: {b_rmse:.3f}")

            # Save bandwise performance plot
            try:
                import matplotlib.pyplot as plt
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

                plot_path = os.path.join(reports_dir, "bandwise_results.png")
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"Saved performance plot to: {plot_path}")
            except Exception as e:
                pass

            # Save text summary report
            summary_path = os.path.join(reports_dir, "evaluation_summary.txt")
            with open(summary_path, "w") as f:
                f.write(f"BiLSTM Gate Architecture - Evaluation Report (2025-26 Test Set)\n")
                f.write(f"===============================================================\n")
                f.write(f"Total Test Samples: {len(test_actuals):,}\n")
                f.write(f"Overall MAE: {cum_mae:.4f} | RMSE: {cum_rmse:.4f} | Spearman: {rho:.4f}\n\n")
                for name, m, b_mae, b_rmse in zip(BAND_NAMES, masks, band_maes, band_rmses):
                    f.write(f"Band {name:<5} | Samples: {m.sum():<7,} | MAE: {b_mae:.3f} | RMSE: {b_rmse:.3f}\n")
            print(f"Saved report summary to: {summary_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved {len(out_df):,} predictions to: {output_path}")
    return out_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    predict(data_path=args.data, output_path=args.output, eval_mode=(not args.no_eval))
