"""Generate publication-quality evaluation figures for Fantasy-XI PPO v4 vs Baseline."""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
})

def generate_graphs():
    reports_dir = os.path.join("rl_decision_layer", "ppo", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Load PPO v4 detailed data
    ppo_csv = os.path.join(reports_dir, "ppo_38gw_detailed_breakdown.csv")
    df_ppo = pd.read_csv(ppo_csv)
    
    # Baseline points per GW from deterministic MILP evaluation
    # Baseline takes 28 hits and finishes at 1974.0 pts
    baseline_pts = [
        65.0, 41.0, 89.0, 66.0, 55.0, 55.0, 57.0, 93.0, 44.0, 81.0,
        54.0, 61.0, 38.0, 68.0, 38.0, 61.0, 59.0, 38.0, 37.0, 53.0,
        67.0, 48.0, 37.0, 76.0, 59.0, 53.0, 55.0, 67.0, 52.0, 35.0,
        0.0, 47.0, 33.0, 0.0, 42.0, 44.0, 60.0, 46.0
    ]
    df_base = pd.DataFrame({
        "GW": range(1, 39),
        "Base_Net_Pts": baseline_pts,
        "Base_Cum": np.cumsum(baseline_pts)
    })
    
    # ----------------------------------------------------
    # GRAPH 1: GW vs Reward (PPO Gameweek Points & Chips)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    gws = df_ppo["GW"]
    pts = df_ppo["Match_Pts"]
    
    # Bar colors: highlight chip gameweeks
    colors = []
    for chip in df_ppo["Chip"]:
        if chip == "triple_captain":
            colors.append("#ff0055")
        elif chip == "wildcard":
            colors.append("#00d4ff")
        elif chip == "bench_boost":
            colors.append("#ffaa00")
        elif chip == "free_hit":
            colors.append("#a020f0")
        else:
            colors.append("#00ff87")
            
    bars = ax.bar(gws, pts, color=colors, edgecolor="#1a1a1a", linewidth=0.8, alpha=0.9, width=0.7)
    
    # Annotate mega haul chip weeks with distinct heights
    chip_offsets = {
        1: (1, 82),     # Free Hit (65 pts)
        2: (1.8, 116),  # Wildcard (101 pts)
        3: (3, 137),    # Triple Captain (129 pts)
        4: (4.2, 114),  # Bench Boost (99 pts)
    }
    for idx, row in df_ppo[df_ppo["Chip"] != "none"].iterrows():
        gw = int(row["GW"])
        p = row["Match_Pts"]
        chip_label = row["Chip"].replace("_", " ").title()
        xytext = chip_offsets.get(gw, (gw, p + 12))
        ax.annotate(f"{chip_label}\n({p:.0f} pts)",
                    xy=(gw, p),
                    xytext=xytext,
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    arrowprops=dict(facecolor="#1a1a1a", shrink=0.08, width=1, headwidth=4.5))
                    
    mean_pts = pts.mean()
    ax.set_title("PPO Agent: Points Scored per Gameweek", fontweight="bold", pad=15)
    ax.set_xlabel("Gameweek (GW 1 – 38)", fontweight="bold")
    ax.set_ylabel("Points Scored", fontweight="bold")
    ax.set_xticks(range(1, 39))
    ax.set_ylim(0, 150)
    
    # Stat box in upper right corner
    ax.text(0.98, 0.94, f"Season Average: {mean_pts:.1f} pts/GW\nTotal Points: {pts.sum():.0f} pts",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold", color="#1a1a1a",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fdf2f8", ec="#38003c", lw=1.2))
    plt.tight_layout()
    
    path_gw_reward = os.path.join(reports_dir, "gw_vs_reward.png")
    plt.savefig(path_gw_reward)
    plt.close()
    print(f"Saved: {path_gw_reward}")

    # ----------------------------------------------------
    # GRAPH 2: MILP Baseline vs PPO (Head-to-Head Comparison)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    x = np.arange(len(gws))
    width = 0.38
    
    ax.bar(x - width/2, pts, width=width, label=f"PPO v4 Strategic Agent (Total: {pts.sum():.0f} pts)", color="#00ff87", edgecolor="#111", alpha=0.9)
    ax.bar(x + width/2, df_base["Base_Net_Pts"], width=width, label=f"Deterministic MILP Baseline (Total: {df_base['Base_Net_Pts'].sum():.0f} pts)", color="#38003c", edgecolor="#111", alpha=0.85)
    
    ax.set_title("Gameweek Comparison: PPO Strategic Agent vs MILP Baseline", fontweight="bold", pad=15)
    ax.set_xlabel("Gameweek", fontweight="bold")
    ax.set_ylabel("Gameweek Net Points", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(gws)
    ax.set_ylim(0, 145)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    
    path_milp_vs_ppo = os.path.join(reports_dir, "milp_baseline_vs_ppo.png")
    plt.savefig(path_milp_vs_ppo)
    plt.close()
    print(f"Saved: {path_milp_vs_ppo}")

if __name__ == "__main__":
    generate_graphs()
