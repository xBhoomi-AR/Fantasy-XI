"""Quick manual check that the deterministic MILP-only baseline still runs
end to end, without launching a full season.

Run with:
    python -m rl_decision_layer.run_baseline
"""

from __future__ import annotations

from .historical_loop import run_backtest


def main():
    logs = run_backtest(start_gameweek=1, num_gameweeks=3)
    for log in logs:
        print(f"GW{log.gameweek}: transfers={log.transfers_made} hits={log.hits} "
              f"actual={log.actual_points:.1f} reward={log.reward:.1f}")
    print(f"{len(logs)} gameweeks completed, total reward: {sum(l.reward for l in logs):.1f}")


if __name__ == "__main__":
    main()
