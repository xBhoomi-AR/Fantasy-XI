"""Prints a human-readable squad/XI recommendation from a saved PPO model for
one gameweek - names, not bare player IDs.

Reuses the exact same PPOEnv.step() call chain evaluate.py uses (PPO action ->
action_to_milp_kwargs() -> select_squad() -> pick_starting_xi()); this file
adds no new selection logic, it only looks up names/teams for display.

Player names come from players.csv's `player_name` column - never `web_name`,
which is known to be unreliable (see rl_decision_layer/README.md). Team names
come from the canonical prediction data's own team_id, joined against
teams.csv, the same mapping the rest of the pipeline already uses.

Run with:
    python -m rl_decision_layer.ppo.show_squad --model ppo_fpl --start-gameweek 1
"""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from ..predictions.interface import load_player_metadata, load_team_metadata
from .env import PPOEnv
from .train import MODELS_DIR

POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


def _sorted_by_position(pids: list[int], candidates) -> list[int]:
    def key(pid):
        pos = candidates.loc[pid, "position"] if pid in candidates.index else "?"
        return (POSITION_ORDER.get(pos, 9), pid)
    return sorted(pids, key=key)


def _format_player(pid: int, candidates, players, teams, captain_id, vice_captain_id) -> str:
    pos = candidates.loc[pid, "position"] if pid in candidates.index else "?"
    team_id = candidates.loc[pid, "team_id"] if pid in candidates.index else None
    team = teams.loc[team_id, "team_name"] if team_id in teams.index else "?"
    name = players.loc[pid, "player_name"] if pid in players.index else f"#{pid}"
    tag = ""
    if pid == captain_id:
        tag = "  (C)"
    elif pid == vice_captain_id:
        tag = "  (VC)"
    return f"  {pos:<4} {name:<28} {team:<18}{tag}"


def show_squad(model_name: str = "ppo_fpl_v4", start_gameweek: int = 1, season: str = "2025-26") -> dict:
    """Runs one PPO decision for start_gameweek and prints the resulting
    squad/XI/captain/vice in human-readable form. Returns the raw info dict
    (as produced by PPOEnv.step()) for programmatic use."""
    players = load_player_metadata().set_index("player_id")
    teams = load_team_metadata().set_index("team_id")

    model = PPO.load(MODELS_DIR / model_name)
    env = PPOEnv(start_gameweek=start_gameweek, num_gameweeks=1, season=season)
    obs, _ = env.reset()

    candidates = env._state.candidates.drop_duplicates("player_id").set_index("player_id")
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    act = tuple(int(a) for a in action)
    print(f"GAMEWEEK: {info.get('gameweek', start_gameweek)}")
    print(f"PPO ACTION: aggressiveness={act[0]}  budget_level={act[1]}"
          + (f"  position_bias={act[2]}  chip_choice={act[3]}" if len(act) == 4 else ""))

    if terminated and info.get("status") not in (None, "Optimal"):
        print(f"\nNo legal squad found under this action (status={info.get('status')}).")
        print("Episode ends here - nothing further to show for this gameweek.")
        return info

    squad_ids = _sorted_by_position(info["squad_ids"], candidates)
    starting_ids = set(info["starting_ids"])
    bench_ids = _sorted_by_position(info["bench_ids"], candidates)

    print(f"\n15-PLAYER SQUAD ({info['squad_size']}/15):")
    for pid in squad_ids:
        marker = " *" if pid in starting_ids else "  "
        print(marker + _format_player(pid, candidates, players, teams,
                                       info["captain_id"], info["vice_captain_id"]))
    print("  (* = starting XI)")

    print(f"\nSTARTING XI ({len(info['starting_ids'])}/11):")
    for pid in _sorted_by_position(info["starting_ids"], candidates):
        print(_format_player(pid, candidates, players, teams, info["captain_id"], info["vice_captain_id"]))

    captain_name = players.loc[info["captain_id"], "player_name"] if info["captain_id"] in players.index else "?"
    vice_name = players.loc[info["vice_captain_id"], "player_name"] if info["vice_captain_id"] in players.index else "?"
    print(f"\nCAPTAIN: {captain_name}")
    print(f"VICE-CAPTAIN: {vice_name}")

    print(f"\nBENCH ({len(bench_ids)}):")
    for pid in bench_ids:
        print(_format_player(pid, candidates, players, teams, info["captain_id"], info["vice_captain_id"]))

    print(f"\nCHIP USED: {info.get('chip_used', 'none')}")
    print(f"TRANSFERS: {info['transfers']}")
    print(f"HITS: {info['hits']}")
    print(f"BANK: {info['bank']}")
    print(f"FREE TRANSFERS: {info['free_transfers']}")
    print(f"REWARD: {reward:.1f}")
    print(f"LEGALITY: {'LEGAL' if info['legality_violations'] == [] else info['legality_violations']}")

    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_fpl_v4")
    parser.add_argument("--start-gameweek", type=int, default=1)
    parser.add_argument("--season", default="2025-26")
    args = parser.parse_args()

    show_squad(args.model, args.start_gameweek, args.season)
