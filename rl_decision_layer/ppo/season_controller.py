# Sequential season runner: loops PPO -> MILP -> XI over multiple gameweeks,
# printing human-readable output (transfers, squad, captain, bench, bank)
# and optionally saving/restoring state for resumable runs.

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from stable_baselines3 import PPO

from ..predictions.interface import load_player_metadata, load_team_metadata
from .env import PPOEnv
from .train import MODELS_DIR

POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


@dataclass
class SeasonState:
    """Minimal state needed to resume a sequential run."""
    gameweek: int
    squad_ids: list[int] = field(default_factory=list)
    bank: float = 0.0
    free_transfers: int = 1

    @classmethod
    def load(cls, path: str) -> "SeasonState":
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def _sorted_by_position(pids, candidates) -> list[int]:
    def key(pid):
        pos = candidates.loc[pid, "position"] if pid in candidates.index else "?"
        return (POSITION_ORDER.get(pos, 9), pid)
    return sorted(pids, key=key)


def _name(pid, players) -> str:
    return players.loc[pid, "player_name"] if pid in players.index else f"#{pid}"


def _format_player(pid, candidates, players, teams, captain_id, vice_captain_id) -> str:
    pos = candidates.loc[pid, "position"] if pid in candidates.index else "?"
    team_id = candidates.loc[pid, "team_id"] if pid in candidates.index else None
    team = teams.loc[team_id, "team_name"] if team_id in teams.index else "?"
    tag = ""
    if pid == captain_id:
        tag = "  (C)"
    elif pid == vice_captain_id:
        tag = "  (VC)"
    return f"  {pos:<4} {_name(pid, players):<28} {team:<18}{tag}"


def run_season(model_name: str = "ppo_fpl", start_gameweek: int = 1, num_gameweeks: int = 5,
               season: str = "2025-26", load_state: str | None = None,
               save_state: str | None = None) -> SeasonState | None:
    players = load_player_metadata().set_index("player_id")
    teams = load_team_metadata().set_index("team_id")
    model = PPO.load(MODELS_DIR / model_name)

    resuming = load_state is not None
    env_kwargs = dict(season=season)
    if resuming:
        prior = SeasonState.load(load_state)
        start_gameweek = prior.gameweek
        env_kwargs.update(initial_squad=prior.squad_ids, initial_bank=prior.bank,
                           initial_free_transfers=prior.free_transfers)
        print(f"Resumed from {load_state}: GW{start_gameweek}, bank={prior.bank}, "
              f"free_transfers={prior.free_transfers}, squad_size={len(prior.squad_ids)}\n")

    env = PPOEnv(start_gameweek=start_gameweek, num_gameweeks=num_gameweeks, **env_kwargs)
    obs, _ = env.reset()
    previous_squad_ids = set(env._state.squad_ids)

    final_state: SeasonState | None = None

    for i in range(num_gameweeks):
        pre_candidates = env._state.candidates.drop_duplicates("player_id").set_index("player_id")
        gw_label = env._state.gameweek

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\n{'=' * 60}")
        print(f"GAMEWEEK {gw_label}")
        print(f"{'=' * 60}")
        print(f"PPO ACTION: aggressiveness={int(action[0])}  budget_level={int(action[1])}")

        if terminated and info.get("status") not in (None, "Optimal"):
            print(f"\nNo legal squad found under this action (status={info.get('status')}).")
            print("Sequential run ends here.")
            break

        squad_ids = set(info["squad_ids"])

        if i == 0 and not resuming:
            print("\nINITIAL SQUAD (no prior gameweek to compare against)")
        else:
            transfers_in = squad_ids - previous_squad_ids
            transfers_out = previous_squad_ids - squad_ids
            print(f"\nTRANSFERS IN  ({len(transfers_in)}):")
            for pid in transfers_in:
                print("  + " + _name(pid, players))
            print(f"TRANSFERS OUT ({len(transfers_out)}):")
            for pid in transfers_out:
                print("  - " + _name(pid, players))

        ordered_squad = _sorted_by_position(list(squad_ids), pre_candidates)
        starting_ids = set(info["starting_ids"])
        print(f"\n15-PLAYER SQUAD ({info['squad_size']}/15):")
        for pid in ordered_squad:
            marker = " *" if pid in starting_ids else "  "
            print(marker + _format_player(pid, pre_candidates, players, teams,
                                           info["captain_id"], info["vice_captain_id"]))
        print("  (* = starting XI)")

        print(f"\nSTARTING XI ({len(info['starting_ids'])}/11):")
        for pid in _sorted_by_position(info["starting_ids"], pre_candidates):
            print(_format_player(pid, pre_candidates, players, teams, info["captain_id"], info["vice_captain_id"]))

        print(f"\nCAPTAIN: {_name(info['captain_id'], players)}")
        print(f"VICE-CAPTAIN: {_name(info['vice_captain_id'], players)}")

        bench_ids = _sorted_by_position(info["bench_ids"], pre_candidates)
        print(f"\nBENCH ({len(bench_ids)}):")
        for pid in bench_ids:
            print(_format_player(pid, pre_candidates, players, teams, info["captain_id"], info["vice_captain_id"]))

        print(f"\nTRANSFERS: {info['transfers']}   HITS: {info['hits']}")
        print(f"BANK: {info['bank']}   FREE TRANSFERS: {info['free_transfers']}")
        print(f"REWARD: {reward:.1f}")
        print(f"LEGALITY: {'LEGAL' if info['legality_violations'] == [] else info['legality_violations']}")

        previous_squad_ids = squad_ids
        final_state = SeasonState(gameweek=info["gameweek"] + 1, squad_ids=sorted(squad_ids),
                                   bank=info["bank"], free_transfers=info["free_transfers"])

        if terminated or truncated:
            break

    if save_state and final_state is not None:
        final_state.save(save_state)
        print(f"\nSeason state saved to {save_state} - resume with --load-state {save_state}")

    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_fpl")
    parser.add_argument("--start-gameweek", type=int, default=1,
                         help="ignored if --load-state is given (resumes from the saved gameweek instead)")
    parser.add_argument("--num-gameweeks", type=int, default=5)
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--load-state", default=None, help="resume from a previously saved SeasonState JSON file")
    parser.add_argument("--save-state", default=None, help="save the final SeasonState to this JSON file")
    args = parser.parse_args()

    run_season(args.model, args.start_gameweek, args.num_gameweeks, args.season,
               args.load_state, args.save_state)
