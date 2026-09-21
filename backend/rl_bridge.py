"""Thin bridge between the existing, already-validated sequential RL pipeline
(PPOEnv, SeasonState) and a JSON-friendly interface the backend API calls.

This module does NOT reimplement squad selection, transfer calculation,
chip logic, or scoring. Every decision is made exactly where it already is -
inside PPOEnv.step(), which is the same call show_squad.py and
season_controller.py already make. This file only:

  1. runs one gameweek's decision (fresh reset+predict+step, or
     resume-from-state+predict+step) - identical to one iteration of
     season_controller.py's run_season() loop;
  2. reuses the existing SeasonState class for persistence (its own
     .save()/.load(), unmodified) as the on-disk state store, one JSON file
     per demo session;
  3. formats env.step()'s own returned info dict into a JSON-ready shape,
     enriched with player/team names the same way show_squad.py already does.

No PPOEnv/action/observation/MILP/scoring code is duplicated or modified.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from stable_baselines3 import PPO

from rl_decision_layer.ppo.env import PPOEnv
from rl_decision_layer.ppo.season_controller import SeasonState
from rl_decision_layer.ppo.train import MODELS_DIR
from rl_decision_layer.predictions.interface import load_player_metadata, load_team_metadata

SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL = "ppo_fpl_v4"

# PPO models are expensive to load (disk + deserialize) but are pure,
# stateless policies once loaded - safe to reuse across requests/sessions.
_model_cache: dict[str, PPO] = {}


def _get_model(model_name: str) -> PPO:
    if model_name not in _model_cache:
        _model_cache[model_name] = PPO.load(MODELS_DIR / model_name)
    return _model_cache[model_name]


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def _player_name(pid, players) -> str:
    return players.loc[pid, "player_name"] if pid in players.index else f"#{pid}"


def _player_summary(pid: int, candidates, players, teams, captain_id, vice_captain_id, starting_ids) -> dict:
    pos = candidates.loc[pid, "position"] if pid in candidates.index else "?"
    team_id = candidates.loc[pid, "team_id"] if pid in candidates.index else None
    team = teams.loc[team_id, "team_name"] if team_id in teams.index else "?"
    return {
        "player_id": int(pid),
        "name": _player_name(pid, players),
        "position": pos,
        "team": team,
        "is_starting": pid in starting_ids,
        "is_captain": pid == captain_id,
        "is_vice_captain": pid == vice_captain_id,
    }


def _build_gameweek_response(action, reward, info, pre_candidates, previous_squad_ids,
                              players, teams, is_initial: bool, available_chips: dict) -> dict:
    """Pure formatting of env.step()'s own return values - no decisions made here."""
    act = tuple(int(a) for a in action)
    chip_names = {0: "none", 1: "wildcard", 2: "free_hit", 3: "bench_boost", 4: "triple_captain"}

    if info.get("status") not in (None, "Optimal"):
        return {
            "gameweek": info.get("gameweek"),
            "status": info.get("status"),
            "legal": False,
            "message": f"No legal squad found under this action (status={info.get('status')}).",
        }

    squad_ids = set(info["squad_ids"])
    starting_ids = set(info["starting_ids"])
    captain_id, vice_id = info["captain_id"], info["vice_captain_id"]

    squad = [
        _player_summary(pid, pre_candidates, players, teams, captain_id, vice_id, starting_ids)
        for pid in squad_ids
    ]
    bench = [p for p in squad if not p["is_starting"]]

    transfers_in = [] if is_initial else sorted(squad_ids - previous_squad_ids)
    transfers_out = [] if is_initial else sorted(previous_squad_ids - squad_ids)

    return {
        "gameweek": info["gameweek"],
        "ppo_action": {
            "aggressiveness": act[0],
            "budget_level": act[1],
            "position_bias": act[2] if len(act) > 2 else None,
            "chip_requested": chip_names.get(act[3], "none") if len(act) > 3 else "none",
        },
        "chip_used": info.get("chip_used", "none"),
        "squad": squad,
        "starting_xi": [p for p in squad if p["is_starting"]],
        "bench": bench,
        "captain": _player_name(captain_id, players) if captain_id is not None else None,
        "vice_captain": _player_name(vice_id, players) if vice_id is not None else None,
        "transfers_in": [_player_name(pid, players) for pid in transfers_in],
        "transfers_out": [_player_name(pid, players) for pid in transfers_out],
        "transfers": info["transfers"],
        "hits": info["hits"],
        "bank": info["bank"],
        "free_transfers": info["free_transfers"],
        "available_chips": available_chips,
        "reward": reward,
        "legal": info["legality_violations"] == [],
        "legality_violations": info["legality_violations"],
    }


def start_season(model_name: str = DEFAULT_MODEL, start_gameweek: int = 1, season: str = "2025-26") -> dict:
    """Starts a brand-new sequential season: fresh legal starting squad,
    default bank/free-transfers/chips, GW = start_gameweek. Identical to
    season_controller.py's run_season() with no --load-state."""
    players = load_player_metadata().set_index("player_id")
    teams = load_team_metadata().set_index("team_id")
    model = _get_model(model_name)

    env = PPOEnv(start_gameweek=start_gameweek, num_gameweeks=1, season=season)
    obs, _ = env.reset()
    pre_candidates = env._state.candidates.drop_duplicates("player_id").set_index("player_id")

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    response = _build_gameweek_response(action, reward, info, pre_candidates, set(), players, teams,
                                         is_initial=True, available_chips=dict(env.available_chips))

    session_id = str(uuid.uuid4())
    if response.get("legal", True):
        state = SeasonState(gameweek=info["gameweek"] + 1, squad_ids=sorted(info["squad_ids"]),
                             bank=info["bank"], free_transfers=info["free_transfers"],
                             available_chips=dict(env.available_chips))
        state.save(str(_session_path(session_id)))

    response["session_id"] = session_id
    response["model"] = model_name
    return response


def next_gameweek(session_id: str, model_name: str = DEFAULT_MODEL, season: str = "2025-26") -> dict:
    """Continues an existing session from its saved SeasonState - the exact
    same resume mechanism season_controller.py's --load-state uses."""
    path = _session_path(session_id)
    if not path.exists():
        raise FileNotFoundError(f"No such session: {session_id}")

    players = load_player_metadata().set_index("player_id")
    teams = load_team_metadata().set_index("team_id")
    model = _get_model(model_name)

    prior = SeasonState.load(str(path))
    env = PPOEnv(
        start_gameweek=prior.gameweek, num_gameweeks=1, season=season,
        initial_squad=prior.squad_ids, initial_bank=prior.bank,
        initial_free_transfers=prior.free_transfers, initial_available_chips=prior.available_chips,
    )
    obs, _ = env.reset()
    pre_candidates = env._state.candidates.drop_duplicates("player_id").set_index("player_id")
    previous_squad_ids = set(prior.squad_ids)

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    response = _build_gameweek_response(action, reward, info, pre_candidates, previous_squad_ids,
                                         players, teams, is_initial=False, available_chips=dict(env.available_chips))

    if response.get("legal", True):
        state = SeasonState(gameweek=info["gameweek"] + 1, squad_ids=sorted(info["squad_ids"]),
                             bank=info["bank"], free_transfers=info["free_transfers"],
                             available_chips=dict(env.available_chips))
        state.save(str(path))

    response["session_id"] = session_id
    response["model"] = model_name
    return response


def get_session_state(session_id: str) -> dict:
    path = _session_path(session_id)
    if not path.exists():
        raise FileNotFoundError(f"No such session: {session_id}")
    state = SeasonState.load(str(path))
    return {
        "session_id": session_id,
        "next_gameweek": state.gameweek,
        "bank": state.bank,
        "free_transfers": state.free_transfers,
        "available_chips": state.available_chips,
        "squad_size": len(state.squad_ids),
    }
