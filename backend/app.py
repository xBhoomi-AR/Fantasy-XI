"""Minimal FastAPI backend exposing the existing, finalized sequential RL
pipeline (PPOEnv + ppo_fpl_v4.zip + season_controller.py's SeasonState) as a
small HTTP API for a future frontend.

This file contains NO squad-selection/transfer/chip/scoring logic itself -
every endpoint just calls into rl_bridge.py, which in turn calls the exact
same PPOEnv/SeasonState machinery already validated by
rl_decision_layer/tests/test_season_controller.py.

Run locally with:
    uvicorn backend.app:app --reload --port 8000

Then, e.g.:
    curl -X POST http://localhost:8000/season/start
    curl -X POST http://localhost:8000/season/<session_id>/next
    curl http://localhost:8000/season/<session_id>/state
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.types import Scope

from backend import rl_bridge

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(title="Fantasy XI RL Backend", version="0.1.0")

# The frontend is normally served by this same app (mounted below) at "/",
# so it's same-origin and CORS isn't actually needed for that path. This is
# only here as a small, permissive safety net for local development if
# someone instead serves frontend/ from a separate dev server/port (e.g.
# `python -m http.server` in frontend/) - never used for anything beyond
# this localhost demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartSeasonRequest(BaseModel):
    model: str = rl_bridge.DEFAULT_MODEL
    start_gameweek: int = 1
    season: str = "2025-26"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/season/start")
def start_season(body: StartSeasonRequest) -> dict:
    """Starts a brand-new sequential season and returns GW1's full result
    (squad, starting XI, captain/vice, bank, free transfers, chips, reward).
    The returned session_id must be used for every subsequent /next call."""
    return rl_bridge.start_season(model_name=body.model, start_gameweek=body.start_gameweek, season=body.season)


@app.post("/season/{session_id}/next")
def next_gameweek(session_id: str, model: str = rl_bridge.DEFAULT_MODEL, season: str = "2025-26") -> dict:
    """Continues the given session from its saved sequential state (squad,
    bank, free transfers, chip availability all carried forward - not an
    independent per-gameweek optimization) and returns the next gameweek's
    full result."""
    try:
        return rl_bridge.next_gameweek(session_id, model_name=model, season=season)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/season/{session_id}/state")
def season_state(session_id: str) -> dict:
    """Returns the session's current saved state (next gameweek, bank, free
    transfers, chip availability, squad size) without advancing it."""
    try:
        return rl_bridge.get_session_state(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class NoCacheStaticFiles(StaticFiles):
    """Starlette's StaticFiles sends ETag/Last-Modified but no Cache-Control,
    so a browser is free to reuse a stale copy of index.html/styles.css/app.js
    without even asking the server first. This is actively edited during
    development, so every response is marked no-cache: the browser must
    always revalidate with the server (a cheap 304 if unchanged) instead of
    silently serving whatever it rendered last time."""

    async def get_response(self, path: str, scope: Scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


# Serves the static frontend (frontend/index.html + styles.css + app.js) at
# "/" - mounted last so it never shadows the API routes above. This means
# `uvicorn backend.app:app` alone serves the whole demo (API + UI) from one
# process/port; no separate frontend dev server is required.
if FRONTEND_DIR.exists():
    app.mount("/", NoCacheStaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
