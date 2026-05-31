# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A reinforcement learning board game platform featuring the game [Dao](https://boardgamegeek.com/boardgame/948/dao) (4x4 abstract strategy game). The project has three layers:

- **`games/`** — Pure Python game logic (no external framework dependency). `dao.py` is the primary game implementation.
- **`agents/`** — AI agents that implement a `step(state) -> action` interface. Currently: `NaiveAgent` (one-step lookahead) and `RandomAgent` (uniform random).
- **`backend/`** — FastAPI server exposing the game over HTTP. `game_manager.py` manages in-memory game sessions (keyed by UUID). `main.py` defines REST endpoints.
- **`frontend/`** — React + Vite frontend (plain JS/TS, no framework beyond React).

## Commands

### Backend

```bash
# Install dependencies (uses uv)
uv sync

# Run backend server (auto-reload)
uv run serve
# or
uv run uvicorn backend.main:app --reload --port 8000

# Run game from CLI (no server needed)
python dao_test.py
python dao_test.py --player1 naive --player2 random --num_games 100 --quiet
```

### Frontend

```bash
cd frontend
npm install
npm run dev      # dev server on http://localhost:5173
npm run build    # production build
```

## Architecture

### Game state (`games/dao.py`)

`DaoGame` is a factory; `DaoState` holds all mutable state. Key methods:
- `legal_actions(player_id)` — returns list of valid action IDs
- `apply_action(action_id)` — mutates state in place
- `child(action_id)` — returns a copy with the action applied (used by agents for lookahead)
- `is_terminal()` / `player_return(player_id)` — terminal detection and scoring

Actions are encoded as integers via a static mapping: `(row, col, direction) <-> action_id` (128 total: 16 cells × 8 directions). The mapping is built at module load in `_ACTIONID_TO_ACTION` / `_ACTION_TO_ACTIONID`.

Win conditions: occupy all 4 corners, form a 2×2 square, or align all 4 tokens in a row/column/diagonal.

### Agent interface

Agents must expose:
- `player_id() -> int`
- `step(state) -> int` (returns an action ID, or `INVALID_ACTION = -1`)

### Backend session model

`GameManager` (singleton `game_manager`) stores `GameSession` objects by UUID string. Sessions hold a `DaoState`, the game mode (`human_vs_human`, `human_vs_ai`, `ai_vs_ai`), and references to agent objects. State is in-memory only — sessions are lost on server restart.

### API surface

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/game/new` | Create session |
| GET | `/api/game/{id}` | Get state |
| POST | `/api/game/{id}/move` | Human move |
| POST | `/api/game/{id}/ai-move` | Trigger AI move |
| GET | `/api/agents` | List available agents |
| DELETE | `/api/game/{id}` | Delete session |

API docs available at `http://localhost:8000/docs` when server is running.
