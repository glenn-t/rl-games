"""
Shared components for Dao LLM agents.

Imported by dao_agent.py (direct Anthropic API) and dao_agent_deep.py (deepagents).
"""

import argparse
from typing import Literal

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from games.dao import (
    DaoGame,
    DaoState,
    _ACTION_TO_ACTIONID,
    _ACTIONID_TO_ACTION,
    _DIRECTION_TUPLES,
    _NUM_ROWS,
    _NUM_COLS,
    _PLAYER_TOKENS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Directions
# ─────────────────────────────────────────────────────────────────────────────

DIRECTIONS = {
    "UP", "DOWN", "LEFT", "RIGHT",
    "UP LEFT", "UP RIGHT", "DOWN LEFT", "DOWN RIGHT",
}

# ─────────────────────────────────────────────────────────────────────────────
# Rules text — shared across system prompts
# ─────────────────────────────────────────────────────────────────────────────

RULES = """\
WIN CONDITIONS — first to achieve ANY of these:
  1. LINE    — all 4 of your pieces in the same row or column
  2. SQUARE  — your 4 pieces form a 2×2 block anywhere on the board
  3. CORNERS — your 4 pieces occupy all four corner squares (0,0)(0,3)(3,0)(3,3)
  4. TRAP    — one of the OPPONENT's pieces is in a corner with all 3 adjacent
               squares occupied by YOUR pieces (the trapped piece's owner wins —
               avoid trapping your own pieces, try to trap theirs)

STRATEGY HINTS
  - Two simultaneous threats are hard to defend.
  - Corners are powerful; contest them early.
  - A piece on the board edge has fewer slide options; centralise where possible.
  - Forcing a piece into a corner then filling the 3 adjacent squares wins.\
"""

# ─────────────────────────────────────────────────────────────────────────────
# Board rendering
# ─────────────────────────────────────────────────────────────────────────────

_TOKEN_CH = {0: ".", 1: "x", 2: "o"}


def render_board(state: DaoState) -> str:
    b = state.board_array()
    rows = ["     0   1   2   3"]
    for i in range(_NUM_ROWS):
        cells = "  ".join(f"[{_TOKEN_CH[b[i,j]]}]" for j in range(_NUM_COLS))
        rows.append(f"  {i} {cells}")
    return "\n".join(rows)


def piece_list(state: DaoState, player: int) -> str:
    token = _PLAYER_TOKENS[player]
    b = state.board_array()
    coords = [(r, c) for r in range(_NUM_ROWS) for c in range(_NUM_COLS) if b[r, c] == token]
    return "  ".join(f"({r},{c})" for r, c in coords)


def landing_square(board, row: int, col: int, direction: str) -> tuple[int, int]:
    """Compute where a piece at (row, col) slides to in direction."""
    dr, dc = _DIRECTION_TUPLES[direction]
    r, c = row, col
    while True:
        nr, nc = r + dr, c + dc
        if nr < 0 or nc < 0 or nr >= _NUM_ROWS or nc >= _NUM_COLS or board[nr, nc] != 0:
            break
        r, c = nr, nc
    return r, c


def build_turn_message(state: DaoState, player: int) -> str:
    """Full user message for a player's turn: board + piece lists + legal moves."""
    b = state.board_array()
    symbol = "x" if player == 0 else "o"

    lines = [
        f"--- Board (move {state._move_count}) ---",
        render_board(state),
        "",
        f"x pieces (player 0): {piece_list(state, 0)}",
        f"o pieces (player 1): {piece_list(state, 1)}",
        "",
        f"To move: {symbol} (player {player})",
        "Legal moves:",
    ]

    token = _PLAYER_TOKENS[player]
    for r in range(_NUM_ROWS):
        for c in range(_NUM_COLS):
            if b[r, c] != token:
                continue
            for direction in DIRECTIONS:
                action = _ACTION_TO_ACTIONID.get((r, c, direction))
                if action is None or action not in state.legal_actions():
                    continue
                lr, lc = landing_square(b, r, c, direction)
                lines.append(f"  ({r},{c}) {direction:<10} → ({lr},{lc})")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph plumbing
# ─────────────────────────────────────────────────────────────────────────────

class GameState(TypedDict):
    dao_state: DaoState
    current_player: int
    winner: int | None
    move_log: list[str]
    verbose: bool


def route(state: GameState) -> Literal["player0", "player1", "__end__"]:
    if state["dao_state"].is_terminal():
        return END
    return "player0" if state["current_player"] == 0 else "player1"


def build_graph(make_player_node, x_model: str, o_model: str) -> object:
    builder = StateGraph(GameState)
    builder.add_node("player0", make_player_node(0, x_model))
    builder.add_node("player1", make_player_node(1, o_model))
    builder.set_conditional_entry_point(route)
    builder.add_conditional_edges("player0", route)
    builder.add_conditional_edges("player1", route)
    return builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Game loop + CLI helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_move_and_update(dao: DaoState, action_id: int, state: GameState, log_entry: str) -> dict:
    """Apply action_id to dao and return the GameState update dict."""
    dao.apply_action(action_id)
    updates: dict = {
        "dao_state": dao,
        "move_log": state["move_log"] + [log_entry],
    }
    if dao.is_terminal():
        updates["winner"] = dao.winner()
        updates["current_player"] = -1
    else:
        updates["current_player"] = dao.current_player()
    return updates


def print_turn_header(player: int, dao: DaoState) -> None:
    sym = "x" if player == 0 else "o"
    print(f"\n{'═'*60}")
    print(f"  Player {player} ({sym}) to move — move #{dao._move_count}")
    print('═'*60)
    print(dao)


def print_game_over(dao: DaoState) -> None:
    print(f"\n{'═'*60}")
    print(f"  GAME OVER after {dao._move_count} moves")
    winner = dao.winner()
    if winner is not None:
        print(f"  Winner: player {winner} ({'x' if winner == 0 else 'o'})")
    else:
        print("  Draw / max moves reached")
    print('═'*60)
    print(dao)


def play_game(build_graph_fn, x_model: str, o_model: str, verbose: bool) -> int | None:
    game = DaoGame()
    graph = build_graph_fn(x_model=x_model, o_model=o_model)

    final = graph.invoke({
        "dao_state": game.new_initial_state(),
        "current_player": 0,
        "winner": None,
        "move_log": [],
        "verbose": verbose,
    })

    if verbose:
        print_game_over(final["dao_state"])

    return final["winner"]


def run_cli(build_graph_fn, default_model: str, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--x-model", default=default_model)
    parser.add_argument("--o-model", default=default_model)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--games", type=int, default=1)
    args = parser.parse_args()

    verbose = not args.quiet
    wins = {0: 0, 1: 0, None: 0}

    for g in range(args.games):
        if args.games > 1:
            print(f"\n{'━'*60}\n  Game {g+1}/{args.games}")
        wins[play_game(build_graph_fn, args.x_model, args.o_model, verbose)] += 1

    if args.games > 1:
        print(f"\nResults over {args.games} games:")
        print(f"  x (player 0) wins: {wins[0]}")
        print(f"  o (player 1) wins: {wins[1]}")
        print(f"  Draws:             {wins[None]}")
