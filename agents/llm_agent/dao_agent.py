"""
LangGraph Dao agent — direct Anthropic API, single call per turn.

Usage:
    python dao_agent.py
    python dao_agent.py --quiet
    python dao_agent.py --x-model claude-haiku-4-5-20251001
    python dao_agent.py --games 5
"""

import re
import argparse
from functools import partial

from anthropic import Anthropic

from games.dao import _ACTION_TO_ACTIONID, _ACTIONID_TO_ACTION
from dao_agent_common import (
    DIRECTIONS, RULES,
    GameState, build_turn_message, build_graph,
    apply_move_and_update, print_turn_header, run_cli,
)

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3

SYSTEM_PROMPT = """\
You are playing Dao, a two-player abstract strategy game on a 4×4 board.

PIECES
  You are player {{player}} ({{symbol}}).
  Your opponent is player {{opponent}} ({{opp_symbol}}).
  Each player always has exactly 4 pieces.

BOARD NOTATION
  Rows 0–3 top to bottom, columns 0–3 left to right.
  x = player 0 pieces   o = player 1 pieces   . = empty

MOVEMENT RULES
  Pick ONE of your pieces and slide it in a straight line
  (orthogonal or diagonal — 8 directions):
    UP  DOWN  LEFT  RIGHT  UP LEFT  UP RIGHT  DOWN LEFT  DOWN RIGHT
  The piece slides as far as possible until it hits a board edge or any
  other piece. It cannot stop mid-slide or jump over pieces.

{rules}

RESPONSE FORMAT
  Output EXACTLY one line — nothing else:
    MOVE (row,col) DIRECTION
  Examples:
    MOVE (2,1) RIGHT
    MOVE (0,3) DOWN LEFT
  Reason silently first if you like, but the final line of your response
  MUST be the MOVE line and nothing else after it.\
""".format(rules=RULES)

_MOVE_RE = re.compile(r"MOVE\s+\((\d)\s*,\s*(\d)\)\s+([A-Z ]+)", re.IGNORECASE)


def parse_move(text: str):
    for line in reversed(text.strip().splitlines()):
        m = _MOVE_RE.search(line.strip())
        if m:
            row, col = int(m.group(1)), int(m.group(2))
            direction = m.group(3).strip().upper()
            if direction in DIRECTIONS:
                return row, col, direction
    return None


def get_action_id(state, text: str):
    parsed = parse_move(text)
    if parsed is None:
        return None, "Could not parse a MOVE line from your response."
    row, col, direction = parsed
    action = _ACTION_TO_ACTIONID.get((row, col, direction))
    if action is None:
        return None, f"({row},{col}) {direction} is not a recognised action."
    if action not in state.legal_actions():
        return None, f"MOVE ({row},{col}) {direction} is not legal in this position."
    return action, None


def make_player_node(player: int, model: str):
    client = Anthropic()
    symbol = "x" if player == 0 else "o"
    system = SYSTEM_PROMPT.replace("{{player}}", str(player)) \
                          .replace("{{symbol}}", symbol) \
                          .replace("{{opponent}}", str(1 - player)) \
                          .replace("{{opp_symbol}}", "o" if player == 0 else "x")

    def node(state: GameState) -> dict:
        dao = state["dao_state"]
        if state["verbose"]:
            print_turn_header(player, dao)

        user_msg = build_turn_message(dao, player) + "\n\nYour move:"
        action_id = None
        last_error = None

        for attempt in range(MAX_RETRIES):
            prompt = user_msg
            if last_error and attempt > 0:
                prompt += f"\n\n[Previous response was invalid: {last_error}. Please try again.]"

            raw = client.messages.create(
                model=model, max_tokens=512, system=system,
                messages=[{"role": "user", "content": prompt}],
            ).content[0].text

            action_id, last_error = get_action_id(dao, raw)
            if action_id is not None:
                break
            if state["verbose"]:
                print(f"  [attempt {attempt+1}] {last_error} | raw: {raw[:80]!r}")

        if action_id is None:
            action_id = dao.legal_actions()[0]
            if state["verbose"]:
                print(f"  [forfeit]")

        r, c, d = _ACTIONID_TO_ACTION[action_id]
        if state["verbose"]:
            print(f"  → {symbol} plays ({r},{c}) {d}")

        return apply_move_and_update(dao, action_id, state, f"Move {dao._move_count}: {symbol} ({r},{c}) {d}")

    return node


def _build_graph(x_model=DEFAULT_MODEL, o_model=DEFAULT_MODEL):
    return build_graph(make_player_node, x_model, o_model)


if __name__ == "__main__":
    run_cli(_build_graph, DEFAULT_MODEL, "LangGraph Dao agent")
