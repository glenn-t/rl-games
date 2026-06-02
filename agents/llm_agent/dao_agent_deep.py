"""
Deep-agent Dao player — deepagents SDK, tool-calling loop per turn.

The agent receives board + legal moves in each user message, reasons freely,
then calls make_move() to commit. Illegal moves return an error; the agent
retries within the same invocation.

Usage:
    python dao_agent_deep.py
    python dao_agent_deep.py --quiet
    python dao_agent_deep.py --x-model anthropic:claude-haiku-4-5-20251001
    python dao_agent_deep.py --games 5
"""

from deepagents import create_deep_agent

from dao_game import _ACTION_TO_ACTIONID, _ACTIONID_TO_ACTION, _DIRECTION_TUPLES, _NUM_ROWS, _NUM_COLS
from dao_agent_common import (
    DIRECTIONS, RULES,
    GameState, build_turn_message, build_graph,
    apply_move_and_update, print_turn_header, run_cli,
    landing_square,
)

DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"

SYSTEM_PROMPT = """\
You are playing Dao as player {{player}} ({{symbol}}) against player {{opponent}} ({{opp_symbol}}).

MOVEMENT
  Pieces slide as far as possible in the chosen direction — they cannot stop
  mid-slide or jump over other pieces.
  Valid directions: UP  DOWN  LEFT  RIGHT  UP LEFT  UP RIGHT  DOWN LEFT  DOWN RIGHT

{rules}

Each turn you will receive the current board and full list of legal moves.
Reason through your options, then call make_move() once to commit.
If make_move returns an error, correct your choice and try again.\
""".format(rules=RULES)


def make_player_node(player: int, model: str):
    symbol = "x" if player == 0 else "o"
    system = SYSTEM_PROMPT.replace("{{player}}", str(player)) \
                          .replace("{{symbol}}", symbol) \
                          .replace("{{opponent}}", str(1 - player)) \
                          .replace("{{opp_symbol}}", "o" if player == 0 else "x")

    # Mutable slot so the tool closure can signal back the chosen move
    result = {"last_move": None}

    def make_move(row: int, col: int, direction: str) -> str:
        """
        Commit your move. Slide the piece at (row, col) in the given direction.
        Direction must be one of:
          UP  DOWN  LEFT  RIGHT  UP LEFT  UP RIGHT  DOWN LEFT  DOWN RIGHT
        The piece slides as far as possible until it hits a wall or another piece.
        Returns confirmation on success, or an error — in which case try again.
        """
        direction = direction.strip().upper()
        action = _ACTION_TO_ACTIONID.get((row, col, direction))
        dao = result["state"]
        if action is None or action not in dao.legal_actions():
            return f"ERROR: ({row},{col}) {direction} is not legal. Check the legal moves list and try again."
        lr, lc = landing_square(dao.board_array(), row, col, direction)
        result["last_move"] = (row, col, direction)
        return f"OK: ({row},{col}) → ({lr},{lc}) via {direction}"

    agent = create_deep_agent(model=model, tools=[make_move], system_prompt=system)

    def node(state: GameState) -> dict:
        dao = state["dao_state"]
        result["state"] = dao
        result["last_move"] = None

        if state["verbose"]:
            print_turn_header(player, dao)

        agent.invoke({
            "messages": [{"role": "user", "content": build_turn_message(dao, player)}]
        })

        if result["last_move"] is not None:
            r, c, d = result["last_move"]
            action_id = _ACTION_TO_ACTIONID[(r, c, d)]
        else:
            action_id = dao.legal_actions()[0]
            r, c, d = _ACTIONID_TO_ACTION[action_id]
            if state["verbose"]:
                print(f"  [forfeit]")

        if state["verbose"]:
            print(f"  → {symbol} plays ({r},{c}) {d}")

        return apply_move_and_update(dao, action_id, state, f"Move {dao._move_count}: {symbol} ({r},{c}) {d}")

    return node


def _build_graph(x_model=DEFAULT_MODEL, o_model=DEFAULT_MODEL):
    return build_graph(make_player_node, x_model, o_model)


if __name__ == "__main__":
    run_cli(_build_graph, DEFAULT_MODEL, "Deep-agent Dao player")
