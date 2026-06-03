"""Afterstate value agent for Dao.

Loads a trained W-table where values are always from the mover's perspective:
+1 = mover wins, -1 = mover loses. Both players argmax over W.

When player 1 looks up an afterstate, tokens are swapped (1↔2) before hashing
so the lookup is always done as if the mover is player 0. This lets both players
share the same W-table without any sign flip.
"""

import numpy as np

from games.dao_hash import canonical_index

INVALID_ACTION = -1


class AfterstateAgent:
    """Greedy agent over a learned afterstate value function."""

    def __init__(self, player_id: int, w_table: np.ndarray):
        self._player_id = player_id
        self._w = w_table  # shape (N_CANONICAL_STATES,)

    def player_id(self) -> int:
        return self._player_id

    def step(self, state) -> int:
        legal = state.legal_actions()
        if not legal:
            return INVALID_ACTION
        return max(legal, key=lambda a: self._w[_afterstate_idx(state, a, self._player_id)])

    @classmethod
    def load(cls, player_id: int, path: str) -> "AfterstateAgent":
        return cls(player_id, np.load(path))


def _afterstate_idx(state, action: int, player_id: int = 0) -> int:
    """Canonical index of the board reached by taking action in state.

    Passes player_id to board_after_action so the returned board is always
    from player 0's perspective — no swap logic needed here.
    """
    board = state.board_after_action(action, perspective_player=player_id).flatten().astype(np.int8)
    return canonical_index(board)
