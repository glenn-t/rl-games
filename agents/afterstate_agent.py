"""Afterstate value agent for Dao.

Loads a trained W-table (value for the player who just moved to each canonical
afterstate) and plays greedily by picking the action whose afterstate has the
highest W value.
"""

import numpy as np

from games.dao_hash import canonical_index

INVALID_ACTION = -1


class AfterstateAgent:
    """Greedy agent over a learned afterstate value function."""

    def __init__(self, player_id: int, w_table: np.ndarray):
        self._player_id = player_id
        self._w = w_table  # shape (N_CANONICAL_STATES,)
        # slight issue is that the terminal states are set to -1
        # check the transition of the learning agent
        # positive w means player 0 advantage, negative w means player 1 advantage
        if self._player_id == 1:
            self._w = self._w * -1

    def player_id(self) -> int:
        return self._player_id

    def step(self, state) -> int:
        legal = state.legal_actions()
        if not legal:
            return INVALID_ACTION
        return max(legal, key=lambda a: self._w[_afterstate_idx(state, a)])

    @classmethod
    def load(cls, player_id: int, path: str) -> "AfterstateAgent":
        return cls(player_id, np.load(path))


def _afterstate_idx(state, action: int) -> int:
    """Canonical index of the board reached by taking action in state."""
    return canonical_index(state.board_after_action(action).flatten().astype(np.int8))
