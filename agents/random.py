"""Random agent - selects uniformly at random from legal actions."""

import numpy as np
from agents.naive import INVALID_ACTION


class RandomAgent:

    def __init__(self, player_id, rng=None):
        self._player_id = player_id
        self._rng = rng or np.random.default_rng()

    def player_id(self):
        return self._player_id

    def step(self, state):
        actions = state.legal_actions(self._player_id)
        if not actions:
            return INVALID_ACTION
        return self._rng.choice(actions)
