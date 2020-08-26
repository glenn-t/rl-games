"""
Naive agent - one step look ahead

Move priority:
1) Win
2) Prevent opponent win on next turn
3) Random
"""

import numpy as np
import pyspiel


class NaiveAgent(pyspiel.Bot):
    """Naive agent class."""

    def __init__(self, player_id, num_actions, name="naive_agent"):

        pyspiel.Bot.__init__(self)
        assert num_actions > 0
        self._player_id = player_id
        self._num_actions = num_actions

    def restart_at(self, state):
        pass

    def player_id(self):
        return self._player_id

    def provides_policy(self):
        return True

    def step_with_policy(self, state):
        """Returns the stochastic policy and selected action in the given state.

        Args:
        state: The current state of the game.

        Returns:
        A `(policy, action)` pair, where policy is a `list` of
        `(action, probability)` pairs for each legal action
        The `action` is selected uniformly at random from the best actions
            (one step look ahead)
        or `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        cur_legal_actions = state.legal_actions(self._player_id)
        if not cur_legal_actions:
            # Empty - no valid actions
            return [], pyspiel.INVALID_ACTION
        # Try each action
        action_rewards = np.zeros(len(cur_legal_actions))
        for i, action in enumerate(cur_legal_actions):
            child = state.child(action)
            if child.is_terminal():
                # Take reward value
                action_rewards[i] = child.player_returns(self._player_id)
            else:
                # Try each opponent move
                child_legal_actions = child.legal_actions()
                child_actions_returns = np.zeros(len(child_legal_actions))
                for j, child_action in enumerate(child_legal_actions):
                    grandchild = child.child(child_action)
                    child_actions_returns[j] = grandchild.player_returns(self._player_id)
                # Assign return for action i as worst return of opponent move
                action_rewards[i] = np.min(child_actions_returns)

        # Filter actions to best actions
        best_action_ids = action_rewards == np.max(action_rewards)
        # Of the best actions, take a random one
        action = cur_legal_actions[np.random.choice(best_action_ids)]
        # Create probability vector
        probs = np.zeros(self._num_actions)
        # Assign equal probability to each maximum action
        probs[np.array(cur_legal_actions)[best_action_ids]] = 1.0 / np.sum(best_action_ids)
        # Choose a random action based on probabilities
        action = np.random.choice(self._num_actions, p=probs)

        return probs, action

    def step(self, state):
        return self.step_with_policy(state)[1]
