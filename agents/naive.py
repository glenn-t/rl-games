"""
Naive agent - one step look ahead
Move priority:
1) Win
2) Prevent opponent win on next turn
3) Random
"""
import numpy as np

INVALID_ACTION = -1


class NaiveAgent:
    """Naive one-step lookahead agent."""

    def __init__(self, player_id, num_actions, rng):
        assert num_actions > 0
        self._player_id = player_id
        self._num_actions = num_actions

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

    def player_id(self):
        return self._player_id

    def step_with_policy(self, state):
        """Returns (policy, action) for the given state.

        Policy is a list of (action, probability) pairs over all legal actions.
        Action is sampled uniformly from the best actions under one-step lookahead.

        Returns ([], INVALID_ACTION) if no legal actions are available.
        """
        cur_legal_actions = state.legal_actions(self._player_id)
        if not cur_legal_actions:
            return [], INVALID_ACTION

        action_rewards = np.zeros(len(cur_legal_actions))

        for i, action in enumerate(cur_legal_actions):
            child = state.child(action)
            if child.is_terminal():
                action_rewards[i] = child.player_return(self._player_id)
            else:
                # Assume opponent plays; take the worst case outcome
                child_legal_actions = child.legal_actions()
                child_returns = np.array([
                    child.child(child_action).player_return(self._player_id)
                    for child_action in child_legal_actions
                ])
                action_rewards[i] = np.min(child_returns)

        # Select best actions and assign uniform probability over them
        best_mask = action_rewards == np.max(action_rewards)
        probs = np.zeros(self._num_actions)
        probs[np.array(cur_legal_actions)[best_mask]] = 1.0 / np.sum(best_mask)

        action = self._rng.choice(self._num_actions, p=probs)
        policy = list(zip(cur_legal_actions, probs[cur_legal_actions]))
        return policy, action

    def step(self, state):
        return self.step_with_policy(state)[1]