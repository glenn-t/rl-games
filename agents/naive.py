"""
Naive agent - one step look ahead

Move priority:
1) Win
2) Prevent opponent win on next turn
3) Random
"""

import numpy as np
from open_spiel.python import rl_agent
import pdb

# TODO needs to be pyspiel.Bot class


class NaiveAgent(rl_agent.AbstractAgent):
    """Naive agent class."""

    def __init__(self, player_id, num_actions, name="naive_agent"):
        assert num_actions > 0
        self._player_id = player_id
        self._num_actions = num_actions

    def step(self, time_step, is_evaluation=False):
        # If it is the end of the episode, don't select an action.
        if time_step.is_terminal():
            return

        # Pick a random legal action.
        cur_legal_actions = time_step.legal_actions(self._player_id)
        # Try each action
        action_rewards = np.zeros(len(cur_legal_actions))
        for i, action in enumerate(cur_legal_actions):
            child = time_step.child(action)
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
        probs = np.zeros(self._num_actions)
        probs[np.array(cur_legal_actions)[best_action_ids]] = 1.0 / len(best_action_ids)

        return rl_agent.StepOutput(action=action, probs=probs)
