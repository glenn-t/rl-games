"""
Run Dao games between combinations of agents.

Usage:
    python run_game.py
    python run_game.py --player1 human --player2 naive
    python run_game.py --player1 naive --player2 random --num_games 100 --quiet
"""

import argparse
import collections
import sys
import numpy as np

from games.dao import DaoGame
from agents.naive import NaiveAgent, INVALID_ACTION
from agents.random import RandomAgent
from agents.afterstate_agent import AfterstateAgent


class HumanAgent:
    """Prompts the user to enter a move."""

    def __init__(self, player_id):
        self._player_id = player_id

    def player_id(self):
        return self._player_id

    def step(self, state):
        legal = state.legal_actions(self._player_id)
        print("\nLegal actions:")
        for action in legal:
            print(f"  {action}: {state.action_to_string(action, self._player_id)}")
        while True:
            try:
                choice = int(input("Enter action ID: ").strip())
                if choice in legal:
                    return choice
                print("Not a legal action, try again.")
            except ValueError:
                print("Please enter a number.")


# ---------------------------------------------------------------------------
# Bot factory
# ---------------------------------------------------------------------------

def _make_agent(agent_type, player_id, game, rng):
    if agent_type == "naive":
        return NaiveAgent(player_id, num_actions=game.num_distinct_actions())
    if agent_type == "random":
        return RandomAgent(player_id, rng=rng)
    if agent_type == "human":
        return HumanAgent(player_id)
    if agent_type == "afterstate":
        # return AfterstateAgent.load(player_id, "trained_models/w_100_s1_naive_td.npy")
        return AfterstateAgent.load(player_id, "trained_models/w_100_s1_2m_td.npy")
    raise ValueError(f"Unknown agent type: {agent_type}")


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def _opt_print(quiet, *args, **kwargs):
    if not quiet:
        print(*args, **kwargs)


def play_game(game, agents, quiet=False):
    """Plays one game and returns (returns, history)."""
    state = game.new_initial_state()
    _opt_print(quiet, f"Initial state:\n{state}\n")

    history = []

    while not state.is_terminal():
        current_player = state.current_player()
        agent = agents[current_player]
        action = agent.step(state)

        if action == INVALID_ACTION:
            raise RuntimeError(f"Agent for player {current_player} returned INVALID_ACTION")

        action_str = state.action_to_string(action, current_player)
        _opt_print(quiet, f"Player {current_player} ({type(agent).__name__}): {action_str}")

        state.apply_action(action)

        _opt_print(quiet, f"\n{state}\n")

    returns = state.returns()
    winner = state.winner()
    if winner is not None:
        _opt_print(quiet, f"Player {winner} wins. Returns: {returns}")
    else:
        _opt_print(quiet, f"Draw (max game length reached). Returns: {returns}")

    return returns, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Play Dao between agents.")
    parser.add_argument("--player1", default="naive",
                        choices=["naive", "random", "human", "afterstate"],
                        help="Agent type for player 0")
    parser.add_argument("--player2", default="naive",
                        choices=["naive", "random", "human", "afterstate"],
                        help="Agent type for player 1")
    parser.add_argument("--num_games", type=int, default=1,
                        help="Number of games to play")
    parser.add_argument("--max_game_length", type=int, default=200,
                        help="Maximum number of turns before draw")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-move output")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    game = DaoGame(max_game_length=args.max_game_length)

    agents = [
        _make_agent(args.player1, 0, game, rng),
        _make_agent(args.player2, 1, game, rng),
    ]

    histories = collections.defaultdict(int)
    overall_returns = [0.0, 0.0]
    overall_wins = [0, 0]
    draws = 0

    try:
        for game_num in range(args.num_games):
            returns, history = play_game(game, agents, quiet=args.quiet)
            histories[" ".join(history)] += 1
            for i, v in enumerate(returns):
                overall_returns[i] += v
                if v > 0:
                    overall_wins[i] += 1
            if all(v == 0 for v in returns):
                draws += 1
            if returns[0] <= 0:
                break

    except KeyboardInterrupt:
        print("\nStopped early.")

    print("\n--- Results ---")
    print(f"Games played:        {args.num_games}")
    print(f"Distinct games:      {len(histories)}")
    print(f"Players:             {args.player1} (p0) vs {args.player2} (p1)")
    print(f"Wins [p0, p1]:       {overall_wins}")
    print(f"Draws:               {draws}")
    print(f"Total returns [p0, p1]: {overall_returns}")


if __name__ == "__main__":
    main()