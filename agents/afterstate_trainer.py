"""Train an afterstate value function for Dao.

Value convention: W[canonical_index(s')] = expected return for the player who
just moved to afterstate s'. Higher = better for the mover.

Two update methods:

  mc  (Monte Carlo, default): use the actual discounted game return.
        W(s_k') <- W(s_k') + alpha * (gamma^(n-1-k) * r - W(s_k'))
      No bootstrapping — every afterstate gets a direct signal each episode.
      Best for sparse rewards. With gamma < 1 the agent prefers faster wins.

  td  (TD(0)): bootstrap off the next afterstate estimate.
        W(s_k') <- W(s_k') + alpha * (gamma * W(s_{k+1}') - W(s_k'))
      Lower variance once values are meaningful, but slow to propagate signal
      from zero-initialised tables through long games.

Training phases:
  Phase 1 (off-policy curriculum): generate games from the curriculum agent
    (default: RandomAgent; use --curriculum-agent naive for higher-quality
    experience at ~100x slower) and run updates — valid because the target
    does not depend on the behaviour policy.
  Phase 2 (epsilon-greedy self-play): epsilon-greedy self-play with a decaying
    epsilon, warm-started from Phase 1.

Usage:
    python -m agents.afterstate_trainer
    python -m agents.afterstate_trainer --update-method td --gamma 0.99
    python -m agents.afterstate_trainer --curriculum 50000 --selfplay 150000
    python -m agents.afterstate_trainer --curriculum 1000 --selfplay 1000  # smoke test
"""

import argparse
import os
import time

import numpy as np

from games.dao import DaoGame
from games.dao_hash import canonical_index, N_CANONICAL_STATES
from agents.naive import NaiveAgent
from agents.random import RandomAgent
from agents.afterstate_agent import AfterstateAgent, _afterstate_idx


def _update_w_mc(steps, returns, w_table, alpha, gamma):
    """Monte Carlo update: assign actual discounted return to every afterstate."""
    for p in range(2):
        p_indices = [idx for (idx, pl) in steps if pl == p]
        r_terminal = float(returns[p])
        n = len(p_indices)
        for k, idx in enumerate(p_indices):
            target = (gamma ** (n - 1 - k)) * r_terminal
            w_table[idx] += alpha * (target - w_table[idx])


def _update_w_td(steps, returns, w_table, alpha, gamma):
    """TD(0) update: bootstrap off the next afterstate estimate.

    Processed in reverse so each target uses the just-updated next value.
    """
    for p in range(2):
        p_indices = [idx for (idx, pl) in steps if pl == p]
        r_terminal = float(returns[p])
        for i in range(len(p_indices) - 1, -1, -1):
            idx = p_indices[i]
            if i == len(p_indices) - 1:
                target = r_terminal
            else:
                target = gamma * w_table[p_indices[i + 1]]
            w_table[idx] += alpha * (target - w_table[idx])


_UPDATE_FNS = {"mc": _update_w_mc, "td": _update_w_td}


def _evaluate(game, w_table, opponent_factory, n_games=200, rng=None):
    """Win rate of AfterstateAgent against opponent, alternating sides."""
    if rng is None:
        rng = np.random.default_rng()
    wins = 0
    for trial in range(n_games):
        our_side = trial % 2
        agents = {
            our_side: AfterstateAgent(our_side, w_table),
            1 - our_side: opponent_factory(1 - our_side, rng),
        }
        state = game.new_initial_state()
        while not state.is_terminal():
            p = state.current_player()
            state.apply_action(agents[p].step(state))
        if state.winner() == our_side:
            wins += 1
    return wins / n_games


def train(
    curriculum_episodes=50_000,
    curriculum_agent="random",
    selfplay_episodes=150_000,
    update_method="mc",
    alpha=0.1,
    gamma=0.99,
    epsilon_start=0.3,
    epsilon_min=0.05,
    eval_every=5_000,
    output_path="trained_models/w_table.npy",
    seed=None,
):
    update_w = _UPDATE_FNS[update_method]

    rng = np.random.default_rng(seed)
    game = DaoGame(max_game_length=200)
    w_table = np.zeros(N_CANONICAL_STATES, dtype=np.float32)

    # --- Phase 1: Curriculum (off-policy) ---
    if curriculum_episodes > 0:
        if curriculum_agent == "naive":
            cur_agents = [
                NaiveAgent(0, num_actions=game.num_distinct_actions()),
                NaiveAgent(1, num_actions=game.num_distinct_actions()),
            ]
        else:
            cur_agents = [RandomAgent(0, rng=rng), RandomAgent(1, rng=rng)]

        print(
            f"Phase 1: {curriculum_episodes} curriculum episodes "
            f"({curriculum_agent} vs {curriculum_agent}, update={update_method}, gamma={gamma})..."
        )
        t0 = time.time()
        log_every = max(1, curriculum_episodes // 10)
        for ep in range(curriculum_episodes):
            state = game.new_initial_state()
            steps = []
            while not state.is_terminal():
                p = state.current_player()
                action = cur_agents[p].step(state)
                steps.append((_afterstate_idx(state, action), p))
                state.apply_action(action)
            update_w(steps, state.returns(), w_table, alpha, gamma)

            if (ep + 1) % log_every == 0:
                print(f"  {ep + 1:>7}/{curriculum_episodes} ({time.time() - t0:.0f}s)", flush=True)

        print(f"Phase 1 complete ({time.time() - t0:.0f}s).\n")

    # --- Phase 2: Epsilon-greedy self-play ---
    if selfplay_episodes > 0:
        print(
            f"Phase 2: {selfplay_episodes} self-play episodes "
            f"(update={update_method}, gamma={gamma}, eps {epsilon_start} -> {epsilon_min})..."
        )
        epsilon = epsilon_start
        decay = (epsilon_min / epsilon_start) ** (1.0 / max(selfplay_episodes, 1))
        t0 = time.time()

        for ep in range(selfplay_episodes):
            state = game.new_initial_state()
            steps = []
            while not state.is_terminal():
                p = state.current_player()
                legal = state.legal_actions()
                if rng.random() < epsilon:
                    action = int(rng.choice(legal))
                    w_idx = _afterstate_idx(state, action)
                else:
                    action, w_idx = None, None
                    best_w = -np.inf
                    for a in legal:
                        idx = _afterstate_idx(state, a)
                        w = w_table[idx]
                        if w > best_w:
                            best_w, action, w_idx = w, a, idx
                steps.append((w_idx, p))
                state.apply_action(action)
            update_w(steps, state.returns(), w_table, alpha, gamma)
            epsilon = max(epsilon_min, epsilon * decay)

            if eval_every and (ep + 1) % eval_every == 0:
                wr_random = _evaluate(
                    game, w_table,
                    lambda pid, r: RandomAgent(pid, rng=r),
                    n_games=200, rng=rng,
                )
                wr_naive = _evaluate(
                    game, w_table,
                    lambda pid, _: NaiveAgent(pid, num_actions=128),
                    n_games=200, rng=rng,
                )
                print(
                    f"  Ep {ep + 1:>7}/{selfplay_episodes} | "
                    f"eps={epsilon:.4f} | "
                    f"vs Random: {wr_random:.1%} | "
                    f"vs Naive: {wr_naive:.1%} | "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )

        print(f"Phase 2 complete ({time.time() - t0:.0f}s).\n")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(output_path, w_table)
    print(f"Saved W-table ({N_CANONICAL_STATES} states) to {output_path}.")
    return w_table


def main():
    parser = argparse.ArgumentParser(description="Train an afterstate agent for Dao.")
    parser.add_argument("--curriculum", type=int, default=50_000,
                        help="Number of off-policy curriculum episodes")
    parser.add_argument("--curriculum-agent", choices=["random", "naive"], default="random",
                        help="Agent for curriculum games (random: ~2ms/game; naive: ~140ms/game)")
    parser.add_argument("--selfplay", type=int, default=150_000,
                        help="Number of epsilon-greedy self-play episodes")
    parser.add_argument("--update-method", choices=["mc", "td"], default="mc",
                        help="mc: Monte Carlo returns (recommended for sparse rewards); "
                             "td: TD(0) bootstrapping")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor. With mc, values <1 encourage faster wins. "
                             "Use 1.0 for no discounting.")
    parser.add_argument("--eval-every", type=int, default=5_000,
                        help="Evaluate every N self-play episodes (0 to disable)")
    parser.add_argument("--output", default="trained_models/w_table.npy",
                        help="Path to save the trained W-table (.npy)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    train(
        curriculum_episodes=args.curriculum,
        curriculum_agent=args.curriculum_agent,
        selfplay_episodes=args.selfplay,
        update_method=args.update_method,
        alpha=args.alpha,
        gamma=args.gamma,
        eval_every=args.eval_every,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
