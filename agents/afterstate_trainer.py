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

Initialisation:
  The W-table is pre-seeded with TERMINAL_W_INIT (from dao_hash): all canonical
  states that are winning positions start at W=1.0. This gives the agent
  immediate win-seeking behaviour from episode 1, without any training.

Training phases:
  Phase 1 — optional curriculum (off-policy, --curriculum N):
    Generate games from a fixed agent and run updates. Valid because the target
    does not depend on the behaviour policy. Two options:
      --curriculum-agent naive  (recommended): NaiveAgent vs NaiveAgent.
        ~140ms/game; provides strategic experience (win, block, positional play).
      --curriculum-agent random: random vs random. Not recommended — equivalent
        to self-play at epsilon=1.0, which Phase 2 already covers.
  Phase 2 — epsilon-greedy self-play (--selfplay N):
    Both players share the W-table and select moves greedily with probability
    1-epsilon, decaying from epsilon_start to epsilon_min.
    --mix and --mix-agent let a fraction of episodes pit the W-table agent
    against a fixed opponent (e.g. 50% self-play + 50% vs NaiveAgent).
    The fixed-agent side's moves are also used for W-table updates (off-policy).

Usage:
    # Recommended: terminal init + self-play only
    python -m agents.afterstate_trainer --selfplay 200000

    # With naive curriculum for stronger opening play
    python -m agents.afterstate_trainer --curriculum 50000 --curriculum-agent naive --selfplay 150000

    # Mixed self-play: 50% vs naive, 50% self-play
    python -m agents.afterstate_trainer --selfplay 200000 --mix 0.5 --mix-agent naive

    # Smoke test
    python -m agents.afterstate_trainer --selfplay 300 --eval-every 100
"""

import argparse
import os
import time

import numpy as np

from games.dao import DaoGame
from games.dao_hash import canonical_index, N_CANONICAL_STATES, TERMINAL_W_INIT
from agents.naive import NaiveAgent
from agents.random import RandomAgent
from agents.afterstate_agent import AfterstateAgent, _afterstate_idx


def _update_w_mc(steps, returns, w_table, alpha, gamma):
    """Monte Carlo update: assign actual discounted return to every afterstate."""
    for p in range(2):
        p_indices = [idx for (idx, pl) in steps if pl == p]
        r_terminal = float(returns[p])
        if p == 1:
            r_terminal = r_terminal * -1
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
        if p == 1:
            r_terminal = r_terminal * -1
        for i in range(len(p_indices) - 1, -1, -1):
            idx = p_indices[i]
            if i == len(p_indices) - 1:
                target = r_terminal
            else:
                target = gamma * w_table[p_indices[i + 1]]
            w_table[idx] += alpha * (target - w_table[idx])


_UPDATE_FNS = {"mc": _update_w_mc, "td": _update_w_td}


def _evaluate(game, w_table, opponent_factory, n_games=200, rng=None):
    """Returns (win_rate, loss_rate, draw_rate) against opponent, alternating sides."""
    if rng is None:
        rng = np.random.default_rng()
    wins = losses = draws = 0
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
        winner = state.winner()
        if winner == our_side:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1
    return wins / n_games, losses / n_games, draws / n_games


def _fmt_eval(win, loss, draw):
    return f"{win:.1%}W/{loss:.1%}L/{draw:.1%}D"


def _print_eval(ep, total, elapsed, epsilon, w_table, game, rng, alpha=None):
    wr, lr, dr = _evaluate(game, w_table, lambda pid, r: RandomAgent(pid, rng=r),
                            n_games=200, rng=rng)
    wn, ln, dn = _evaluate(game, w_table, lambda pid, _: NaiveAgent(pid, num_actions=128),
                            n_games=200, rng=rng)
    eps_str = f"eps={epsilon:.4f} | " if epsilon is not None else ""
    alpha_str = f"alpha={alpha:.4f} | " if alpha is not None else ""
    print(
        f"  Ep {ep:>7}/{total} | {eps_str}{alpha_str}"
        f"vs Random: {_fmt_eval(wr, lr, dr)} | "
        f"vs Naive: {_fmt_eval(wn, ln, dn)} | "
        f"{elapsed:.0f}s",
        flush=True,
    )


def train(
    curriculum_episodes=0,
    curriculum_agent="naive",
    selfplay_episodes=200_000,
    mix_ratio=0.0,
    mix_agent="naive",
    update_method="mc",
    alpha=0.1,
    alpha_final=None,
    gamma=0.99,
    epsilon_start=None,
    epsilon_min=0.05,
    eval_every=5_000,
    output_path="trained_models/w_table.npy",
    checkpoint=None,
    seed=None,
):
    update_w = _UPDATE_FNS[update_method]

    rng = np.random.default_rng(seed)
    game = DaoGame(max_game_length=200, track_history=False)

    if checkpoint is not None:
        w_table = np.load(checkpoint).astype(np.float32)
        n_nonzero = int((w_table != 0).sum())
        if epsilon_start is None:
            epsilon_start = epsilon_min
            print(f"Loaded checkpoint: {checkpoint} "
                  f"({n_nonzero} non-zero states, {n_nonzero / N_CANONICAL_STATES:.1%} of {N_CANONICAL_STATES}). "
                  f"Resuming with epsilon={epsilon_start} (use --epsilon-start to override).",
                  flush=True)
        else:
            print(f"Loaded checkpoint: {checkpoint} "
                  f"({n_nonzero} non-zero states, {n_nonzero / N_CANONICAL_STATES:.1%} of {N_CANONICAL_STATES}).",
                  flush=True)
    else:
        if epsilon_start is None:
            epsilon_start = 1.0
        w_table = TERMINAL_W_INIT.copy()
        n_terminal = int((w_table > 0).sum())
        print(f"W-table initialised: {n_terminal} terminal winning states set to 1.0 "
              f"({n_terminal / N_CANONICAL_STATES:.1%} of {N_CANONICAL_STATES} states).", flush=True)

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
            f"\nPhase 1: {curriculum_episodes} curriculum episodes "
            f"({curriculum_agent} vs {curriculum_agent}, update={update_method}, gamma={gamma})...",
            flush=True,
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

            if eval_every and (ep + 1) % eval_every == 0:
                _print_eval(ep + 1, curriculum_episodes, time.time() - t0,
                            None, w_table, game, rng)
            elif (ep + 1) % log_every == 0:
                print(f"  {ep + 1:>7}/{curriculum_episodes} ({time.time() - t0:.0f}s)", flush=True)

        print(f"Phase 1 complete ({time.time() - t0:.0f}s).", flush=True)

    # --- Phase 2: Epsilon-greedy self-play (with optional agent mix) ---
    if selfplay_episodes > 0:
        alpha_decaying = alpha_final is not None and alpha_final != alpha
        alpha_cur = alpha
        alpha_decay_rate = (
            (alpha_final / alpha) ** (1.0 / max(selfplay_episodes, 1))
            if alpha_decaying else 1.0
        )
        alpha_eff_min = alpha_final if alpha_decaying else alpha

        mix_str = f", mix={mix_ratio:.0%} {mix_agent}" if mix_ratio > 0 else ""
        alpha_str = f", alpha {alpha} -> {alpha_final}" if alpha_decaying else f", alpha {alpha}"
        print(
            f"\nPhase 2: {selfplay_episodes} self-play episodes "
            f"(update={update_method}, gamma={gamma}, eps {epsilon_start} -> {epsilon_min}"
            f"{alpha_str}{mix_str})...",
            flush=True,
        )

        mix_agents = None
        if mix_ratio > 0:
            if mix_agent == "naive":
                mix_agents = [NaiveAgent(0, num_actions=game.num_distinct_actions()),
                              NaiveAgent(1, num_actions=game.num_distinct_actions())]
            else:
                mix_agents = [RandomAgent(0, rng=rng), RandomAgent(1, rng=rng)]

        epsilon = epsilon_start
        decay = (epsilon_min / epsilon_start) ** (1.0 / max(selfplay_episodes, 1))
        t0 = time.time()

        for ep in range(selfplay_episodes):
            # Decide episode type: pure self-play or W-table vs fixed agent
            if mix_agents and rng.random() < mix_ratio:
                w_side = int(rng.integers(2))  # which side uses the W-table
            else:
                w_side = None  # both sides use the W-table

            state = game.new_initial_state()
            steps = []
            while not state.is_terminal():
                p = state.current_player()

                if w_side is not None and p != w_side:
                    # Fixed agent's turn
                    action = mix_agents[p].step(state)
                    w_idx = _afterstate_idx(state, action)
                else:
                    # Epsilon-greedy W-table turn
                    legal = state.legal_actions()
                    if rng.random() < epsilon:
                        action = int(rng.choice(legal))
                        w_idx = _afterstate_idx(state, action)
                    else:
                        action, w_idx = None, None
                        best_w = -np.inf
                        sign = 1 if p == 0 else -1
                        for a in legal:
                            idx = _afterstate_idx(state, a)
                            w = w_table[idx] * sign
                            if w > best_w:
                                best_w, action, w_idx = w, a, idx

                steps.append((w_idx, p))
                state.apply_action(action)

            update_w(steps, state.returns(), w_table, alpha_cur, gamma)
            epsilon = max(epsilon_min, epsilon * decay)
            alpha_cur = max(alpha_eff_min, alpha_cur * alpha_decay_rate)

            if eval_every and (ep + 1) % eval_every == 0:
                _print_eval(ep + 1, selfplay_episodes, time.time() - t0,
                            epsilon, w_table, game, rng,
                            alpha=alpha_cur if alpha_decaying else None)

        print(f"Phase 2 complete ({time.time() - t0:.0f}s).", flush=True)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(output_path, w_table)
    print(f"\nSaved W-table ({N_CANONICAL_STATES} states) to {output_path}.")
    return w_table


def main():
    parser = argparse.ArgumentParser(
        description="Train an afterstate agent for Dao.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Fast: terminal init + self-play only (~10 min)\n"
            "  python -m agents.afterstate_trainer --selfplay 200000\n\n"
            "  # Quality: naive curriculum + self-play (~3 hrs total)\n"
            "  python -m agents.afterstate_trainer --curriculum 50000 "
            "--curriculum-agent naive --selfplay 150000\n\n"
            "  # Smoke test\n"
            "  python -m agents.afterstate_trainer --selfplay 300 --eval-every 100"
        ),
    )
    parser.add_argument("--curriculum", type=int, default=0,
                        help="Number of off-policy curriculum episodes (default: 0)")
    parser.add_argument("--curriculum-agent", choices=["naive", "random"], default="naive",
                        help="Agent for curriculum games: naive (~140ms/game, recommended) "
                             "or random (~2ms/game, no benefit over self-play)")
    parser.add_argument("--selfplay", type=int, default=200_000,
                        help="Number of epsilon-greedy self-play episodes")
    parser.add_argument("--mix", type=float, default=0.0,
                        help="Fraction of self-play episodes played against a fixed opponent "
                             "(0.0 = pure self-play, 0.5 = half vs fixed agent)")
    parser.add_argument("--mix-agent", choices=["naive", "random"], default="naive",
                        help="Fixed opponent used in mixed episodes")
    parser.add_argument("--update-method", choices=["mc", "td"], default="mc",
                        help="mc: Monte Carlo returns (default, recommended); td: TD(0)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--alpha-final", type=float, default=None,
                        help="Final learning rate for exponential decay (default: constant)")
    parser.add_argument("--epsilon-start", type=float, default=None,
                        help="Starting epsilon for self-play (default: 1.0 for fresh training, "
                             "epsilon-min when resuming a checkpoint)")
    parser.add_argument("--epsilon-min", type=float, default=0.05,
                        help="Minimum (final) epsilon for self-play (default: 0.05)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (<1 encourages faster wins; 1.0 = no discounting)")
    parser.add_argument("--eval-every", type=int, default=5_000,
                        help="Evaluate every N self-play episodes (0 to disable)")
    parser.add_argument("--output", default="trained_models/w_table.npy",
                        help="Path to save the trained W-table (.npy)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to an existing W-table (.npy) to resume training from")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    train(
        curriculum_episodes=args.curriculum,
        curriculum_agent=args.curriculum_agent,
        selfplay_episodes=args.selfplay,
        mix_ratio=args.mix,
        mix_agent=args.mix_agent,
        update_method=args.update_method,
        alpha=args.alpha,
        alpha_final=args.alpha_final,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        eval_every=args.eval_every,
        output_path=args.output,
        checkpoint=args.checkpoint,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
