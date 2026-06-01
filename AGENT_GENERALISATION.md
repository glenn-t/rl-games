# Agent Generalisation Plan

Design notes from conversation on 2026-06-02. Do not implement until ready —
this is a planning document, not a spec.

---

## Goal

Generalise the afterstate trainer so it can train agents across multiple games,
and lay groundwork for:
1. PettingZoo environment wrappers (for neural-net / model-free algorithms)
2. A neural-network afterstate trainer (for games too large for a table)

---

## Layer diagram

```
games/dao.py   games/ttt.py   games/<future>.py
      |               |
  dao_hash.py    ttt_hash.py          ← tabular indexing (small games)
      |               |
      +-------+-------+
              |
         GameSpec                     ← trainer-agnostic game descriptor
              |
    +---------+-----------+
    |                     |
afterstate_trainer.py   nn_afterstate_trainer.py  (future)
(tabular, W-table)      (neural network)

              |
         games/dao_pz.py  games/ttt_pz.py   (future)
              |
    PettingZoo-compatible algorithms (PPO, MADDPG, …)
```

---

## Phase 1 — Tabular generalisation (implement first)

### 1. `agents/game_spec.py`

```python
@dataclass
class GameSpec:
    name: str
    game_factory: Callable[[], Any]        # () -> game instance
    n_states: int                          # W-table size (tabular only)
    terminal_w_init: np.ndarray            # pre-seeded W-table, shape (n_states,)
    afterstate_idx: Callable               # (state, action) -> int  [tabular]
    afterstate_obs: Callable               # (state, action) -> np.ndarray  [neural]
    named_opponents: dict[str, Callable]   # name -> (player_id, rng) -> Agent
    default_output: str
```

`afterstate_obs` can be added now at low cost (it's just the flattened board for
Dao and TTT). This makes the spec ready for the neural trainer without changing
anything the tabular trainer uses.

For Dao and TTT, `afterstate_obs` returns the int-encoded flat board (length 16
or 9). For games where only neural training makes sense, `afterstate_idx` /
`n_states` / `terminal_w_init` can be left as None.

### 2. `games/ttt_hash.py`

Mirror of `dao_hash.py` for TicTacToe:
- Board encoding: `'.' -> 0, 'x' -> 1, 'o' -> 2`, flat array of length 9
- Hash: base-3 number (max 3^9 = 19 683 raw states)
- Symmetries: 8 (4 rotations × flip of a 3×3 grid)
- Enumerate reachable boards via DFS from initial state (~5 478 states)
- Build `_LOOKUP`, `N_CANONICAL_STATES`, `TERMINAL_W_INIT` with same caching
  pattern as `dao_hash.py`
- Public API: `canonical_index(flat_int_array)`, `board_to_int(board_2d)`

Afterstate index for TTT uses `state.child(action)._board` (already available
on `TicTacToeState`).

### 3. `agents/afterstate_agent.py`

Add optional `afterstate_idx_fn` parameter:

```python
class AfterstateAgent:
    def __init__(self, player_id, w_table, afterstate_idx_fn=_dao_afterstate_idx):
        ...

    @classmethod
    def load(cls, player_id, path, afterstate_idx_fn=_dao_afterstate_idx):
        ...
```

Default remains the Dao function → backend and `dao_test.py` unchanged.

### 4. `agents/afterstate_trainer.py`

- `train(game_spec: GameSpec, ...)` — remove all Dao-specific hardcoding
- `_evaluate` accepts `game_spec` and evaluates against all `named_opponents`
- Internal `_make_dao_spec()` and `_make_ttt_spec()` functions
- CLI: add `--game dao|ttt` (default: dao); `--output` default comes from spec

Dao spec named_opponents: `{"random": ..., "naive": ...}`
TTT spec named_opponents: `{"random": ...}` (no NaiveAgent for TTT)

---

## Phase 2 — PettingZoo wrappers (implement when needed)

Create `games/dao_pz.py` and `games/ttt_pz.py`: AEC wrappers over the existing
game objects. Required interface:

- `env.reset(seed, options)` → `{agent: obs}`
- `env.step(action)`
- `env.last()` → `(obs, reward, terminated, truncated, info)`
- `env.agent_selection`, `env.agents`, `env.possible_agents`
- `env.observation_spaces[agent]` — `gymnasium.spaces.Box`
- `env.action_spaces[agent]` — `gymnasium.spaces.Discrete(n)`
- `info["action_mask"]` — binary array of legal actions (critical for board
  games; most PZ-compatible algorithms support it)

These wrappers have no dependency on `GameSpec` or the afterstate trainer — they
are a parallel track over the same underlying game logic.

---

## Phase 3 — Neural network afterstate trainer (implement when needed)

### Why not DQN?

The afterstate approach has a structural advantage: two different positions that
lead to the same afterstate automatically receive the same value estimate.
DQN (Q(s, a)) must learn this equivalence from data.

The afterstate approach becomes **less efficient than DQN** when both:
- Action space is large (many afterstates to evaluate per move)
- State transitions are expensive (e.g. physics, complex rules)

For cheap-transition games (board games, card placement), afterstate is
generally preferable.

### Architecture

`agents/nn_afterstate_trainer.py` — parallel to the tabular trainer, shares
`GameSpec` but uses `afterstate_obs` instead of `afterstate_idx`:

- Value network: `f(afterstate_obs) -> scalar` (player-0 perspective)
- Move selection: `argmax_a f(game_spec.afterstate_obs(state, a))` over legal actions
- Updates: same MC / TD logic but with gradient descent on network params instead
  of `w_table[idx] += alpha * delta`
- Batch updates per episode (or replay buffer for stability)

The `GameSpec.afterstate_obs` field is the only addition needed to support this —
it should be included in Phase 1 at no extra cost.

### When to prefer neural over tabular

| Criterion | Tabular | Neural |
|---|---|---|
| State space | Must be enumerable (~<10M) | Unbounded |
| Training speed | Fast (array indexing) | Slower (GPU helps) |
| Generalisation | None (exact lookup) | Yes (feature sharing) |
| Symmetry exploit | Via canonical index | Via architecture / augmentation |

---

## Open questions

- Should there be a `GameRegistry` singleton mapping game names to `GameSpec`
  instances, rather than building specs inside `afterstate_trainer.py`? Useful
  once there are 3+ games.
- For the neural trainer: shared or separate networks per player? (Shared is
  natural here since W is already from player-0's perspective.)
- PZ wrappers: support `render_mode="human"` for the frontend, or keep rendering
  in the existing backend?
