"""Dao state hasher and canonical state dictionary for afterstate value learning.

Canonical state: the minimum-hash representative over all equivalent states
produced by the 8 board symmetries (4 rotations × flip). Player-swap is
intentionally excluded: W is anchored to player 0's perspective (+1 = player 0
wins, -1 = player 1 wins), so player 0's strong position and player 1's
mirror-image strong position must remain distinct entries.

Usage:
    from games.dao_hash import canonical_index, N_CANONICAL_STATES, TERMINAL_W_INIT

    idx = canonical_index(state._board.flatten())  # int in [0, N_CANONICAL_STATES)
    w_table = TERMINAL_W_INIT.copy()               # pre-initialised W-table
"""

import pickle as _pickle
from itertools import combinations
from pathlib import Path as _Path

import numpy as np

from games.dao import DaoState as _DaoState


class _BoardProxy:
    """Minimal duck-typed state for DaoState.check_victory()."""
    __slots__ = ("_board",)
    def __init__(self, board_4x4: np.ndarray):
        self._board = board_4x4

PIECES_PER_PLAYER = 4
_BOARD_SIZE = 16  # 4×4 flattened
_BASE = 3         # tokens are 0, 1, 2


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

_HASH_POWERS = np.int64(_BASE) ** np.arange(_BOARD_SIZE - 1, -1, -1, dtype=np.int64)


def _hash(state_array: np.ndarray) -> int:
    """Treat the flattened board as a base-3 number."""
    return int(np.dot(state_array, _HASH_POWERS))


# ---------------------------------------------------------------------------
# Symmetry transformations
# ---------------------------------------------------------------------------

def _all_symmetries(state_array: np.ndarray):
    """Yield all 16 equivalent boards (8 spatial × player swap)."""
    mat = state_array.reshape(4, 4)
    for flipped in (mat, np.fliplr(mat)):
        rotated = flipped
        for _ in range(4):
            rotated = np.rot90(rotated)
            flat = rotated.flatten()
            yield flat
            # # Player swap: 1 <-> 2, 0 stays 0
            # swapped = flat.copy()
            # swapped[flat == 1] = 2
            # swapped[flat == 2] = 1
            # yield swapped


def canonical_state(state_array: np.ndarray) -> np.ndarray:
    """Return the canonical (minimum-hash) representative of a board state."""
    best = min(_all_symmetries(state_array), key=_hash)
    return best



def _winner_of(flat: np.ndarray) -> int | None:
    """Return player_id (0 or 1) of the winner, or None. Delegates to DaoState.check_victory."""
    return _DaoState.check_victory(_BoardProxy(flat.reshape(4, 4)))


# ---------------------------------------------------------------------------
# Enumerate all reachable boards
# ---------------------------------------------------------------------------

def _enumerate_boards() -> list:
    """Return all boards reachable by placing PIECES_PER_PLAYER tokens per player."""
    positions = range(_BOARD_SIZE)
    results = []
    for p1_pos in combinations(positions, PIECES_PER_PLAYER):
        remaining = [p for p in positions if p not in p1_pos]
        for p2_pos in combinations(remaining, PIECES_PER_PLAYER):
            b = np.zeros(_BOARD_SIZE, dtype=np.int8)
            b[list(p1_pos)] = 1
            b[list(p2_pos)] = 2
            results.append(b)
    return results


# ---------------------------------------------------------------------------
# Build canonical state lookup table + terminal W initialisation
# ---------------------------------------------------------------------------

def _build_lookup() -> tuple[dict, int, np.ndarray]:
    """Build lookup and terminal W-table over all reachable boards.

    Returns:
        lookup:          {raw_hash: canonical_index}
        n_canonical:     number of distinct canonical states
        terminal_w_init: float32 array of shape (n_canonical,) with +1.0 for
                         player-0 (token=1) wins and -1.0 for player-1 (token=2) wins.
    """
    all_boards = _enumerate_boards()

    canonical_hash_to_index: dict[int, int] = {}
    canonical_boards: dict[int, np.ndarray] = {}  # idx -> one canonical board
    lookup: dict[int, int] = {}

    for board in all_boards:
        raw_h = _hash(board)
        if raw_h in lookup:
            continue
        canon = canonical_state(board)
        canon_h = _hash(canon)
        if canon_h not in canonical_hash_to_index:
            idx = len(canonical_hash_to_index)
            canonical_hash_to_index[canon_h] = idx
            canonical_boards[idx] = canon
        idx = canonical_hash_to_index[canon_h]
        for sym in _all_symmetries(board):
            h = _hash(sym)
            if h not in lookup:
                lookup[h] = idx

    n = len(canonical_hash_to_index)

    terminal_w_init = np.zeros(n, dtype=np.float32)
    for idx, canon in canonical_boards.items():
        winner = _winner_of(canon)
        if winner == 0:
            terminal_w_init[idx] = 1.0   # player 0 wins
        elif winner == 1:
            terminal_w_init[idx] = -1.0  # player 1 wins

    return lookup, n, terminal_w_init


# ---------------------------------------------------------------------------
# Cache — invalidates automatically when this source file is modified
# ---------------------------------------------------------------------------

_CACHE_PATH = _Path(__file__).parent / "dao_hash_cache.pkl"
_SOURCE_PATH = _Path(__file__)


def _load_or_build():
    if _CACHE_PATH.exists() and _CACHE_PATH.stat().st_mtime >= _SOURCE_PATH.stat().st_mtime:
        try:
            with open(_CACHE_PATH, "rb") as f:
                cached = _pickle.load(f)
            if isinstance(cached, tuple) and len(cached) == 3:
                return cached
        except Exception:
            pass
        _CACHE_PATH.unlink(missing_ok=True)

    result = _build_lookup()
    with open(_CACHE_PATH, "wb") as f:
        _pickle.dump(result, f)
    return result


_LOOKUP, N_CANONICAL_STATES, TERMINAL_W_INIT = _load_or_build()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def canonical_index(state_array: np.ndarray) -> int:
    """Return the canonical index for a flattened board state.

    Args:
        state_array: 1-D int array of length 16 with values in {0, 1, 2}.

    Returns:
        Integer in [0, N_CANONICAL_STATES).
    """
    return _LOOKUP[_hash(state_array)]
