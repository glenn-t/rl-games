"""Dao state hasher and canonical state dictionary for Q-learning.

Canonical state: the minimum-hash representative over all equivalent states
produced by the 8 board symmetries (4 rotations × flip) and player swap
(exchanging token values 1 and 2). This collapses 16 raw states into one,
reducing the Q-table size by up to 16×.

Usage:
    from games.dao_hash import canonical_index, N_CANONICAL_STATES

    idx = canonical_index(state._board.flatten())  # int in [0, N_CANONICAL_STATES)
"""

from itertools import combinations

import numpy as np
import games.dao as dao

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
            # Player swap: 1 <-> 2, 0 stays 0
            swapped = flat.copy()
            swapped[flat == 1] = 2
            swapped[flat == 2] = 1
            yield swapped


def canonical_state(state_array: np.ndarray) -> np.ndarray:
    """Return the canonical (minimum-hash) representative of a board state."""
    best = min(_all_symmetries(state_array), key=_hash)
    return best


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
# Build canonical state lookup table
# ---------------------------------------------------------------------------

def _build_lookup() -> tuple[dict, int]:
    """Build {raw_hash: canonical_index} over all reachable boards.

    Returns (lookup_dict, n_canonical_states).
    """
    all_boards = _enumerate_boards()

    # Map each canonical hash to a stable index
    canonical_hash_to_index: dict[int, int] = {}
    lookup: dict[int, int] = {}

    for board in all_boards:
        raw_h = _hash(board)
        if raw_h in lookup:
            continue
        canon = canonical_state(board)
        canon_h = _hash(canon)
        if canon_h not in canonical_hash_to_index:
            canonical_hash_to_index[canon_h] = len(canonical_hash_to_index)
        idx = canonical_hash_to_index[canon_h]
        # Register every symmetry of this board so any equivalent state hits the cache
        for sym in _all_symmetries(board):
            h = _hash(sym)
            if h not in lookup:
                lookup[h] = idx

    return lookup, len(canonical_hash_to_index)


import pickle as _pickle
from pathlib import Path as _Path

_CACHE_PATH = _Path(__file__).parent / "dao_hash_cache.pkl"
if _CACHE_PATH.exists():
    with open(_CACHE_PATH, "rb") as _f:
        _LOOKUP, N_CANONICAL_STATES = _pickle.load(_f)
else:
    _LOOKUP, N_CANONICAL_STATES = _build_lookup()
    with open(_CACHE_PATH, "wb") as _f:
        _pickle.dump((_LOOKUP, N_CANONICAL_STATES), _f)


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
    h = _hash(state_array)
    return _LOOKUP[h]
