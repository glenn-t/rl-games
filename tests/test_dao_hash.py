"""Tests for games/dao_hash.py"""

import numpy as np
import pytest

from games.dao_hash import (
    _hash,
    _all_symmetries,
    canonical_state,
    canonical_index,
    N_CANONICAL_STATES,
    _enumerate_boards,
    PIECES_PER_PLAYER,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_board():
    """A valid board with 4 tokens per player."""
    # Player 1 (1) in top row, player 2 (2) in bottom row
    b = np.zeros(16, dtype=np.int8)
    b[0:4] = 1
    b[12:16] = 2
    return b


@pytest.fixture
def initial_board():
    """Starting Dao board (diagonal setup)."""
    from games.dao import DaoGame
    game = DaoGame()
    state = game.new_initial_state()
    return state._board.flatten().astype(np.int8)


# ---------------------------------------------------------------------------
# _hash
# ---------------------------------------------------------------------------

def test_hash_blank_board():
    assert _hash(np.zeros(16, dtype=np.int8)) == 0


def test_hash_single_token():
    b = np.zeros(16, dtype=np.int8)
    b[0] = 1
    # Token at position 0 (most significant): 1 * 3^15
    assert _hash(b) == 3 ** 15


def test_hash_distinct_boards(simple_board):
    b2 = simple_board.copy()
    b2[0], b2[15] = 2, 1
    assert _hash(simple_board) != _hash(b2)


def test_hash_deterministic(simple_board):
    assert _hash(simple_board) == _hash(simple_board.copy())


# ---------------------------------------------------------------------------
# _all_symmetries
# ---------------------------------------------------------------------------

def test_all_symmetries_count(simple_board):
    syms = list(_all_symmetries(simple_board))
    assert len(syms) == 16


def test_all_symmetries_player_swap_present(simple_board):
    syms = list(_all_symmetries(simple_board))
    swapped = simple_board.copy()
    swapped[simple_board == 1] = 2
    swapped[simple_board == 2] = 1
    assert any(np.array_equal(s, swapped) for s in syms)


def test_all_symmetries_only_valid_tokens(simple_board):
    for sym in _all_symmetries(simple_board):
        assert set(sym.tolist()).issubset({0, 1, 2})


def test_all_symmetries_preserves_token_counts(simple_board):
    for sym in _all_symmetries(simple_board):
        # Player swap may exchange counts, but total non-empty stays the same
        assert np.sum(sym != 0) == np.sum(simple_board != 0)


# ---------------------------------------------------------------------------
# canonical_state
# ---------------------------------------------------------------------------

def test_canonical_state_idempotent(simple_board):
    c = canonical_state(simple_board)
    assert np.array_equal(canonical_state(c), c)


def test_canonical_state_same_for_all_symmetries(simple_board):
    canon = canonical_state(simple_board)
    for sym in _all_symmetries(simple_board):
        assert np.array_equal(canonical_state(sym), canon)


def test_canonical_state_player_swap_equivalent():
    """Two boards that are player-swaps of each other share a canonical state."""
    b = np.zeros(16, dtype=np.int8)
    b[0:4] = 1
    b[12:16] = 2
    swapped = b.copy()
    swapped[b == 1] = 2
    swapped[b == 2] = 1
    assert np.array_equal(canonical_state(b), canonical_state(swapped))


def test_canonical_state_rotation_equivalent(simple_board):
    rotated = np.rot90(simple_board.reshape(4, 4)).flatten()
    assert np.array_equal(canonical_state(simple_board), canonical_state(rotated))


# ---------------------------------------------------------------------------
# _enumerate_boards
# ---------------------------------------------------------------------------

def test_enumerate_boards_token_counts():
    for board in _enumerate_boards():
        assert np.sum(board == 1) == PIECES_PER_PLAYER
        assert np.sum(board == 2) == PIECES_PER_PLAYER


def test_enumerate_boards_no_duplicates():
    boards = _enumerate_boards()
    hashes = [_hash(b) for b in boards]
    assert len(hashes) == len(set(hashes))


def test_enumerate_boards_count():
    """C(16,4) * C(12,4) = 1820 * 495 = 900900 distinct placements."""
    boards = _enumerate_boards()
    assert len(boards) == 900900


# ---------------------------------------------------------------------------
# canonical_index / N_CANONICAL_STATES
# ---------------------------------------------------------------------------

def test_canonical_index_in_range(initial_board):
    idx = canonical_index(initial_board)
    assert 0 <= idx < N_CANONICAL_STATES


def test_canonical_index_consistent(initial_board):
    assert canonical_index(initial_board) == canonical_index(initial_board.copy())


def test_canonical_index_same_for_symmetries(simple_board):
    idx = canonical_index(simple_board)
    for sym in _all_symmetries(simple_board):
        assert canonical_index(sym) == idx


def test_canonical_index_player_swap():
    b = np.zeros(16, dtype=np.int8)
    b[0:4] = 1
    b[12:16] = 2
    swapped = b.copy()
    swapped[b == 1] = 2
    swapped[b == 2] = 1
    assert canonical_index(b) == canonical_index(swapped)


def test_n_canonical_states_positive():
    assert N_CANONICAL_STATES > 0


def test_n_canonical_states_less_than_raw():
    """Canonical count must be strictly smaller than raw board count."""
    assert N_CANONICAL_STATES < 900900


def test_canonical_index_initial_board(initial_board):
    """Initial board should be a valid, registered state."""
    idx = canonical_index(initial_board)
    assert isinstance(idx, int)
