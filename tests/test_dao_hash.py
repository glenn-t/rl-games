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
    TERMINAL_W_INIT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_board():
    """A valid board with 4 tokens per player."""
    # Player 1 (token=1) in top row, player 2 (token=2) in bottom row
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


def _player_swap(board: np.ndarray) -> np.ndarray:
    swapped = board.copy()
    swapped[board == 1] = 2
    swapped[board == 2] = 1
    return swapped


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
# _all_symmetries — 8 spatial only (no player swap)
# ---------------------------------------------------------------------------

def test_all_symmetries_count(simple_board):
    syms = list(_all_symmetries(simple_board))
    assert len(syms) == 8


def test_all_symmetries_no_player_swap():
    """Player-swap is excluded from the symmetry group.
    Uses a board where player-swap cannot be reproduced by any spatial rotation.
    """
    # token=1 fills row 0 (wins), token=2 fills row 1 — rotating this never
    # produces the player-swapped version (token=2 row 0, token=1 row 1).
    b = np.zeros(16, dtype=np.int8)
    b[0:4] = 1
    b[4:8] = 2
    syms = list(_all_symmetries(b))
    swapped = _player_swap(b)
    assert not any(np.array_equal(s, swapped) for s in syms)


def test_all_symmetries_only_valid_tokens(simple_board):
    for sym in _all_symmetries(simple_board):
        assert set(sym.tolist()).issubset({0, 1, 2})


def test_all_symmetries_preserves_token_counts(simple_board):
    for sym in _all_symmetries(simple_board):
        assert np.sum(sym == 1) == np.sum(simple_board == 1)
        assert np.sum(sym == 2) == np.sum(simple_board == 2)


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


def test_canonical_state_player_swap_distinct():
    """Without player-swap symmetry, a board and its player-swap have different canonical states."""
    b = np.zeros(16, dtype=np.int8)
    b[0:4] = 1   # token=1 row 0
    b[4:8] = 2   # token=2 row 1 — player-swap is not a spatial rotation of this
    assert not np.array_equal(canonical_state(b), canonical_state(_player_swap(b)))


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


def test_canonical_index_player_swap_distinct():
    """Player-swapped boards must map to different canonical indices."""
    b = np.zeros(16, dtype=np.int8)
    b[0:4] = 1   # token=1 row 0
    b[4:8] = 2   # token=2 row 1 — player-swap is not a spatial rotation of this
    assert canonical_index(b) != canonical_index(_player_swap(b))


def test_n_canonical_states_positive():
    assert N_CANONICAL_STATES > 0


def test_n_canonical_states_less_than_raw():
    """Canonical count must be strictly smaller than raw board count."""
    assert N_CANONICAL_STATES < 900900


def test_n_canonical_states_expected():
    """Without player-swap, 8 spatial symmetries only — expect ~113k states."""
    assert N_CANONICAL_STATES == 113028


def test_canonical_index_initial_board(initial_board):
    """Initial board should be a valid, registered state."""
    idx = canonical_index(initial_board)
    assert isinstance(idx, int)


# ---------------------------------------------------------------------------
# TERMINAL_W_INIT
# ---------------------------------------------------------------------------

def test_terminal_w_init_shape():
    assert TERMINAL_W_INIT.shape == (N_CANONICAL_STATES,)


def test_terminal_w_init_values():
    """Only +1.0, -1.0, and 0.0 are valid values."""
    unique = set(np.unique(TERMINAL_W_INIT).tolist())
    assert unique.issubset({-1.0, 0.0, 1.0})


def test_terminal_w_init_player0_win():
    """A known player-0 terminal state should have W=+1.0.
    token=1 fills row 0, token=2 fills row 1 — no spatial rotation produces
    a token=2 win, so the canonical form unambiguously has token=1 winning.
    """
    b = np.zeros(16, dtype=np.int8)
    b[0:4] = 1
    b[4:8] = 2
    assert TERMINAL_W_INIT[canonical_index(b)] == 1.0


def test_terminal_w_init_player1_win():
    """A known player-1 terminal state should have W=-1.0.
    token=2 occupies all 4 corners — rotationally invariant, so the canonical
    form always has token=2 in the corners regardless of spatial symmetry.
    token=1 placed where it cannot satisfy any win condition.
    """
    b = np.zeros(16, dtype=np.int8)
    b[[0, 3, 12, 15]] = 2  # token=2 in all 4 corners — player 1 wins
    b[[1, 4, 7, 8]] = 1    # token=1 scattered — no win condition
    assert TERMINAL_W_INIT[canonical_index(b)] == -1.0


def test_terminal_w_init_non_terminal_is_zero(initial_board):
    """The initial board is not terminal and should have W=0."""
    assert TERMINAL_W_INIT[canonical_index(initial_board)] == 0.0
