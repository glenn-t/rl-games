# Glenn Thomas
# 2020-09-21

"""Dao state hasher"""

# Two main approaches
# 1) Create a class, when initialised creates a dictionary that is accessible
# 2) Essentially create an object when imported

import numpy as np
import games.dao as dao
import pdb

PIECES_PER_PLAYER = 1  # TODO CHANGE TO 4


class DaoStateHasher():

    def __init__(self):
        """Creates a DaoState Hasher"""

        # Create a board
        game = dao.DaoGame(20)
        dao_state = game.new_initial_state()
        board = dao_state._board
        board[:] = dao._PLAYER_TOKENS[None]
        board = board.flatten()

        remaining_pieces = {}
        for key, val in dao._PLAYER_TOKENS.items():
            if key is not None:
                remaining_pieces[val] = PIECES_PER_PLAYER

        self.all_states = _place_next_piece(board, remaining_pieces)

# Helper functions


def _hash_dao_state(state_array):
    """Takes a Dao state and converts it to a unique number

    Params:
    state_array: a flattened numpy array (of integers) representing the Dao
      state (must be 0, 1, 2)

    Details: The hash is done by treating the array as a 16 digit base 3 number,
    then translting it base 10.
    """

    # assert np.all(np.isin(state_array, [0, 1, 2])
    column_values = np.power(3, np.arange(16, 0, -1))
    result = np.sum(state_array * column_values)
    return(result)


def _place_next_piece(board, remaining_pieces):
    """Recursive function that gets all Dao states"""
    state_list = []
    total_pieces_remaining = np.sum(list(remaining_pieces.values()))

    # Identify the last piece of this colour
    last_piece = np.where(board != dao._PLAYER_TOKENS[None])
    if np.any(last_piece):
        last_piece = np.max(last_piece)
    else:
        last_piece = 0

    for piece, n_remaining in remaining_pieces.items():
        if n_remaining > 0:
            remaining_pieces_copy = remaining_pieces.copy()
            remaining_pieces_copy[piece] += -1

            # place a piece in all possible positions after the last piece
            for i in range(last_piece, board.shape[0]):
                if board[i] == dao._PLAYER_TOKENS[None]:
                    board_copy = board.copy()
                    board_copy[i] = piece

                    if total_pieces_remaining > 1:
                        # There is still another piece to place
                        new_states = _place_next_piece(board_copy, remaining_pieces_copy)
                        state_list = state_list + new_states
                    else:
                        # No more pieces to place
                        state_list = state_list + [board_copy]

    return(state_list)


def find_cannonical_state(state_array):
    """Finds the cannonical state of a given state array.

    Params:
    state_array: flattened numpy array containing 0, 1 or 2.

    Return:
    The cannonical state (equivalent state)

    Details:
    Takes a state array, generates all equivalant states, and takes the one
    with the lowest hast.
    """

    # Set up tracking
    cannonical_state = state_array
    cannonical_hash = _hash_dao_state(state_array)

    # Convert to matrix
    state_matrix = np.reshape(state_array, (4, 4))

    # Try regular and flipped board
    for flip in [False, True]:
        if flip:
            state_matrix = np.flip(state_matrix)

        # For each, get all four rotations
        for i in range(4):
            state_matrix = np.rot90(state_matrix)
            state_array = state_matrix.flatten()
            state_hash = _hash_dao_state(state_array)
            if state_hash < cannonical_hash:
                # Cannonical state is one with lowest hash
                cannonical_state = state_array
                cannonical_hash = state_hash
    return(cannonical_state)


state_hasher = DaoStateHasher()
len(state_hasher.all_states)
