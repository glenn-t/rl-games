# Glenn Thomas
# 2020-09-21

"""Dao state hasher"""

import numpy as np
import games.dao as dao
import pdb

# Create a board
game = dao.DaoGame(20)
dao_state = game.new_initial_state()
board = dao_state._board
board[:] = dao._PLAYER_TOKENS[None]
board = board.flatten()

remaining_pieces = {}
for key, val in dao._PLAYER_TOKENS.items():
    if key is not None:
        remaining_pieces[val] = 4


def place_next_piece(board, remaining_pieces):
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
                        new_states = place_next_piece(board_copy, remaining_pieces_copy)
                        state_list = state_list + new_states
                    else:
                        # No more pieces to place
                        state_list = state_list + [board_copy]

    return(state_list)


out = place_next_piece(board, remaining_pieces)
len(out)


def hash_dao_state():
    pass
