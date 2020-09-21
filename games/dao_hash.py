# Glenn Thomas
# 2020-09-21

"""Dao state hasher"""

import numpy as np
import games.dao as dao

game = dao.DaoGame(20)
dao_state = game.new_initial_state()
board = dao_state._board
board[:] = 0
board = board.flatten()


def place_next_piece(board, remaining_x, remaining_o):
    if remaining_x > 0:
        pass


def generate_all_dao_states():
    all_states = []
    pass


def hash_dao_state():
    pass
