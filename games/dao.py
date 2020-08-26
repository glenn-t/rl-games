"""Dao board game"""

import pickle
import numpy as np
import pyspiel

_NUM_PLAYERS = 2
_NUM_ROWS = 4
_NUM_COLS = _NUM_ROWS  # Must be square
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "UP LEFT", "UP RIGHT", "DOWN LEFT", "DOWN RIGHT"]
_PLAYER_TOKENS = {
    0: "x",
    1: "o",
    None: " "
}
_DIRECTION_COORDS = {
    "UP": np.array([-1, 0]),
    "DOWN": np.array([1, 0]),
    "LEFT": np.array([0, -1]),
    "RIGHT": np.array([0, 1]),
    "UP LEFT": np.array([-1, -1]),
    "UP RIGHT": np.array([-1, 1]),
    "DOWN LEFT": np.array([1, -1]),
    "DOWN RIGHT": np.array([1, 1])
}


def _create_action_mapping(num_rows, num_cols, directions):
    """Creates two dictionaries, from mapping from action ID to action and another from action to action ID"""
    # TODO could improve efficiency by removing invalid actions here
    action_id_mapping = {}
    action_mapping = {}
    action_id = 0
    for i in range(num_rows):
        for j in range(num_cols):
            for direction in directions:
                action_id_mapping[action_id] = (i, j, direction)
                action_mapping[(i, j, direction)] = action_id
                action_id += 1
    return(action_id_mapping, action_mapping)


_ACTIONID_TO_ACTION, _ACTION_TO_ACTIONID = _create_action_mapping(_NUM_ROWS, _NUM_COLS, _DIRECTION_COORDS.keys())
_CORNER_COODS = [(0, 0), (0, _NUM_COLS - 1), (_NUM_ROWS - 1, 0), (_NUM_ROWS - 1, _NUM_COLS - 1)]
# Coordinates of cells adjacent to each corner
_CORNER_COORDS_ADJACENT = {
    # Top left
    (0, 0): [(0, 1), (1, 1), (1, 0)],
    # Top right
    (0, _NUM_COLS - 1): [(0, _NUM_COLS - 2), (1, _NUM_COLS - 1), (1, _NUM_COLS - 2)],
    # Bottom left
    (_NUM_ROWS - 1, 0): [(_NUM_ROWS - 2, 0), (_NUM_ROWS - 1, 1), (_NUM_ROWS - 2, 1)],
    # Bottom right
    (_NUM_ROWS - 1, _NUM_COLS - 1): [(_NUM_ROWS - 2, _NUM_COLS - 1), (_NUM_ROWS - 1, _NUM_COLS - 2), (_NUM_ROWS - 2, _NUM_COLS - 2)]
}


def _initial_board(num_rows, player_tokens):
    """Generate initial board"""
    board = np.full((num_rows, num_rows), player_tokens[None])
    for i in range(num_rows):
        # Fill in player 0 as leading diagonal
        board[(i, i)] = player_tokens[0]
        # Fill in player 1 as the off diagonal
        board[(i, num_rows - i - 1)] = player_tokens[1]
    return(board)


class DaoState(pyspiel.State):
    """Dao state

    This class implements all the pyspiel.State API functions. Please see spiel.h
    for more thorough documentation of each function.

    Note that this class does not inherit from pyspiel.State since pickle
    serialization is not possible due to what is required on the C++ side
    (backpointers to the C++ game object, which we can't get from here).
    """

    def __init__(self, game, max_game_length):
        self._game = game
        self._max_game_length = max_game_length
        self.set_state(
            cur_player=0,
            winner=None,
            is_terminal=False,
            history=[],
            board=_initial_board(_NUM_ROWS, _PLAYER_TOKENS))

    # Helper functions (not part of the OpenSpiel API).

    def set_state(self, cur_player, winner, is_terminal, history, board):
        self._cur_player = cur_player
        self._winner = winner
        self._is_terminal = is_terminal
        self._history = history
        self._board = board

    def get_player_token(self, player):
        return (_PLAYER_TOKENS[player])

    def check_victory(self):
        """Checks for victory and returns the player_id of the winning player or None if there is no winning player"""
        winner = self.line_exists()
        if winner is None:
            winner = self.square_exists()
            if winner is None:
                winner = self.corner_exists()
                if winner is None:
                    winner = self.corner_blocked()

        if winner is not None:
            # Return player_id of winning player
            winner = [key for key, value in _PLAYER_TOKENS.items() if value == winner][0]
            return winner
        else:
            return None

    def line_exists(self):
        """Checks if a line exists, returns "x" or "o" if so, and None otherwise."""
        # Check rows
        for i in range(_NUM_ROWS):
            for p in range(_NUM_PLAYERS):
                win = np.all(self._board[i, :] == _PLAYER_TOKENS[p])
                if win:
                    return _PLAYER_TOKENS[p]

        # Check columns
        for i in range(_NUM_COLS):
            for p in range(_NUM_PLAYERS):
                win = np.all(self._board[:, i] == _PLAYER_TOKENS[p])
                if win:
                    return _PLAYER_TOKENS[p]

        # No winner, return None
        return None

    def square_exists(self):
        """Checks if a square exists, returns "x" or "o" if so, and None otherwise."""
        for i in range(_NUM_ROWS - 1):
            for j in range(_NUM_COLS - 1):
                for p in range(_NUM_PLAYERS):
                    # Check square for each player
                    win = np.all(self._board[i:(i + 2), j:(j + 2)] == _PLAYER_TOKENS[p])
                    if win:
                        return _PLAYER_TOKENS[p]
        # No winner, return None
        return None

    def corner_exists(self):
        """Checks if a all corners are filled, returns "x" or "o" if so, and None otherwise."""
        corner_tokens = np.array([self._board[coord] for coord in _CORNER_COODS])
        for p in range(_NUM_PLAYERS):
            win = np.all(corner_tokens == _PLAYER_TOKENS[p])
            if win:
                return _PLAYER_TOKENS[p]

        # If no winner, return None
        return None

    def corner_blocked(self):
        """Checks if a players piece is cornered in a corner.
        Returns the symbol of the players piece that is cornered (i.e. the winner)
        """
        for p in range(_NUM_PLAYERS):
            for coord in _CORNER_COODS:
                if self._board[coord] == _PLAYER_TOKENS[p]:
                    # Players piece is in corner
                    # Check surrounds
                    surrounds = np.array([self._board[adj_coord] for adj_coord in _CORNER_COORDS_ADJACENT[coord]])
                    win = np.all(surrounds == _PLAYER_TOKENS[1 - p])
                    if win:
                        return _PLAYER_TOKENS[p]
        # No winner, return None
        return None

        # OpenSpiel (PySpiel) API functions are below. These need to be provided by
        # every game. Some not-often-used methods have been omitted.

    def current_player(self):
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def legal_actions(self, player=None):
        """Returns a list of legal actions, sorted in ascending order.

        Args:
          player: the player whose legal moves

        Returns:
          A list of legal moves, where each move is in [0, num_distinct_actions - 1]
          at non - terminal states, and empty list at terminal states.
        """

        if player is not None and player != self._cur_player:
            return []
        elif self.is_terminal():
            return []
        else:
            actions = []
            # 8 actions for each cell
            for i in range(_NUM_ROWS):
                for j in range(_NUM_COLS):
                    # Check if players piece is on board
                    if self._board[i, j] == _PLAYER_TOKENS[self._cur_player]:
                        current_cell = np.array([i, j])
                        for direction in _DIRECTIONS:
                            cell_to_move_to = current_cell + _DIRECTION_COORDS[direction]
                            # Check if direction is valid
                            if np.all(cell_to_move_to >= 0) & np.all(cell_to_move_to < _NUM_ROWS):
                                # Check if direction is free
                                if(self._board[cell_to_move_to[0], cell_to_move_to[1]] == _PLAYER_TOKENS[None]):
                                    actions.append(_ACTION_TO_ACTIONID[(i, j, direction)])
            return(actions)

    def legal_actions_mask(self, player=None):
        """Get a list of legal actions.

        Args:
          player: the player whose moves we want; defaults to the current player.

        Returns:
          A list of legal moves, where each move is in [0, num_distinct_actios - 1].
          Returns an empty list at terminal states, or if it is not the specified
          player's turn.
        """
        if player is not None and player != self._cur_player:
            return []
        elif self.is_terminal():
            return []
        else:
            action_mask = [0] * _NUM_CELLS * 8
            for action in self.legal_actions():
                action_mask[action] = 1
            return action_mask

    def apply_action(self, action):
        """Applies the specified action to the state."""

        # Check if action is legal
        assert action in self.legal_actions()
        # TODO could cache this result for effeciency (agent will call it)

        # Get action from action_id
        action = _ACTIONID_TO_ACTION[action]
        # Cell of piece to move
        current_cell = action[0:2]
        # Remove piece from the current cell
        self._board[current_cell[0], current_cell[1]] = _PLAYER_TOKENS[None]
        direction_id = action[2]
        direction_vector = _DIRECTION_COORDS[direction_id]
        # Keep moving in specified direction until piece cannot move
        blocked = False
        while (not blocked):
            next_cell = current_cell + direction_vector
            # Check if we have reached the edge of the board
            blocked = np.any(next_cell < 0) | np.any(next_cell >= _NUM_ROWS)
            if not blocked:
                # If not at edge of board, check if the cell is taken
                blocked = self._board[next_cell[0], next_cell[1]] != _PLAYER_TOKENS[None]
            if not blocked:
                # Move piece
                current_cell = next_cell

        # Move piece to current cell
        self._board[current_cell[0], current_cell[1]] = _PLAYER_TOKENS[self._cur_player]

        self._history.append(action)

        winner = self.check_victory()
        if winner is not None:
            self._is_terminal = True
            self._winner = winner
        elif len(self._history) == self._max_game_length:
            self._is_terminal = True
        else:
            # Switch player
            self._cur_player = 1 - self._cur_player

    def undo_action(self, action):
        # Optional function. Not used in many places.

        # Revert to previous player
        self._cur_player = 1 - self._cur_player

        # Get action from action_id
        action = _ACTIONID_TO_ACTION[action]

        # Find piece to move back
        current_cell = action[0:2]
        # Place piece back in the current cell
        self._board[current_cell[0], current_cell[1]] = _PLAYER_TOKENS[self._cur_player]
        direction_id = action[2]
        direction_vector = _DIRECTION_COORDS[direction_id]
        # Keep searching in specified direction until a piece is found
        blocked = False
        while (not blocked):
            current_cell = current_cell + direction_vector
            # Check if we have reached the edge of the board
            blocked = np.any(current_cell < 0) | np.any(current_cell > _NUM_ROWS)
            if not blocked:
                # If not at edge of board, check if the cell is taken
                blocked = self._board[current_cell[0], current_cell[1]] != _PLAYER_TOKENS[None]
        # Remove piece
        self._board[current_cell[0], current_cell[1]] = _PLAYER_TOKENS[None]

        self._history.pop()
        self._winner = None
        self._is_terminal = False

    def action_to_string(self, arg0, arg1=None):
        """Action -> string. Args either(player, action) or (action)."""
        player = self.current_player() if arg1 is None else arg0
        action = arg0 if arg1 is None else arg1
        action = _ACTIONID_TO_ACTION[action]
        return "Player: {}, Move ({}, {}) {}".format("x" if player == 0 else "o", action[0], action[1], action[2])

    def is_terminal(self):
        return self._is_terminal

    def returns(self):
        if self.is_terminal():
            if self._winner == 0:
                return [1.0, -1.0]
            elif self._winner == 1:
                return [-1.0, 1.0]
        return [0.0, 0.0]

    def rewards(self):
        return self.returns()

    def player_reward(self, player):
        return self.rewards()[player]

    def player_returns(self, player):
        return self.returns()[player]

    def is_chance_node(self):
        return False

    def is_simultaneous_node(self):
        return False

    def history(self):
        return self._history

    def history_str(self):
        return str(self._history)

    def information_state_string(self, player=None):
        del player  # Same information state for both players.
        return self.history_str()

    def information_state_tensor(self, player=None):
        raise NotImplementedError

    def observation_string(self, player=None):
        del player  # Same observation for both players.
        return str(self)

    def observation_tensor(self, player=None):
        del player  # Same observation for both players.
        observation = np.zeros((1 + _NUM_PLAYERS, _NUM_ROWS, _NUM_COLS))
        for row in range(_NUM_ROWS):
            for col in range(_NUM_COLS):
                index = ".ox".index(self._board[row, col])
                observation[index, row, col] = 1.0
        return list(observation.flatten())

    def child(self, action):
        cloned_state = self.clone()
        cloned_state.apply_action(action)
        return cloned_state

    def apply_actions(self, actions):
        raise NotImplementedError  # Only applies to simultaneous move games

    def num_distinct_actions(self):
        return _NUM_CELLS

    def num_players(self):
        return _NUM_PLAYERS

    def chance_outcomes(self):
        return []

    def get_game(self):
        return self._game

    def get_type(self):
        return self._game.get_type()

    def serialize(self):
        return pickle.dumps(self)

    def resample_from_infostate(self):
        return [self.clone()]

    def __str__(self):
        sep = "|"
        out = ""
        for row in self._board:
            out = out + sep + sep.join(row) + sep + "\n"
        return(out)

    def clone(self):
        cloned_state = DaoState(self._game, self._max_game_length)
        cloned_state.set_state(self._cur_player, self._winner, self._is_terminal,
                               self._history[:], np.array(self._board))
        return cloned_state


class DaoGame(object):
    """Dao

    This class implements all the pyspiel.Gae API functions. Please see spiel.h
    for more thorough documentation of each function.

    Note that this class does not inherit from pyspiel.Game since pickle
    serialization is not possible due to what is required on the C + + side
    (backpointers to the C + + game object, which we can't get from here).
    """

    def __init__(self, max_game_length):
        """
        max_length: maximum game length
        """
        self._max_game_length = max_game_length

    def new_initial_state(self):
        return DaoState(self, max_game_length=self._max_game_length)

    def num_distinct_actions(self):
        return _NUM_CELLS * 8  # 8 directions per cell

    def policy_tensor_shape(self):
        return (_NUM_ROWS, _NUM_COLS, 8)

    def clone(self):
        return DaoGame(max_game_length=self._max_game_length)

    def max_chance_outcomes(self):
        return 0

    def get_parameters(self):
        return {}

    def num_players(self):
        return _NUM_PLAYERS

    def min_utility(self):
        return -1.0

    def max_utility(self):
        return 1.0

    def get_type(self):
        return pyspiel.GameType(
            short_name="dao",
            long_name="Dao",
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
            information=pyspiel.GameType.Information.PERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.ZERO_SUM,
            reward_model=pyspiel.GameType.RewardModel.TERMINAL,
            max_num_players=_NUM_PLAYERS,
            min_num_players=_NUM_PLAYERS,
            provides_information_state_string=True,
            provides_information_state_tensor=False,
            provides_observation_string=True,
            provides_observation_tensor=True,
            parameter_specification={},
        )

    def utility_sum(self):
        return 0.0

    def observation_tensor_shape(self):
        return [1 + _NUM_PLAYERS, _NUM_ROWS, _NUM_COLS]

    def observation_tensor_layout(self):
        return pyspiel.TensorLayout.CHW

    def observation_tensor_size(self):
        return np.product(self.observation_tensor_shape())

    def deserialize_state(self, string):
        return pickle.loads(string)

    def max_game_length(self):
        return self._max_game_length

    def __str__(self):
        return "dao"
