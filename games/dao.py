"""Dao board game - standalone implementation (no pyspiel dependency)"""

import numpy as np

_NUM_PLAYERS = 2
_NUM_ROWS = 4
_NUM_COLS = _NUM_ROWS  # Must be square
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "UP LEFT", "UP RIGHT", "DOWN LEFT", "DOWN RIGHT"]
_PLAYER_TOKENS = {
    0: 1,
    1: 2,
    None: 0
}
_PLAYER_TOKENS_PRINT = {
    1: "x",
    2: "o",
    0: " "
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
    """Creates two dicts: action ID -> action and action -> action ID"""
    action_id_mapping = {}
    action_mapping = {}
    action_id = 0
    for i in range(num_rows):
        for j in range(num_cols):
            for direction in directions:
                action_id_mapping[action_id] = (i, j, direction)
                action_mapping[(i, j, direction)] = action_id
                action_id += 1
    return action_id_mapping, action_mapping


_ACTIONID_TO_ACTION, _ACTION_TO_ACTIONID = _create_action_mapping(_NUM_ROWS, _NUM_COLS, _DIRECTION_COORDS.keys())
_CORNER_COORDS = [(0, 0), (0, _NUM_COLS - 1), (_NUM_ROWS - 1, 0), (_NUM_ROWS - 1, _NUM_COLS - 1)]
_CORNER_COORDS_ADJACENT = {
    (0, 0): [(0, 1), (1, 1), (1, 0)],
    (0, _NUM_COLS - 1): [(0, _NUM_COLS - 2), (1, _NUM_COLS - 1), (1, _NUM_COLS - 2)],
    (_NUM_ROWS - 1, 0): [(_NUM_ROWS - 2, 0), (_NUM_ROWS - 1, 1), (_NUM_ROWS - 2, 1)],
    (_NUM_ROWS - 1, _NUM_COLS - 1): [(_NUM_ROWS - 2, _NUM_COLS - 1), (_NUM_ROWS - 1, _NUM_COLS - 2), (_NUM_ROWS - 2, _NUM_COLS - 2)]
}

# Terminal state constants (replaces pyspiel.PlayerId)
TERMINAL = -1
INVALID_PLAYER = -2


def _initial_board(num_rows, player_tokens):
    """Generate initial board"""
    board = np.full((num_rows, num_rows), player_tokens[None])
    for i in range(num_rows):
        board[(i, i)] = player_tokens[0]
        board[(i, num_rows - i - 1)] = player_tokens[1]
    return board


class DaoGame:
    """Factory for creating Dao game states."""

    def __init__(self, max_game_length=200):
        self._max_game_length = max_game_length

    def new_initial_state(self):
        return DaoState(self, max_game_length=self._max_game_length)

    def num_distinct_actions(self):
        return _NUM_CELLS * 8

    def num_players(self):
        return _NUM_PLAYERS

    def max_game_length(self):
        return self._max_game_length

    def __str__(self):
        return "dao"


class DaoState:
    """Dao game state.

    Tracks the board, current player, history, and terminal conditions.
    Designed to be the sole interface for playing and inspecting Dao games.
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

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_state(self, cur_player, winner, is_terminal, history, board):
        self._cur_player = cur_player
        self._winner = winner
        self._is_terminal = is_terminal
        self._history = history
        self._board = board

    def clone(self):
        cloned = DaoState(self._game, self._max_game_length)
        cloned.set_state(
            self._cur_player,
            self._winner,
            self._is_terminal,
            self._history[:],
            np.array(self._board))
        return cloned

    # ------------------------------------------------------------------
    # Player / terminal status
    # ------------------------------------------------------------------

    def current_player(self):
        """Returns TERMINAL if game is over, otherwise the current player id (0 or 1)."""
        return TERMINAL if self._is_terminal else self._cur_player

    def is_terminal(self):
        return self._is_terminal

    def winner(self):
        """Returns the winning player id (0 or 1), or None if game is not over / draw."""
        return self._winner

    # ------------------------------------------------------------------
    # Victory conditions
    # ------------------------------------------------------------------

    def check_victory(self):
        """Returns the player_id of the winning player, or None."""
        winner = self.line_exists()
        if winner is None:
            winner = self.square_exists()
        if winner is None:
            winner = self.corner_exists()
        if winner is None:
            winner = self.corner_blocked()

        if winner is not None:
            return [key for key, value in _PLAYER_TOKENS.items() if value == winner][0]
        return None

    def line_exists(self):
        """Returns the token value of any player with a full row or column, else None."""
        for i in range(_NUM_ROWS):
            for p in range(_NUM_PLAYERS):
                if np.all(self._board[i, :] == _PLAYER_TOKENS[p]):
                    return _PLAYER_TOKENS[p]
        for i in range(_NUM_COLS):
            for p in range(_NUM_PLAYERS):
                if np.all(self._board[:, i] == _PLAYER_TOKENS[p]):
                    return _PLAYER_TOKENS[p]
        return None

    def square_exists(self):
        """Returns the token value of any player with a 2x2 square, else None."""
        for i in range(_NUM_ROWS - 1):
            for j in range(_NUM_COLS - 1):
                for p in range(_NUM_PLAYERS):
                    if np.all(self._board[i:(i + 2), j:(j + 2)] == _PLAYER_TOKENS[p]):
                        return _PLAYER_TOKENS[p]
        return None

    def corner_exists(self):
        """Returns the token value of any player occupying all four corners, else None."""
        corner_tokens = np.array([self._board[coord] for coord in _CORNER_COORDS])
        for p in range(_NUM_PLAYERS):
            if np.all(corner_tokens == _PLAYER_TOKENS[p]):
                return _PLAYER_TOKENS[p]
        return None

    def corner_blocked(self):
        """Returns the token value of any player trapped in a corner by the opponent, else None."""
        for p in range(_NUM_PLAYERS):
            for coord in _CORNER_COORDS:
                if self._board[coord] == _PLAYER_TOKENS[p]:
                    surrounds = np.array([self._board[adj] for adj in _CORNER_COORDS_ADJACENT[coord]])
                    if np.all(surrounds == _PLAYER_TOKENS[1 - p]):
                        return _PLAYER_TOKENS[p]
        return None

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def legal_actions(self, player=None):
        """Returns a sorted list of legal action IDs for the current player."""
        if player is not None and player != self._cur_player:
            return []
        if self.is_terminal():
            return []

        actions = []
        for i in range(_NUM_ROWS):
            for j in range(_NUM_COLS):
                if self._board[i, j] == _PLAYER_TOKENS[self._cur_player]:
                    current_cell = np.array([i, j])
                    for direction in _DIRECTIONS:
                        cell_to_move_to = current_cell + _DIRECTION_COORDS[direction]
                        if np.all(cell_to_move_to >= 0) and np.all(cell_to_move_to < _NUM_ROWS):
                            if self._board[cell_to_move_to[0], cell_to_move_to[1]] == _PLAYER_TOKENS[None]:
                                actions.append(_ACTION_TO_ACTIONID[(i, j, direction)])
        return actions

    def legal_actions_mask(self, player=None):
        """Returns a binary list of length num_distinct_actions indicating legal moves."""
        if player is not None and player != self._cur_player:
            return []
        if self.is_terminal():
            return []

        mask = [0] * _NUM_CELLS * 8
        for action in self.legal_actions():
            mask[action] = 1
        return mask

    def apply_action(self, action):
        """Applies an action (by action ID) to the state, mutating it in place."""
        assert action in self.legal_actions(), f"Illegal action: {action}"

        row, col, direction = _ACTIONID_TO_ACTION[action]
        current_cell = np.array([row, col])
        direction_vector = _DIRECTION_COORDS[direction]

        # Lift piece
        self._board[row, col] = _PLAYER_TOKENS[None]

        # Slide until blocked
        blocked = False
        while not blocked:
            next_cell = current_cell + direction_vector
            blocked = np.any(next_cell < 0) or np.any(next_cell >= _NUM_ROWS)
            if not blocked:
                blocked = self._board[next_cell[0], next_cell[1]] != _PLAYER_TOKENS[None]
            if not blocked:
                current_cell = next_cell

        # Place piece
        self._board[current_cell[0], current_cell[1]] = _PLAYER_TOKENS[self._cur_player]
        self._history.append(action)

        winner = self.check_victory()
        if winner is not None:
            self._is_terminal = True
            self._winner = winner
        elif len(self._history) >= self._max_game_length:
            self._is_terminal = True
        else:
            self._cur_player = 1 - self._cur_player

    def child(self, action):
        """Returns a new state with the given action applied, leaving this state unchanged."""
        cloned = self.clone()
        cloned.apply_action(action)
        return cloned

    def undo_action(self, action):
        """Reverses the last action. Assumes action was the most recent move."""
        self._cur_player = 1 - self._cur_player

        row, col, direction = _ACTIONID_TO_ACTION[action]
        current_cell = np.array([row, col])
        direction_vector = _DIRECTION_COORDS[direction]

        # Place piece back at its origin
        self._board[row, col] = _PLAYER_TOKENS[self._cur_player]

        # Find where the piece slid to and remove it
        blocked = False
        while not blocked:
            current_cell = current_cell + direction_vector
            blocked = np.any(current_cell < 0) or np.any(current_cell >= _NUM_ROWS)
            if not blocked:
                blocked = self._board[current_cell[0], current_cell[1]] != _PLAYER_TOKENS[None]

        self._board[current_cell[0], current_cell[1]] = _PLAYER_TOKENS[None]
        self._history.pop()
        self._winner = None
        self._is_terminal = False

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def returns(self):
        """Returns [reward_player0, reward_player1]. Non-zero only at terminal."""
        if self.is_terminal():
            if self._winner == 0:
                return [1.0, -1.0]
            elif self._winner == 1:
                return [-1.0, 1.0]
        return [0.0, 0.0]

    def player_return(self, player):
        return self.returns()[player]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observation_tensor(self):
        """Returns a flat float array of shape (3 * 4 * 4,).

        Three binary planes: empty cells, player-0 pieces, player-1 pieces.
        """
        obs = np.zeros((1 + _NUM_PLAYERS, _NUM_ROWS, _NUM_COLS), dtype=np.float32)
        for row in range(_NUM_ROWS):
            for col in range(_NUM_COLS):
                token = self._board[row, col]
                if token == _PLAYER_TOKENS[None]:
                    obs[0, row, col] = 1.0
                elif token == _PLAYER_TOKENS[0]:
                    obs[1, row, col] = 1.0
                elif token == _PLAYER_TOKENS[1]:
                    obs[2, row, col] = 1.0
        return obs.flatten()

    def board_array(self):
        """Returns the raw 4x4 numpy board array (values 0, 1, 2)."""
        return np.array(self._board)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def history(self):
        return list(self._history)

    def action_to_string(self, action, player=None):
        player = self._cur_player if player is None else player
        row, col, direction = _ACTIONID_TO_ACTION[action]
        label = "x" if player == 0 else "o"
        return f"Player {label}: ({row}, {col}) {direction}"

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self):
        sep = "|"
        rows = []
        for i in range(self._board.shape[0]):
            row = sep + sep.join(_PLAYER_TOKENS_PRINT[self._board[i, j]]
                                 for j in range(self._board.shape[1])) + sep
            rows.append(row)
        return "\n".join(rows)