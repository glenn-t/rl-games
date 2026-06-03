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
# Plain-Python tuples for hot paths — avoids numpy array creation per step
_DIRECTION_TUPLES = {name: (int(v[0]), int(v[1])) for name, v in _DIRECTION_COORDS.items()}
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

    def __init__(self, max_game_length=200, track_history=True):
        self._max_game_length = max_game_length
        self._track_history = track_history

    def new_initial_state(self):
        return DaoState(self, max_game_length=self._max_game_length,
                        track_history=self._track_history)

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

    def __init__(self, game, max_game_length, track_history=True):
        self._game = game
        self._max_game_length = max_game_length
        self._cur_player = 0
        self._winner = None
        self._is_terminal = False
        self._move_count = 0
        self._history = [] if track_history else None
        self._board = _initial_board(_NUM_ROWS, _PLAYER_TOKENS)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_state(self, cur_player, winner, is_terminal, history, board):
        self._cur_player = cur_player
        self._winner = winner
        self._is_terminal = is_terminal
        self._history = history
        self._move_count = len(history)
        self._board = board

    def clone(self):
        cloned = object.__new__(DaoState)
        cloned._game = self._game
        cloned._max_game_length = self._max_game_length
        cloned._cur_player = self._cur_player
        cloned._winner = self._winner
        cloned._is_terminal = self._is_terminal
        cloned._move_count = self._move_count
        cloned._history = None  # clones don't need history
        cloned._board = self._board.copy()
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
        """Returns the player_id of the winning player, or None.

        Inlines all win conditions with plain Python ints (via tolist()) to
        avoid numpy scalar boxing overhead on the hot path.
        """
        ((a, b, c, d),
         (e, f, g, h),
         (i, j, k, l),
         (m, n, o, p)) = self._board.tolist()

        for t in (1, 2):
            v = 3 - t  # opponent token
            # Rows
            if a==t and b==t and c==t and d==t: return t - 1
            if e==t and f==t and g==t and h==t: return t - 1
            if i==t and j==t and k==t and l==t: return t - 1
            if m==t and n==t and o==t and p==t: return t - 1
            # Columns
            if a==t and e==t and i==t and m==t: return t - 1
            if b==t and f==t and j==t and n==t: return t - 1
            if c==t and g==t and k==t and o==t: return t - 1
            if d==t and h==t and l==t and p==t: return t - 1
            # 2×2 squares
            if a==t and b==t and e==t and f==t: return t - 1
            if b==t and c==t and f==t and g==t: return t - 1
            if c==t and d==t and g==t and h==t: return t - 1
            if e==t and f==t and i==t and j==t: return t - 1
            if f==t and g==t and j==t and k==t: return t - 1
            if g==t and h==t and k==t and l==t: return t - 1
            if i==t and j==t and m==t and n==t: return t - 1
            if j==t and k==t and n==t and o==t: return t - 1
            if k==t and l==t and o==t and p==t: return t - 1
            # All four corners
            if a==t and d==t and m==t and p==t: return t - 1
            # Corner blocked: player t trapped, all 3 adjacent cells = opponent
            # (0,0) adjacent: (0,1),(1,0),(1,1)
            if a==t and b==v and e==v and f==v: return t - 1
            # (0,3) adjacent: (0,2),(1,3),(1,2)
            if d==t and c==v and h==v and g==v: return t - 1
            # (3,0) adjacent: (2,0),(3,1),(2,1)
            if m==t and i==v and n==v and j==v: return t - 1
            # (3,3) adjacent: (2,3),(3,2),(2,2)
            if p==t and l==v and o==v and k==v: return t - 1
        return None

    def line_exists(self):
        """Returns the token value of any player with a full row or column, else None."""
        b = self._board
        for t in (1, 2):
            for i in range(4):
                if b[i,0]==t and b[i,1]==t and b[i,2]==t and b[i,3]==t:
                    return t
            for j in range(4):
                if b[0,j]==t and b[1,j]==t and b[2,j]==t and b[3,j]==t:
                    return t
        return None

    def square_exists(self):
        """Returns the token value of any player with a 2x2 square, else None."""
        b = self._board
        for t in (1, 2):
            for i in range(3):
                for j in range(3):
                    if b[i,j]==t and b[i,j+1]==t and b[i+1,j]==t and b[i+1,j+1]==t:
                        return t
        return None

    def corner_exists(self):
        """Returns the token value of any player occupying all four corners, else None."""
        b = self._board
        for t in (1, 2):
            if b[0,0]==t and b[0,3]==t and b[3,0]==t and b[3,3]==t:
                return t
        return None

    def corner_blocked(self):
        """Returns the token value of any player trapped in a corner by the opponent, else None."""
        b = self._board
        for t in (1, 2):
            opp = 3 - t
            if b[0,0]==t and b[0,1]==opp and b[1,0]==opp and b[1,1]==opp:
                return t
            if b[0,3]==t and b[0,2]==opp and b[1,3]==opp and b[1,2]==opp:
                return t
            if b[3,0]==t and b[2,0]==opp and b[3,1]==opp and b[2,1]==opp:
                return t
            if b[3,3]==t and b[2,3]==opp and b[3,2]==opp and b[2,2]==opp:
                return t
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

        token = _PLAYER_TOKENS[self._cur_player]
        b = self._board
        actions = []
        for i in range(_NUM_ROWS):
            for j in range(_NUM_COLS):
                if b[i, j] == token:
                    for direction, (dr, dc) in _DIRECTION_TUPLES.items():
                        ni, nj = i + dr, j + dc
                        if 0 <= ni < _NUM_ROWS and 0 <= nj < _NUM_COLS and b[ni, nj] == 0:
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
        row, col, direction = _ACTIONID_TO_ACTION[action]
        dr, dc = _DIRECTION_TUPLES[direction]
        b = self._board

        b[row, col] = 0  # lift piece
        r, c = row, col
        while True:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= _NUM_ROWS or nc >= _NUM_COLS or b[nr, nc] != 0:
                break
            r, c = nr, nc
        b[r, c] = _PLAYER_TOKENS[self._cur_player]

        self._move_count += 1
        if self._history is not None:
            self._history.append(action)

        winner = self.check_victory()
        if winner is not None:
            self._is_terminal = True
            self._winner = winner
        elif self._move_count >= self._max_game_length:
            self._is_terminal = True
        else:
            self._cur_player = 1 - self._cur_player

    def child(self, action):
        """Returns a new state with the given action applied, leaving this state unchanged."""
        cloned = self.clone()
        cloned.apply_action(action)
        return cloned

    def board_after_action(self, action, perspective_player: int = 0) -> np.ndarray:
        """Returns a board copy with action applied — cheaper than child() for value lookup.

        Skips history copy and terminal check. Only use when you need the
        resulting board array, not a full game state.

        If perspective_player==1, tokens are swapped (1↔2) so the board is
        always from player 0's point of view. Pass self._cur_player (or the
        agent's player_id) to get a player-neutral representation.
        """
        row, col, direction = _ACTIONID_TO_ACTION[action]
        dr, dc = _DIRECTION_TUPLES[direction]

        board = self._board.copy()
        board[row, col] = 0  # lift piece
        r, c = row, col
        while True:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= _NUM_ROWS or nc >= _NUM_COLS or board[nr, nc] != 0:
                break
            r, c = nr, nc
        board[r, c] = _PLAYER_TOKENS[self._cur_player]

        if perspective_player == 1:
            swapped = board.copy()
            swapped[board == 1] = 2
            swapped[board == 2] = 1
            board = swapped

        return board

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
        self._move_count -= 1
        if self._history is not None:
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
                return [1.0, -10.0]
            elif self._winner == 1:
                return [-10.0, 1.0]
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