"""Tug of War — minimal two-player test game.

A marker starts at position 0 on a number line [-10, 10].
  - Player 0 wins by moving the marker to +10.
  - Player 1 wins by moving the marker to -10.

Each turn the active player moves the marker one step left (-1) or right (+1),
so there are exactly 2 actions:
  action 0 → move left  (position - 1)
  action 1 → move right (position + 1)

Rewards:
  Terminal:     +1 / -1 on win / loss, 0 on draw (max-length timeout).
  Intermediate: optional fractional reward proportional to marker progress
                (enabled via reward_mode='intermediate' or 'both').

reward_mode options:
  'terminal'     — reward only at game end (default).
  'intermediate' — per-step reward = Δposition / WIN_POSITION for the mover,
                   no terminal reward.
  'both'         — per-step progress reward AND terminal win/loss reward.

This game is intentionally trivial so the optimal policy is obvious:
  Player 0 should always move right; player 1 should always move left.
It's useful for sanity-checking learning algorithms quickly.
"""

_WIN_POSITION = 10
_NUM_ACTIONS = 2          # 0 = left, 1 = right
_ACTION_DELTA = {0: -1, 1: +1}

TERMINAL = -1


class TugOfWarGame:
    """Factory for TugOfWarState instances."""

    def __init__(
        self,
        max_game_length: int = 100,
        reward_mode: str = "terminal",
    ):
        assert reward_mode in ("terminal", "intermediate", "both"), (
            f"reward_mode must be 'terminal', 'intermediate', or 'both', got {reward_mode!r}"
        )
        self._max_game_length = max_game_length
        self._reward_mode = reward_mode

    def new_initial_state(self):
        return TugOfWarState(self)

    def num_distinct_actions(self) -> int:
        return _NUM_ACTIONS

    def num_players(self) -> int:
        return 2

    def max_game_length(self) -> int:
        return self._max_game_length

    def __str__(self) -> str:
        return "tug_of_war"


class TugOfWarState:
    """Mutable state for a single Tug of War game."""

    def __init__(self, game: TugOfWarGame):
        self._game = game
        self._position = 0          # marker position in [-10, 10]
        self._cur_player = 0
        self._move_count = 0
        self._is_terminal = False
        self._winner = None         # 0, 1, or None (draw)
        self._returns = [0.0, 0.0]  # accumulated returns per player

    # ------------------------------------------------------------------
    # Core status
    # ------------------------------------------------------------------

    def current_player(self) -> int:
        return TERMINAL if self._is_terminal else self._cur_player

    def is_terminal(self) -> bool:
        return self._is_terminal

    def winner(self):
        """Winning player (0 or 1) or None on draw / game still in progress."""
        return self._winner

    def position(self) -> int:
        """Current marker position in [-WIN_POSITION, WIN_POSITION]."""
        return self._position

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def legal_actions(self) -> list[int]:
        if self._is_terminal:
            return []
        return [0, 1]

    def apply_action(self, action: int) -> None:
        assert action in (0, 1), f"Invalid action {action}"
        assert not self._is_terminal, "Game is already over"

        prev_pos = self._position
        self._position += _ACTION_DELTA[action]
        self._move_count += 1

        # Intermediate / 'both' step reward for the mover
        if self._game._reward_mode in ("intermediate", "both"):
            delta = self._position - prev_pos   # +1 or -1
            # Reward is positive for the player whose winning direction matches delta:
            #   player 0 wants +delta, player 1 wants -delta
            step_reward = delta / _WIN_POSITION
            self._returns[0] += step_reward
            self._returns[1] -= step_reward

        # Check terminal conditions
        if self._position >= _WIN_POSITION:
            self._is_terminal = True
            self._winner = 0
            if self._game._reward_mode in ("terminal", "both"):
                self._returns[0] += 1.0
                self._returns[1] -= 1.0
        elif self._position <= -_WIN_POSITION:
            self._is_terminal = True
            self._winner = 1
            if self._game._reward_mode in ("terminal", "both"):
                self._returns[1] += 1.0
                self._returns[0] -= 1.0
        elif self._move_count >= self._game._max_game_length:
            self._is_terminal = True
            # winner stays None → draw, no terminal reward
        else:
            self._cur_player = 1 - self._cur_player

    def child(self, action: int) -> "TugOfWarState":
        """Return a new state with action applied, leaving this state unchanged."""
        cloned = self.clone()
        cloned.apply_action(action)
        return cloned

    # ------------------------------------------------------------------
    # Returns / rewards
    # ------------------------------------------------------------------

    def returns(self) -> list[float]:
        """Accumulated returns for [player_0, player_1]."""
        return list(self._returns)

    def player_return(self, player_id: int) -> float:
        return self._returns[player_id]

    # ------------------------------------------------------------------
    # Clone / copy
    # ------------------------------------------------------------------

    def clone(self) -> "TugOfWarState":
        c = object.__new__(TugOfWarState)
        c._game = self._game
        c._position = self._position
        c._cur_player = self._cur_player
        c._move_count = self._move_count
        c._is_terminal = self._is_terminal
        c._winner = self._winner
        c._returns = list(self._returns)
        return c

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        w = _WIN_POSITION
        bar = ["·"] * (2 * w + 1)
        bar[self._position + w] = "●"
        bar[0] = "1"          # player 1 goal
        bar[-1] = "0"         # player 0 goal
        bar_str = "".join(bar)
        status = (
            f"winner=P{self._winner}" if self._winner is not None
            else ("draw" if self._is_terminal else f"P{self._cur_player} to move")
        )
        return f"[{bar_str}]  pos={self._position:+d}  {status}"
