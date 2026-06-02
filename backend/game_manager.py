"""Game session management for Dao games."""

import sys
import os
from typing import Dict, Optional
from datetime import datetime
import uuid

# Add parent directory to path to import game logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.dao import DaoGame, DaoState
from agents.naive import NaiveAgent, INVALID_ACTION
from agents.random import RandomAgent
from agents.afterstate_agent import AfterstateAgent

_W_TABLE_PATH = "trained_models/w_table.npy"
import numpy as np


class GameSession:
    """Represents a single game session."""
    
    def __init__(
        self,
        game_id: str,
        state: DaoState,
        mode: str,
        ai_agent: Optional[object] = None,
        ai_player: Optional[int] = None,
        ai_agent_player0: Optional[object] = None,
        ai_agent_player1: Optional[object] = None
    ):
        self.game_id = game_id
        self.state = state
        self.mode = mode
        self.ai_agent = ai_agent  # For human_vs_ai mode
        self.ai_player = ai_player  # For human_vs_ai mode
        self.ai_agent_player0 = ai_agent_player0  # For ai_vs_ai mode
        self.ai_agent_player1 = ai_agent_player1  # For ai_vs_ai mode
        self.created_at = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert game session to dictionary for API response."""
        # Convert numpy types to Python native types
        current_player = self.state.current_player()
        winner = self.state.winner()
        
        result = {
            "game_id": self.game_id,
            "board": self.state.board_array().tolist(),
            "current_player": int(current_player) if current_player is not None and current_player >= 0 else current_player,
            "is_terminal": bool(self.state.is_terminal()),
            "winner": int(winner) if winner is not None else None,
            "legal_actions": [int(a) for a in self.state.legal_actions()],
            "history": [int(h) for h in self.state.history()],
            "mode": self.mode,
        }
        
        if self.mode == "ai_vs_ai":
            result["ai_agent_player0"] = type(self.ai_agent_player0).__name__ if self.ai_agent_player0 else None
            result["ai_agent_player1"] = type(self.ai_agent_player1).__name__ if self.ai_agent_player1 else None
        else:
            result["ai_agent"] = type(self.ai_agent).__name__ if self.ai_agent else None
            result["ai_player"] = int(self.ai_player) if self.ai_player is not None else None
        
        return result


class GameManager:
    """Manages all active game sessions."""
    
    def __init__(self):
        self.games: Dict[str, GameSession] = {}
        self.dao_game = DaoGame(max_game_length=200)
        # Load afterstate agent for W-value analysis (always available)
        self.analysis_agent_p0 = None
        self.analysis_agent_p1 = None
        try:
            self.analysis_agent_p0 = AfterstateAgent.load(0, _W_TABLE_PATH)
            self.analysis_agent_p1 = AfterstateAgent.load(1, _W_TABLE_PATH)
        except Exception as e:
            print(f"Warning: Could not load afterstate agent for analysis: {e}")
    
    def create_game(
        self,
        mode: str,
        ai_agent_type: Optional[str] = None,
        ai_player: Optional[int] = 1,
        ai_agent_player0_type: Optional[str] = None,
        ai_agent_player1_type: Optional[str] = None
    ) -> GameSession:
        """Create a new game session."""
        game_id = str(uuid.uuid4())
        state = self.dao_game.new_initial_state()
        
        ai_agent = None
        ai_agent_player0 = None
        ai_agent_player1 = None
        
        if mode == "human_vs_ai" and ai_agent_type:
            # Create AI agent for human_vs_ai mode
            if ai_agent_type == "naive":
                ai_agent = NaiveAgent(
                    player_id=ai_player,
                    num_actions=self.dao_game.num_distinct_actions()
                )
            elif ai_agent_type == "random":
                ai_agent = RandomAgent(player_id=ai_player)
            elif ai_agent_type == "afterstate":
                ai_agent = AfterstateAgent.load(ai_player, _W_TABLE_PATH)
            else:
                raise ValueError(f"Unknown agent type: {ai_agent_type}")
        
        elif mode == "ai_vs_ai":
            # Create AI agents for both players
            if ai_agent_player0_type:
                if ai_agent_player0_type == "naive":
                    ai_agent_player0 = NaiveAgent(
                        player_id=0,
                        num_actions=self.dao_game.num_distinct_actions()
                    )
                elif ai_agent_player0_type == "random":
                    ai_agent_player0 = RandomAgent(player_id=0)
                elif ai_agent_player0_type == "afterstate":
                    ai_agent_player0 = AfterstateAgent.load(0, _W_TABLE_PATH)
                else:
                    raise ValueError(f"Unknown agent type: {ai_agent_player0_type}")
            
            if ai_agent_player1_type:
                if ai_agent_player1_type == "naive":
                    ai_agent_player1 = NaiveAgent(
                        player_id=1,
                        num_actions=self.dao_game.num_distinct_actions()
                    )
                elif ai_agent_player1_type == "random":
                    ai_agent_player1 = RandomAgent(player_id=1)
                elif ai_agent_player1_type == "afterstate":
                    ai_agent_player1 = AfterstateAgent.load(1, _W_TABLE_PATH)
                else:
                    raise ValueError(f"Unknown agent type: {ai_agent_player1_type}")
        
        session = GameSession(
            game_id=game_id,
            state=state,
            mode=mode,
            ai_agent=ai_agent,
            ai_player=ai_player,
            ai_agent_player0=ai_agent_player0,
            ai_agent_player1=ai_agent_player1
        )
        
        self.games[game_id] = session
        return session
    
    def get_game(self, game_id: str) -> Optional[GameSession]:
        """Get a game session by ID."""
        return self.games.get(game_id)
    
    def make_move(self, game_id: str, action: int, player: int) -> dict:
        """Make a move in a game."""
        session = self.get_game(game_id)
        if not session:
            raise ValueError(f"Game {game_id} not found")
        
        state = session.state
        
        # Validate it's the correct player's turn
        if state.current_player() != player:
            raise ValueError(f"Not player {player}'s turn")
        
        # Validate the action is legal
        if action not in state.legal_actions():
            raise ValueError(f"Action {action} is not legal")
        
        # Get action description before applying
        action_desc = state.action_to_string(action, player)
        
        # Apply the action
        state.apply_action(action)
        
        # Convert numpy types to Python native types
        current_player = state.current_player()
        winner = state.winner()
        
        return {
            "success": True,
            "board": state.board_array().tolist(),
            "current_player": int(current_player) if current_player is not None and current_player >= 0 else current_player,
            "is_terminal": bool(state.is_terminal()),
            "winner": int(winner) if winner is not None else None,
            "legal_actions": [int(a) for a in state.legal_actions()],
            "action_description": action_desc
        }
    
    def make_ai_move(self, game_id: str) -> dict:
        """Make an AI move in a game (works for both human_vs_ai and ai_vs_ai modes)."""
        session = self.get_game(game_id)
        if not session:
            raise ValueError(f"Game {game_id} not found")
        
        state = session.state
        current_player = state.current_player()
        
        # Determine which AI agent to use
        agent = None
        if session.mode == "human_vs_ai":
            if not session.ai_agent:
                raise ValueError("No AI agent configured for this game")
            if current_player != session.ai_player:
                raise ValueError("Not AI's turn")
            agent = session.ai_agent
        elif session.mode == "ai_vs_ai":
            if current_player == 0:
                agent = session.ai_agent_player0
            else:
                agent = session.ai_agent_player1
            
            if not agent:
                raise ValueError(f"No AI agent configured for player {current_player}")
        else:
            raise ValueError("AI moves not supported in human_vs_human mode")
        
        # Get AI's move
        action = agent.step(state)
        
        if action == INVALID_ACTION:
            raise ValueError("AI returned invalid action")
        
        # Get action description before applying
        action_desc = state.action_to_string(action, current_player)
        
        # Apply the action
        state.apply_action(action)
        
        # Convert numpy types to Python native types
        new_current_player = state.current_player()
        winner = state.winner()
        
        return {
            "success": True,
            "action": int(action),
            "action_description": action_desc,
            "board": state.board_array().tolist(),
            "current_player": int(new_current_player) if new_current_player is not None and new_current_player >= 0 else new_current_player,
            "is_terminal": bool(state.is_terminal()),
            "winner": int(winner) if winner is not None else None,
            "legal_actions": [int(a) for a in state.legal_actions()]
        }
    
    def get_action_details(self, game_id: str, action_id: int) -> dict:
        """Get details about a specific action."""
        session = self.get_game(game_id)
        if not session:
            raise ValueError(f"Game {game_id} not found")
        
        state = session.state
        
        # Import action mapping from dao module
        from games.dao import _ACTIONID_TO_ACTION
        
        if action_id not in _ACTIONID_TO_ACTION:
            raise ValueError(f"Invalid action ID: {action_id}")
        
        row, col, direction = _ACTIONID_TO_ACTION[action_id]
        
        return {
            "action_id": action_id,
            "row": row,
            "col": col,
            "direction": direction,
            "description": state.action_to_string(action_id)
        }
    
    @staticmethod
    def get_available_agents() -> list:
        """Get list of available AI agents."""
        return [
            {
                "id": "naive",
                "name": "Naive Agent",
                "description": "One-step lookahead: wins if possible, blocks opponent wins, otherwise random"
            },
            {
                "id": "random",
                "name": "Random Agent",
                "description": "Selects moves uniformly at random from legal actions"
            },
            {
                "id": "afterstate",
                "name": "Afterstate Agent",
                "description": "Trained afterstate value function (TD learning with curriculum + self-play)"
            }
        ]


# Global game manager instance

    def get_w_values(self, game_id: str, piece_row: Optional[int] = None, piece_col: Optional[int] = None) -> dict:
        """Get W-values for current player's pieces or specific piece."""
        session = self.get_game(game_id)
        if not session:
            raise ValueError(f"Game {game_id} not found")
        
        state = session.state
        current_player = state.current_player()
        
        # Find any afterstate agent in the game (regardless of whose turn it is)
        agent = None
        agent_player_id = None
        if session.mode == "human_vs_ai" and isinstance(session.ai_agent, AfterstateAgent):
            agent = session.ai_agent
            agent_player_id = session.ai_agent.player_id()
        elif session.mode == "ai_vs_ai":
            if isinstance(session.ai_agent_player0, AfterstateAgent):
                agent = session.ai_agent_player0
                agent_player_id = 0
            elif isinstance(session.ai_agent_player1, AfterstateAgent):
                agent = session.ai_agent_player1
                agent_player_id = 1
        
        # If no afterstate agent in game, use analysis agents
        if agent is None:
            if current_player == 0 and self.analysis_agent_p0:
                agent = self.analysis_agent_p0
                agent_player_id = 0
            elif current_player == 1 and self.analysis_agent_p1:
                agent = self.analysis_agent_p1
                agent_player_id = 1
        
        # Check if we have an agent for analysis
        if agent is None:
            return {
                "has_afterstate_agent": False,
                "current_player": int(current_player)
            }
        
        if piece_row is not None and piece_col is not None:
            # Get W-values for specific piece
            return self._get_piece_action_values(state, agent, piece_row, piece_col, current_player, agent_player_id)
        else:
            # Get max W-values for all pieces
            return self._get_all_piece_values(state, agent, current_player, agent_player_id)

    def _get_current_agent(self, session: GameSession, current_player: int):
        """Get the agent for the current player."""
        if session.mode == "human_vs_ai":
            return session.ai_agent if current_player == session.ai_player else None
        elif session.mode == "ai_vs_ai":
            return session.ai_agent_player0 if current_player == 0 else session.ai_agent_player1
        return None

    def _get_all_piece_values(self, state, agent, current_player, agent_player_id) -> dict:
        """Get max W-value for each piece belonging to current player."""
        from games.dao import _ACTIONID_TO_ACTION, _PLAYER_TOKENS
        from games.dao_hash import canonical_index
        
        token = _PLAYER_TOKENS[current_player]
        board = state.board_array()
        legal_actions = state.legal_actions()
        
        # Group actions by piece
        piece_actions = {}
        for action in legal_actions:
            row, col, direction = _ACTIONID_TO_ACTION[action]
            if board[row][col] == token:
                if (row, col) not in piece_actions:
                    piece_actions[(row, col)] = []
                piece_actions[(row, col)].append(action)
        
        # Calculate max W-value for each piece
        # Agent's W-table is already from agent's perspective (negated if agent is player 1)
        # If viewing different player's pieces, negate to get their perspective
        sign = -1 if current_player != agent_player_id else 1
        piece_values = []
        for (row, col), actions in piece_actions.items():
            w_values = [agent._w[canonical_index(state.board_after_action(a).flatten().astype(np.int8))] 
                        for a in actions]
            # Apply sign to each value BEFORE taking max
            signed_values = [v * sign for v in w_values]
            piece_values.append({
                "row": int(row),
                "col": int(col),
                "max_w_value": float(max(signed_values)),
                "action_count": len(actions)
            })
        
        return {
            "has_afterstate_agent": True,
            "current_player": int(current_player),
            "piece_values": piece_values
        }

    def _get_piece_action_values(self, state, agent, piece_row, piece_col, current_player, agent_player_id) -> dict:
        """Get W-values for all actions from a specific piece."""
        from games.dao import _ACTIONID_TO_ACTION, _PLAYER_TOKENS
        from games.dao_hash import canonical_index
        
        token = _PLAYER_TOKENS[current_player]
        board = state.board_array()
        
        # Validate piece belongs to current player
        if board[piece_row][piece_col] != token:
            raise ValueError(f"No piece at ({piece_row}, {piece_col}) for player {current_player}")
        
        # Get all legal actions from this piece
        # Agent's W-table is already from agent's perspective (negated if agent is player 1)
        # If viewing different player's pieces, negate to get their perspective
        sign = -1 if current_player != agent_player_id else 1
        legal_actions = state.legal_actions()
        piece_actions = []
        
        for action in legal_actions:
            row, col, direction = _ACTIONID_TO_ACTION[action]
            if row == piece_row and col == piece_col:
                # Calculate W-value
                afterstate_board = state.board_after_action(action)
                w_value = agent._w[canonical_index(afterstate_board.flatten().astype(np.int8))]
                
                # Find target cell (where piece lands)
                target_row, target_col = self._find_target_cell(board, piece_row, piece_col, direction)
                
                piece_actions.append({
                    "action_id": int(action),
                    "direction": direction,
                    "w_value": float(w_value * sign),
                    "target_row": int(target_row),
                    "target_col": int(target_col)
                })
        
        return {
            "has_afterstate_agent": True,
            "current_player": int(current_player),
            "selected_piece": {
                "row": int(piece_row),
                "col": int(piece_col)
            },
            "action_values": piece_actions
        }

    def _find_target_cell(self, board, row, col, direction) -> tuple:
        """Find where a piece would land when moved in a direction."""
        from games.dao import _DIRECTION_TUPLES
        
        dr, dc = _DIRECTION_TUPLES[direction]
        r, c = row, col
        
        while True:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= 4 or nc >= 4 or board[nr][nc] != 0:
                break
            r, c = nr, nc
        
        return r, c


game_manager = GameManager()

# Made with Bob
