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
import numpy as np


class RandomAgent:
    """Random agent for AI gameplay."""
    
    def __init__(self, player_id, rng=None):
        self._player_id = player_id
        self._rng = rng or np.random.default_rng()
    
    def player_id(self):
        return self._player_id
    
    def step(self, state):
        actions = state.legal_actions(self._player_id)
        if not actions:
            return INVALID_ACTION
        return self._rng.choice(actions)


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
            }
        ]


# Global game manager instance
game_manager = GameManager()

# Made with Bob
