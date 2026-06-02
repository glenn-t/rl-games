"""Pydantic models for API request/response validation."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class NewGameRequest(BaseModel):
    """Request to create a new game."""
    mode: Literal["human_vs_human", "human_vs_ai", "ai_vs_ai"] = Field(
        ..., description="Game mode"
    )
    ai_agent: Optional[Literal["naive", "random", "afterstate"]] = Field(
        None, description="AI agent type for player 1 (required if mode is human_vs_ai or ai_vs_ai)"
    )
    ai_player: Optional[Literal[0, 1]] = Field(
        1, description="Which player is AI in human_vs_ai mode (0 or 1, default 1)"
    )
    ai_agent_player0: Optional[Literal["naive", "random", "afterstate"]] = Field(
        None, description="AI agent type for player 0 in ai_vs_ai mode"
    )
    ai_agent_player1: Optional[Literal["naive", "random", "afterstate"]] = Field(
        None, description="AI agent type for player 1 in ai_vs_ai mode"
    )


class GameStateResponse(BaseModel):
    """Response containing current game state."""
    game_id: str
    board: List[List[int]]
    current_player: int
    is_terminal: bool
    winner: Optional[int]
    legal_actions: List[int]
    history: List[int]
    mode: str
    ai_agent: Optional[str] = None
    ai_player: Optional[int] = None


class MoveRequest(BaseModel):
    """Request to make a move."""
    action: int = Field(..., description="Action ID from legal_actions")
    player: int = Field(..., description="Player making the move (0 or 1)")


class MoveResponse(BaseModel):
    """Response after making a move."""
    success: bool
    board: List[List[int]]
    current_player: int
    is_terminal: bool
    winner: Optional[int]
    legal_actions: List[int]
    action_description: str
    message: Optional[str] = None


class AIAgentInfo(BaseModel):
    """Information about an available AI agent."""
    id: str
    name: str
    description: str


class AgentsResponse(BaseModel):
    """Response listing available AI agents."""
    agents: List[AIAgentInfo]


class ActionDetailsResponse(BaseModel):
    """Response with details about a specific action."""
    action_id: int
    row: int
    col: int
    direction: str
    description: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None


class PieceWValue(BaseModel):
    """W-value information for a piece."""
    row: int
    col: int
    max_w_value: float
    action_count: int


class ActionWValue(BaseModel):
    """W-value information for a specific action."""
    action_id: int
    direction: str
    w_value: float
    target_row: int
    target_col: int


class SelectedPiece(BaseModel):
    """Selected piece coordinates."""
    row: int
    col: int


class WValuesResponse(BaseModel):
    """Response containing W-values."""
    has_afterstate_agent: bool
    current_player: int
    piece_values: Optional[List[PieceWValue]] = None
    selected_piece: Optional[SelectedPiece] = None
    action_values: Optional[List[ActionWValue]] = None

# Made with Bob
