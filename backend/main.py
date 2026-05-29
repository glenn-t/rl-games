"""FastAPI backend for Dao game."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from .models import (
    NewGameRequest,
    GameStateResponse,
    MoveRequest,
    MoveResponse,
    AgentsResponse,
    ActionDetailsResponse,
    ErrorResponse
)
from .game_manager import game_manager

# Create FastAPI app
app = FastAPI(
    title="Dao Game API",
    description="Backend API for the Dao board game",
    version="1.0.0"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default port
        "http://localhost:3000",  # Alternative React port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Dao Game API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.post("/api/game/new", response_model=GameStateResponse)
async def create_new_game(request: NewGameRequest):
    """Create a new game session."""
    try:
        # Validate AI agent is provided if mode is human_vs_ai
        if request.mode == "human_vs_ai" and not request.ai_agent:
            raise HTTPException(
                status_code=400,
                detail="ai_agent is required when mode is human_vs_ai"
            )
        
        # Validate AI agents are provided if mode is ai_vs_ai
        if request.mode == "ai_vs_ai":
            if not request.ai_agent_player0 or not request.ai_agent_player1:
                raise HTTPException(
                    status_code=400,
                    detail="ai_agent_player0 and ai_agent_player1 are required when mode is ai_vs_ai"
                )
        
        # Create the game
        session = game_manager.create_game(
            mode=request.mode,
            ai_agent_type=request.ai_agent,
            ai_player=request.ai_player,
            ai_agent_player0_type=request.ai_agent_player0,
            ai_agent_player1_type=request.ai_agent_player1
        )
        
        return session.to_dict()
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/api/game/{game_id}", response_model=GameStateResponse)
async def get_game_state(game_id: str):
    """Get the current state of a game."""
    try:
        session = game_manager.get_game(game_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
        
        return session.to_dict()
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/api/game/{game_id}/move", response_model=MoveResponse)
async def make_move(game_id: str, request: MoveRequest):
    """Make a move in a game."""
    try:
        result = game_manager.make_move(game_id, request.action, request.player)
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/api/game/{game_id}/ai-move")
async def make_ai_move(game_id: str):
    """Make an AI move in a game."""
    try:
        result = game_manager.make_ai_move(game_id)
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/api/agents", response_model=AgentsResponse)
async def get_agents():
    """Get list of available AI agents."""
    try:
        agents = game_manager.get_available_agents()
        return {"agents": agents}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/api/game/{game_id}/action/{action_id}", response_model=ActionDetailsResponse)
async def get_action_details(game_id: str, action_id: int):
    """Get details about a specific action."""
    try:
        details = game_manager.get_action_details(game_id, action_id)
        return details
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.delete("/api/game/{game_id}")
async def delete_game(game_id: str):
    """Delete a game session."""
    try:
        if game_id in game_manager.games:
            del game_manager.games[game_id]
            return {"message": f"Game {game_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

def serve():
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    serve()    

# Made with Bob
