# Dao Game Backend

FastAPI backend server for the Dao board game.

## Setup

1. **Install dependencies:**

```bash
cd backend
pip install -r requirements.txt
```

Or using uv (if available):

```bash
cd backend
uv pip install -r requirements.txt
```

2. **Run the server:**

**Important:** You must run the server from within the backend directory!

```bash
# Navigate to backend directory first
cd backend

# Then run the server
python main.py
```

Or using uvicorn directly:

```bash
# From the backend directory
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Note:** If you get "ModuleNotFoundError: No module named 'models'", make sure you're in the `backend` directory when running the command.

The server will start on `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## API Endpoints

### Create New Game
```
POST /api/game/new
Body: {
  "mode": "human_vs_human" | "human_vs_ai",
  "ai_agent": "naive" | "random" (optional),
  "ai_player": 0 | 1 (optional, default 1)
}
```

### Get Game State
```
GET /api/game/{game_id}
```

### Make Move
```
POST /api/game/{game_id}/move
Body: {
  "action": <action_id>,
  "player": 0 | 1
}
```

### Make AI Move
```
POST /api/game/{game_id}/ai-move
```

### Get Available Agents
```
GET /api/agents
```

### Get Action Details
```
GET /api/game/{game_id}/action/{action_id}
```

### Delete Game
```
DELETE /api/game/{game_id}
```

## Game Modes

- **human_vs_human**: Two human players take turns
- **human_vs_ai**: One human player vs AI agent

## Available AI Agents

- **naive**: One-step lookahead agent (wins if possible, blocks opponent wins, otherwise random)
- **random**: Selects moves uniformly at random

## Development

The backend uses:
- FastAPI for the web framework
- Pydantic for data validation
- In-memory storage for game sessions
- CORS middleware for React frontend communication

## Project Structure

```
backend/
├── main.py           # FastAPI application and endpoints
├── game_manager.py   # Game session management
├── models.py         # Pydantic models for API
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Notes

- Game sessions are stored in memory and will be lost when the server restarts
- The server allows CORS requests from `localhost:5173` (Vite) and `localhost:3000` (React)
- The backend imports game logic from the parent `games/` directory