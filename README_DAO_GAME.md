# Dao Game - Full Stack Web Application

A complete web-based implementation of the Dao board game with React frontend and FastAPI backend.

## Overview

Dao is a strategic two-player board game played on a 4×4 grid where pieces slide in eight directions until blocked. Players win by achieving one of four victory conditions: completing a line, forming a square, occupying all corners, or trapping an opponent in a corner.

## Features

- **Two Game Modes**: Human vs Human or Human vs AI
- **AI Opponents**: Choose between Naive (strategic) or Random agents
- **Interactive UI**: Click-to-select pieces with directional controls
- **Real-time Updates**: Instant game state synchronization
- **Victory Detection**: Automatic win condition checking
- **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
rl-games/
├── backend/                 # FastAPI backend
│   ├── main.py             # API endpoints
│   ├── game_manager.py     # Game session management
│   ├── models.py           # Pydantic models
│   ├── requirements.txt    # Python dependencies
│   └── README.md           # Backend documentation
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API service
│   │   ├── App.jsx        # Main app
│   │   └── App.css        # Styling
│   ├── package.json       # Node dependencies
│   └── README.md          # Frontend documentation
├── games/
│   └── dao.py             # Core game logic
├── agents/
│   └── naive.py           # AI agents
└── DAO_FRONTEND_PLAN.md   # Implementation plan
```

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 16+
- npm or yarn

### 1. Start the Backend

```bash
# Install Python dependencies
cd backend
pip install -r requirements.txt

# Start the FastAPI server
python main.py
```

Backend will run on `http://localhost:8000`
API docs available at `http://localhost:8000/docs`

### 2. Start the Frontend

```bash
# Install Node dependencies
cd frontend
npm install

# Start the development server
npm run dev
```

Frontend will run on `http://localhost:5173`

### 3. Play the Game

1. Open `http://localhost:5173` in your browser
2. Select game mode (Human vs Human or Human vs AI)
3. If playing vs AI, choose the agent type and which player the AI controls
4. Click "Start Game"
5. Click your piece, then select a direction to move
6. Win by achieving one of the four victory conditions!

## Game Rules

### Setup
- 4×4 board
- Player 0 (Blue X) starts on the main diagonal
- Player 1 (Red O) starts on the anti-diagonal
- Player 0 goes first

### Gameplay
1. On your turn, select one of your pieces
2. Choose a direction (8 directions available)
3. The piece slides in that direction until it hits:
   - The edge of the board
   - Another piece (yours or opponent's)
4. The piece stops in the last empty cell before the obstacle

### Victory Conditions

Win by achieving any of these:

1. **Line**: Fill an entire row or column with your pieces
2. **Square**: Form a 2×2 square with your pieces
3. **Corners**: Occupy all four corner cells
4. **Corner Block**: Trap an opponent's piece in a corner by surrounding it

## API Documentation

### Backend Endpoints

- `POST /api/game/new` - Create a new game
- `GET /api/game/{game_id}` - Get current game state
- `POST /api/game/{game_id}/move` - Make a move
- `POST /api/game/{game_id}/ai-move` - Make an AI move
- `GET /api/agents` - List available AI agents
- `GET /api/game/{game_id}/action/{action_id}` - Get action details
- `DELETE /api/game/{game_id}` - Delete a game

Full API documentation: `http://localhost:8000/docs`

## AI Agents

### Naive Agent
- Uses one-step lookahead strategy
- Prioritizes: Win → Block opponent win → Random
- Good for beginners to intermediate players

### Random Agent
- Selects moves uniformly at random
- Good for testing and casual play

## Development

### Backend Development

```bash
cd backend
# Run with auto-reload
uvicorn main:app --reload
```

### Frontend Development

```bash
cd frontend
# Run with hot module replacement
npm run dev
```

### Building for Production

```bash
# Backend: No build needed, just run with uvicorn
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend: Build static files
cd frontend
npm run build
# Serve from dist/ directory
```

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Pydantic**: Data validation
- **NumPy**: Numerical operations
- **Uvicorn**: ASGI server

### Frontend
- **React 18**: UI framework
- **Vite**: Build tool
- **CSS3**: Styling with gradients and animations
- **Fetch API**: HTTP client

## Architecture

### Backend Architecture
- RESTful API design
- In-memory game session storage
- Stateless request handling
- CORS enabled for local development

### Frontend Architecture
- Component-based React architecture
- Hooks for state management
- Service layer for API communication
- Responsive CSS Grid layout

## Troubleshooting

### Backend Issues

**Import errors:**
```bash
# Ensure you're in the project root when running backend
cd backend
python main.py
```

**Port already in use:**
```bash
# Use a different port
uvicorn main:app --port 8001
```

### Frontend Issues

**Cannot connect to backend:**
- Verify backend is running on port 8000
- Check browser console for CORS errors
- Ensure `vite.config.js` proxy is configured correctly

**Build errors:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

## Future Enhancements

- [ ] Move history with undo functionality
- [ ] Save/load game state
- [ ] Multiplayer over network (WebSockets)
- [ ] Tournament mode
- [ ] More AI agents (MCTS, Neural Network)
- [ ] Game replay and analysis
- [ ] Sound effects and animations
- [ ] Mobile app version

## Contributing

This is a learning project for reinforcement learning and game AI. Contributions are welcome!

## License

See main project LICENSE file.

## Credits

- Game logic based on the Dao board game
- Built as part of the rl-games reinforcement learning project
- Uses OpenSpiel framework concepts for game implementation