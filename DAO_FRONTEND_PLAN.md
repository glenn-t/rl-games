# Dao Game Frontend & Backend Implementation Plan

## Overview
Build a web-based interface for the Dao board game with:
- **Frontend**: React (using Vite)
- **Backend**: FastAPI (Python)
- **Game Modes**: Human vs Human, Human vs AI (with agent selection)
- **Move Interface**: Click piece → Select direction from 8 buttons

## Project Structure

```
rl-games/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── game_manager.py      # Game session management
│   ├── models.py            # Pydantic models for API
│   ├── requirements.txt     # Backend dependencies
│   └── README.md            # Backend setup instructions
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main application component
│   │   ├── components/
│   │   │   ├── GameBoard.jsx       # 4x4 board display
│   │   │   ├── Cell.jsx            # Individual cell component
│   │   │   ├── DirectionButtons.jsx # 8-direction controls
│   │   │   ├── GameInfo.jsx        # Status display
│   │   │   └── GameSetup.jsx       # Mode/agent selection
│   │   ├── services/
│   │   │   └── api.js       # Backend API calls
│   │   ├── hooks/
│   │   │   └── useGameState.js # Game state management
│   │   └── styles/
│   │       └── App.css      # Styling
│   ├── package.json
│   └── README.md            # Frontend setup instructions
├── games/
│   └── dao.py               # Existing game logic
└── agents/
    └── naive.py             # Existing AI agents
```

## Backend Architecture (FastAPI)

### API Endpoints

#### 1. Create New Game
```
POST /api/game/new
Body: {
  "mode": "human_vs_human" | "human_vs_ai",
  "ai_agent": "naive" | "random" (optional, required if mode is human_vs_ai),
  "ai_player": 0 | 1 (optional, which player is AI, default 1)
}
Response: {
  "game_id": "uuid",
  "board": [[0,1,2,1], [1,0,0,2], ...],
  "current_player": 0,
  "is_terminal": false,
  "winner": null,
  "legal_actions": [0, 1, 5, ...],
  "mode": "human_vs_ai",
  "ai_agent": "naive"
}
```

#### 2. Get Game State
```
GET /api/game/{game_id}
Response: {
  "game_id": "uuid",
  "board": [[...]],
  "current_player": 0,
  "is_terminal": false,
  "winner": null,
  "legal_actions": [...],
  "history": [...]
}
```

#### 3. Make Move
```
POST /api/game/{game_id}/move
Body: {
  "action": 42,  # action ID from legal_actions
  "player": 0    # player making the move
}
Response: {
  "success": true,
  "board": [[...]],
  "current_player": 1,
  "is_terminal": false,
  "winner": null,
  "legal_actions": [...],
  "action_description": "Player x: (1, 2) UP"
}
```

#### 4. AI Move (for Human vs AI mode)
```
POST /api/game/{game_id}/ai-move
Response: {
  "success": true,
  "action": 42,
  "action_description": "Player o: (2, 1) DOWN",
  "board": [[...]],
  "current_player": 0,
  "is_terminal": false,
  "winner": null,
  "legal_actions": [...]
}
```

#### 5. Get Available Agents
```
GET /api/agents
Response: {
  "agents": [
    {"id": "naive", "name": "Naive Agent", "description": "One-step lookahead"},
    {"id": "random", "name": "Random Agent", "description": "Random moves"}
  ]
}
```

#### 6. Get Action Details
```
GET /api/game/{game_id}/action/{action_id}
Response: {
  "action_id": 42,
  "row": 1,
  "col": 2,
  "direction": "UP",
  "description": "Player x: (1, 2) UP"
}
```

### Game Session Management
- In-memory dictionary: `games: Dict[str, GameSession]`
- GameSession stores:
  - `game_id`: UUID
  - `state`: DaoState instance
  - `mode`: "human_vs_human" | "human_vs_ai"
  - `ai_agent`: Agent instance (if applicable)
  - `ai_player`: 0 or 1 (if applicable)
  - `created_at`: timestamp

### CORS Configuration
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Frontend Architecture (React)

### Component Hierarchy

```
App
├── GameSetup (if no active game)
│   ├── Mode selector (Human vs Human / Human vs AI)
│   └── Agent selector (if AI mode)
└── GameContainer (if game active)
    ├── GameInfo
    │   ├── Current player indicator
    │   ├── Game status
    │   └── Winner announcement
    ├── GameBoard
    │   └── Cell × 16 (4×4 grid)
    │       ├── Empty cell
    │       ├── Player 0 piece (X)
    │       └── Player 1 piece (O)
    ├── DirectionButtons (if piece selected)
    │   └── 8 direction buttons
    └── GameControls
        ├── New Game button
        └── Reset button
```

### State Management

Using React hooks (`useState`, `useEffect`):

```javascript
const [gameId, setGameId] = useState(null);
const [board, setBoard] = useState(null);
const [currentPlayer, setCurrentPlayer] = useState(0);
const [isTerminal, setIsTerminal] = useState(false);
const [winner, setWinner] = useState(null);
const [legalActions, setLegalActions] = useState([]);
const [selectedCell, setSelectedCell] = useState(null); // {row, col}
const [gameMode, setGameMode] = useState(null);
const [aiAgent, setAiAgent] = useState('naive');
```

### User Interaction Flow

#### Human vs Human Mode:
1. User selects "Human vs Human" mode
2. Click "Start Game" → POST /api/game/new
3. Board displays with Player 0's pieces highlighted
4. Player 0 clicks their piece → Cell highlights, direction buttons appear
5. Player 0 clicks direction → POST /api/game/{id}/move
6. Board updates, switches to Player 1
7. Repeat until game ends

#### Human vs AI Mode:
1. User selects "Human vs AI" mode
2. User selects AI agent (Naive/Random)
3. User selects which player is AI (default: Player 1)
4. Click "Start Game" → POST /api/game/new
5. If AI is Player 0, immediately call POST /api/game/{id}/ai-move
6. Human player makes move → POST /api/game/{id}/move
7. After human move, automatically call POST /api/game/{id}/ai-move
8. Repeat until game ends

### API Service Layer

```javascript
// services/api.js
const API_BASE = 'http://localhost:8000/api';

export const gameApi = {
  createGame: async (mode, aiAgent, aiPlayer) => { ... },
  getGameState: async (gameId) => { ... },
  makeMove: async (gameId, action, player) => { ... },
  makeAiMove: async (gameId) => { ... },
  getAgents: async () => { ... },
  getActionDetails: async (gameId, actionId) => { ... }
};
```

### Styling Approach

**Color Scheme:**
- Player 0 (X): Blue (#3b82f6)
- Player 1 (O): Red (#ef4444)
- Empty cells: Light gray (#f3f4f6)
- Selected cell: Yellow highlight (#fef3c7)
- Board background: Dark gray (#1f2937)

**Layout:**
- Centered game board (400px × 400px)
- Grid layout for 4×4 cells
- Direction buttons in a 3×3 grid below board
- Game info panel above board
- Responsive design for mobile

**Animations:**
- Smooth piece movement (CSS transitions)
- Highlight effects on hover
- Victory celebration animation

## Implementation Steps

### Phase 1: Backend Setup
1. Create `backend/` directory structure
2. Set up FastAPI application with CORS
3. Implement game session manager
4. Create Pydantic models for requests/responses
5. Implement all API endpoints
6. Add agent integration (NaiveAgent, RandomAgent)
7. Test endpoints with curl/Postman

### Phase 2: Frontend Setup
1. Initialize Vite React project in `frontend/`
2. Set up project structure (components, services, hooks)
3. Create API service layer
4. Implement basic component structure

### Phase 3: Core Game UI
1. Build GameBoard component with 4×4 grid
2. Implement Cell component with piece display
3. Add piece selection logic
4. Create DirectionButtons component
5. Implement move execution

### Phase 4: Game Modes
1. Create GameSetup component
2. Add mode selection (Human vs Human / Human vs AI)
3. Add agent selection dropdown
4. Implement AI move automation
5. Add game controls (new game, reset)

### Phase 5: Polish
1. Add comprehensive styling
2. Implement animations and transitions
3. Add error handling and loading states
4. Display move history
5. Show victory conditions explanation

### Phase 6: Testing & Documentation
1. Test all game flows
2. Test with different agents
3. Create setup instructions
4. Add inline code documentation

## Victory Conditions Display

The UI should explain the four victory conditions:
1. **Line**: Complete row or column
2. **Square**: 2×2 square of same pieces
3. **Corners**: All four corner cells
4. **Corner Blocked**: Opponent trapped in corner

## Technical Considerations

### Backend
- Use UUID for game IDs
- Implement game cleanup (remove old games after timeout)
- Add error handling for invalid moves
- Validate player turns
- Handle edge cases (game already ended, invalid game ID)

### Frontend
- Handle network errors gracefully
- Show loading states during API calls
- Disable UI during AI moves
- Validate moves client-side before sending
- Add keyboard shortcuts (optional)
- Make responsive for mobile devices

### Performance
- Backend: In-memory storage is fine for local development
- Frontend: Optimize re-renders with React.memo if needed
- Consider adding move animation delays for better UX

## Development Workflow

1. Start backend: `cd backend && uvicorn main:app --reload`
2. Start frontend: `cd frontend && npm run dev`
3. Access game at `http://localhost:5173`
4. Backend API at `http://localhost:8000`
5. API docs at `http://localhost:8000/docs`

## Future Enhancements (Optional)

- Move history with undo functionality
- Save/load game state
- Multiple simultaneous games
- Game statistics and analytics
- Multiplayer over network (WebSockets)
- Tournament mode
- More AI agents (MCTS, Neural Network)
- Mobile app version
- Sound effects and music