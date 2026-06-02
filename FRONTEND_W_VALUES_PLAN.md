# Frontend W-Values Visualization Plan

## Overview
Add visualization of afterstate W-matrix values to help understand the afterstate agent's decision-making process. This includes showing values on pieces before selection and on board cells/direction buttons after selection.

## Requirements Summary

### Display Requirements
1. **Before piece selection**: Show the maximum W-value of all actions associated with moving each piece (display on the piece itself)
2. **After piece selection**: Show the W-value for moving the selected piece in each direction (display on board cells where piece would land)
3. **Visibility**: Show for both players when any player is using the afterstate agent
4. **Current player only**: Only show values on the current player's pieces
5. **Format**: Display as raw numbers (e.g., "0.85", "-0.42")

### Game Control Requirements
1. **Pause by default**: Games start paused, requiring manual advancement
2. **Step button**: Manually advance one move at a time
3. **Autoplay controls**: Fast and slow autoplay buttons
4. **Apply to both modes**: Works for both AI vs AI and Human vs AI modes
5. **Purpose**: Allow viewing W-values before the AI agent takes a move

## Architecture Analysis

### Backend Components

#### AfterstateAgent (`agents/afterstate_agent.py`)
- Stores W-table as numpy array: `self._w` (shape: `(N_CANONICAL_STATES,)`)
- For player 1, W-values are flipped: `self._w = self._w * -1`
- Uses `_afterstate_idx(state, action)` to get canonical index for afterstate
- Selects action with highest W-value: `max(legal, key=lambda a: self._w[_afterstate_idx(state, a)])`

#### GameManager (`backend/game_manager.py`)
- Manages game sessions with AI agents
- Has access to agent instances (`ai_agent`, `ai_agent_player0`, `ai_agent_player1`)
- Currently exposes: `make_move()`, `make_ai_move()`, `get_action_details()`

#### DaoState (`games/dao.py`)
- Provides `legal_actions()` - list of legal action IDs
- Provides `board_after_action(action)` - efficient board copy with action applied
- Actions encoded as integers via `_ACTIONID_TO_ACTION` mapping

#### Canonical Indexing (`games/dao_hash.py`)
- `canonical_index(state_array)` - converts flattened board to canonical index
- Used by afterstate agent to look up W-values

### Frontend Components

#### App.jsx
- Main game state management
- Handles game flow, move execution, AI moves
- Currently has speed control in GameSetup, needs to move to game controls

#### GameBoard.jsx
- Displays 4x4 board grid
- Renders Cell components
- Handles cell click events

#### Cell.jsx
- Displays individual board cells
- Shows player tokens (X/O)
- Handles selection state

#### DirectionButtons.jsx
- Shows 8 directional movement buttons
- Currently displays after piece selection
- Validates legal moves

## API Design

### New Endpoints

#### 1. GET `/api/game/{game_id}/w-values`
**Purpose**: Get W-values for all legal actions (for displaying on pieces)

**Response**:
```json
{
  "has_afterstate_agent": true,
  "current_player": 0,
  "piece_values": [
    {
      "row": 0,
      "col": 0,
      "max_w_value": 0.85,
      "action_count": 3
    },
    {
      "row": 1,
      "col": 1,
      "max_w_value": 0.42,
      "action_count": 5
    }
  ]
}
```

**Logic**:
- Check if current player is using afterstate agent
- For each piece belonging to current player:
  - Get all legal actions from that piece
  - Calculate W-value for each action
  - Return max W-value for that piece

#### 2. GET `/api/game/{game_id}/w-values/piece/{row}/{col}`
**Purpose**: Get W-values for all actions from a specific piece (for displaying on board cells)

**Response**:
```json
{
  "has_afterstate_agent": true,
  "current_player": 0,
  "row": 0,
  "col": 0,
  "action_values": [
    {
      "action_id": 5,
      "direction": "DOWN",
      "w_value": 0.85,
      "target_row": 3,
      "target_col": 0
    },
    {
      "action_id": 6,
      "direction": "RIGHT",
      "w_value": 0.42,
      "target_row": 0,
      "target_col": 3
    }
  ]
}
```

**Logic**:
- Check if current player is using afterstate agent
- Get all legal actions from specified piece (row, col)
- For each action:
  - Calculate W-value
  - Determine target cell (where piece would land)
  - Return action details with W-value

### Alternative: Single Comprehensive Endpoint

#### GET `/api/game/{game_id}/w-values?piece_row={row}&piece_col={col}`
**Purpose**: Get W-values for all pieces, or for a specific piece if coordinates provided

**Response (no piece specified)**:
```json
{
  "has_afterstate_agent": true,
  "current_player": 0,
  "piece_values": [
    {
      "row": 0,
      "col": 0,
      "max_w_value": 0.85,
      "action_count": 3
    }
  ]
}
```

**Response (piece specified)**:
```json
{
  "has_afterstate_agent": true,
  "current_player": 0,
  "selected_piece": {
    "row": 0,
    "col": 0
  },
  "action_values": [
    {
      "action_id": 5,
      "direction": "DOWN",
      "w_value": 0.85,
      "target_row": 3,
      "target_col": 0
    }
  ]
}
```

**Recommendation**: Use the single comprehensive endpoint for simplicity and fewer API calls.

## Backend Implementation Plan

### 1. Add W-Value Calculation Methods to GameManager

```python
def get_w_values(self, game_id: str, piece_row: Optional[int] = None, piece_col: Optional[int] = None) -> dict:
    """Get W-values for current player's pieces or specific piece."""
    session = self.get_game(game_id)
    if not session:
        raise ValueError(f"Game {game_id} not found")
    
    state = session.state
    current_player = state.current_player()
    
    # Determine which agent to use
    agent = self._get_current_agent(session, current_player)
    
    # Check if agent is afterstate agent
    if not isinstance(agent, AfterstateAgent):
        return {
            "has_afterstate_agent": False,
            "current_player": int(current_player)
        }
    
    if piece_row is not None and piece_col is not None:
        # Get W-values for specific piece
        return self._get_piece_action_values(state, agent, piece_row, piece_col, current_player)
    else:
        # Get max W-values for all pieces
        return self._get_all_piece_values(state, agent, current_player)

def _get_current_agent(self, session: GameSession, current_player: int):
    """Get the agent for the current player."""
    if session.mode == "human_vs_ai":
        return session.ai_agent if current_player == session.ai_player else None
    elif session.mode == "ai_vs_ai":
        return session.ai_agent_player0 if current_player == 0 else session.ai_agent_player1
    return None

def _get_all_piece_values(self, state, agent, current_player) -> dict:
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
    piece_values = []
    for (row, col), actions in piece_actions.items():
        w_values = [agent._w[canonical_index(state.board_after_action(a).flatten().astype(np.int8))] 
                    for a in actions]
        piece_values.append({
            "row": int(row),
            "col": int(col),
            "max_w_value": float(max(w_values)),
            "action_count": len(actions)
        })
    
    return {
        "has_afterstate_agent": True,
        "current_player": int(current_player),
        "piece_values": piece_values
    }

def _get_piece_action_values(self, state, agent, piece_row, piece_col, current_player) -> dict:
    """Get W-values for all actions from a specific piece."""
    from games.dao import _ACTIONID_TO_ACTION, _PLAYER_TOKENS
    from games.dao_hash import canonical_index
    
    token = _PLAYER_TOKENS[current_player]
    board = state.board_array()
    
    # Validate piece belongs to current player
    if board[piece_row][piece_col] != token:
        raise ValueError(f"No piece at ({piece_row}, {piece_col}) for player {current_player}")
    
    # Get all legal actions from this piece
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
                "w_value": float(w_value),
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
```

### 2. Add API Endpoint to main.py

```python
@app.get("/api/game/{game_id}/w-values", response_model=WValuesResponse)
async def get_w_values(
    game_id: str,
    piece_row: Optional[int] = None,
    piece_col: Optional[int] = None
):
    """Get W-values for afterstate agent."""
    try:
        result = game_manager.get_w_values(game_id, piece_row, piece_col)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
```

### 3. Add Pydantic Models to models.py

```python
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
```

## Frontend Implementation Plan

### 1. Add W-Values API Service (services/api.js)

```javascript
export const getWValues = async (gameId, pieceRow = null, pieceCol = null) => {
  const params = new URLSearchParams();
  if (pieceRow !== null) params.append('piece_row', pieceRow);
  if (pieceCol !== null) params.append('piece_col', pieceCol);
  
  const url = `/api/game/${gameId}/w-values${params.toString() ? '?' + params.toString() : ''}`;
  const response = await fetch(`${API_BASE_URL}${url}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get W-values');
  }
  
  return response.json();
};
```

### 2. Update App.jsx State Management

Add new state variables:
```javascript
// W-values state
const [wValues, setWValues] = useState(null);
const [selectedPieceWValues, setSelectedPieceWValues] = useState(null);

// Game control state
const [isPaused, setIsPaused] = useState(true);
const [autoplaySpeed, setAutoplaySpeed] = useState(null); // null, 'slow', 'fast'
```

Add W-values fetching:
```javascript
const fetchWValues = async (gameId, pieceRow = null, pieceCol = null) => {
  try {
    const values = await getWValues(gameId, pieceRow, pieceCol);
    if (pieceRow !== null && pieceCol !== null) {
      setSelectedPieceWValues(values);
    } else {
      setWValues(values);
    }
  } catch (err) {
    console.error('Failed to fetch W-values:', err);
  }
};

// Fetch W-values after game state updates
useEffect(() => {
  if (gameId && !isTerminal && !isViewingHistory) {
    fetchWValues(gameId);
  }
}, [gameId, board, currentPlayer, isTerminal, isViewingHistory]);

// Fetch selected piece W-values when piece is selected
useEffect(() => {
  if (gameId && selectedCell && !isTerminal) {
    fetchWValues(gameId, selectedCell.row, selectedCell.col);
  } else {
    setSelectedPieceWValues(null);
  }
}, [gameId, selectedCell, isTerminal]);
```

### 3. Update Cell.jsx to Display W-Values

```javascript
const Cell = ({ 
  value, 
  row, 
  col, 
  isSelected, 
  isSelectable, 
  onClick,
  wValue = null,  // NEW: W-value to display on piece
  targetWValue = null  // NEW: W-value if this cell is a target
}) => {
  const getCellClass = () => {
    const classes = ['cell'];
    
    if (value === 1) {
      classes.push('player-0');
    } else if (value === 2) {
      classes.push('player-1');
    } else {
      classes.push('empty');
    }
    
    if (isSelected) {
      classes.push('selected');
    }
    
    if (isSelectable) {
      classes.push('selectable');
    }
    
    if (targetWValue !== null) {
      classes.push('target-cell');
    }
    
    return classes.join(' ');
  };
  
  const getCellContent = () => {
    if (value === 1) return 'X';
    if (value === 2) return 'O';
    return '';
  };
  
  return (
    <div
      className={getCellClass()}
      onClick={() => onClick(row, col)}
      data-row={row}
      data-col={col}
    >
      <span className="cell-content">{getCellContent()}</span>
      {wValue !== null && (
        <span className="w-value-badge">{wValue.toFixed(2)}</span>
      )}
      {targetWValue !== null && (
        <span className="target-w-value">{targetWValue.toFixed(2)}</span>
      )}
    </div>
  );
};
```

### 4. Update GameBoard.jsx to Pass W-Values

```javascript
const GameBoard = ({ 
  board, 
  currentPlayer, 
  selectedCell, 
  onCellClick, 
  isTerminal,
  wValues = null,  // NEW: W-values for all pieces
  selectedPieceWValues = null  // NEW: W-values for selected piece actions
}) => {
  // ... existing code ...
  
  // Helper to get W-value for a piece
  const getPieceWValue = (row, col) => {
    if (!wValues || !wValues.has_afterstate_agent) return null;
    const piece = wValues.piece_values?.find(p => p.row === row && p.col === col);
    return piece ? piece.max_w_value : null;
  };
  
  // Helper to get target W-value for a cell
  const getTargetWValue = (row, col) => {
    if (!selectedPieceWValues || !selectedPieceWValues.action_values) return null;
    const action = selectedPieceWValues.action_values.find(
      a => a.target_row === row && a.target_col === col
    );
    return action ? action.w_value : null;
  };
  
  return (
    <div className="game-board">
      {board.map((row, rowIndex) => (
        <div key={rowIndex} className="board-row">
          {row.map((cell, colIndex) => (
            <Cell
              key={`${rowIndex}-${colIndex}`}
              value={cell}
              row={rowIndex}
              col={colIndex}
              isSelected={isCellSelected(rowIndex, colIndex)}
              isSelectable={isCellSelectable(rowIndex, colIndex)}
              onClick={onCellClick}
              wValue={getPieceWValue(rowIndex, colIndex)}
              targetWValue={getTargetWValue(rowIndex, colIndex)}
            />
          ))}
        </div>
      ))}
    </div>
  );
};
```

### 5. Add Game Control Component (GameControls.jsx)

```javascript
const GameControls = ({ 
  isPaused, 
  autoplaySpeed,
  onPause,
  onStep,
  onAutoplay,
  disabled = false
}) => {
  return (
    <div className="game-controls">
      <h3>Game Controls</h3>
      <div className="control-buttons">
        <button
          className={`control-button ${isPaused ? 'active' : ''}`}
          onClick={onPause}
          disabled={disabled}
        >
          ⏸ Pause
        </button>
        <button
          className="control-button"
          onClick={onStep}
          disabled={disabled || !isPaused}
        >
          ⏭ Step
        </button>
        <button
          className={`control-button ${autoplaySpeed === 'slow' ? 'active' : ''}`}
          onClick={() => onAutoplay('slow')}
          disabled={disabled}
        >
          ▶ Slow
        </button>
        <button
          className={`control-button ${autoplaySpeed === 'fast' ? 'active' : ''}`}
          onClick={() => onAutoplay('fast')}
          disabled={disabled}
        >
          ▶▶ Fast
        </button>
      </div>
      <div className="control-status">
        {isPaused && <span>⏸ Paused</span>}
        {autoplaySpeed && <span>▶ Playing ({autoplaySpeed})</span>}
      </div>
    </div>
  );
};
```

### 6. Update App.jsx Game Flow Logic

```javascript
// Remove speed from GameSetup, start paused
const handleStartGame = async (gameMode, selectedAiAgent, selectedAiPlayer, aiAgent0, aiAgent1) => {
  try {
    setLoading(true);
    setError(null);
    setHistory([]);
    setViewingStep(null);
    setIsPaused(true);  // Start paused
    setAutoplaySpeed(null);
    
    const gameState = await createGame(gameMode, selectedAiAgent, selectedAiPlayer, aiAgent0, aiAgent1);
    setGameId(gameState.game_id);
    updateGameState(gameState, 'Game started');
    setMessage('Game started! Use controls to advance.');
    
    // Don't auto-start AI moves
  } catch (err) {
    setError(err.message);
    throw err;
  } finally {
    setLoading(false);
  }
};

// Handle step button
const handleStep = async () => {
  if (isTerminal || loading) return;
  
  if (mode === 'ai_vs_ai') {
    await handleAIMove(gameId);
  } else if (mode === 'human_vs_ai' && currentPlayer === aiPlayer) {
    await handleAIMove(gameId);
  } else {
    setMessage("It's your turn! Select a piece and direction.");
  }
};

// Handle autoplay
const handleAutoplay = (speed) => {
  if (speed === autoplaySpeed) {
    // Toggle off
    setAutoplaySpeed(null);
    setIsPaused(true);
  } else {
    setAutoplaySpeed(speed);
    setIsPaused(false);
    runAutoplay(speed);
  }
};

const runAutoplay = async (speed) => {
  const delay = speed === 'slow' ? 2000 : 500;
  
  while (autoplaySpeed === speed && !isTerminal) {
    await handleStep();
    await new Promise(resolve => setTimeout(resolve, delay));
  }
};

// Handle pause
const handlePause = () => {
  setIsPaused(true);
  setAutoplaySpeed(null);
};
```

### 7. Update CSS (App.css)

```css
/* W-value badges on pieces */
.cell {
  position: relative;
}

.w-value-badge {
  position: absolute;
  top: 2px;
  right: 2px;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  font-size: 10px;
  padding: 2px 4px;
  border-radius: 3px;
  font-weight: bold;
}

/* W-values on target cells */
.target-cell {
  border: 2px solid #4CAF50;
  background: rgba(76, 175, 80, 0.1);
}

.target-w-value {
  position: absolute;
  bottom: 2px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(76, 175, 80, 0.9);
  color: white;
  font-size: 12px;
  padding: 2px 6px;
  border-radius: 3px;
  font-weight: bold;
}

/* Game controls */
.game-controls {
  margin: 20px 0;
  padding: 15px;
  background: #f5f5f5;
  border-radius: 8px;
}

.control-buttons {
  display: flex;
  gap: 10px;
  margin: 10px 0;
}

.control-button {
  padding: 8px 16px;
  border: 2px solid #ddd;
  background: white;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.control-button:hover:not(:disabled) {
  background: #f0f0f0;
  border-color: #999;
}

.control-button.active {
  background: #4CAF50;
  color: white;
  border-color: #4CAF50;
}

.control-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.control-status {
  margin-top: 10px;
  font-size: 14px;
  color: #666;
}
```

## Implementation Steps

### Phase 1: Backend W-Values API
1. Add W-value calculation methods to `GameManager`
2. Add Pydantic models to `models.py`
3. Add API endpoint to `main.py`
4. Test API with curl/Postman

### Phase 2: Frontend W-Values Display
1. Add W-values API service to `api.js`
2. Update `App.jsx` state management for W-values
3. Update `Cell.jsx` to display W-values
4. Update `GameBoard.jsx` to pass W-values to cells
5. Add CSS styling for W-value badges

### Phase 3: Game Controls
1. Create `GameControls.jsx` component
2. Update `App.jsx` state management for game controls
3. Implement pause/step/autoplay logic
4. Remove speed control from `GameSetup.jsx`
5. Update game flow to start paused
6. Add CSS styling for game controls

### Phase 4: Testing & Refinement
1. Test with human vs afterstate agent
2. Test with afterstate vs afterstate
3. Test with afterstate vs naive/random
4. Verify W-values update correctly
5. Verify game controls work smoothly
6. Test edge cases (terminal states, history navigation)

## Edge Cases & Considerations

### W-Values Display
- **No afterstate agent**: Don't show W-values, API returns `has_afterstate_agent: false`
- **Opponent's turn**: Only show W-values on current player's pieces
- **Terminal state**: Don't fetch/display W-values
- **History navigation**: Don't show W-values when viewing history
- **Human player**: Show W-values even when human is playing (if opponent is afterstate)

### Game Controls
- **Human vs AI**: Step button advances AI move, human moves are manual
- **AI vs AI**: Step button advances one AI move at a time
- **Terminal state**: Disable all controls
- **History navigation**: Disable controls while viewing history
- **Autoplay**: Stop autoplay when terminal state reached

### Performance
- **API calls**: Fetch W-values only when needed (after state changes)
- **Debouncing**: Consider debouncing W-value fetches if updates are too frequent
- **Caching**: W-values are deterministic, could cache by board state hash

### UI/UX
- **Value formatting**: Show 2 decimal places (e.g., "0.85")
- **Color coding**: Consider adding color coding later (green=good, red=bad)
- **Tooltips**: Consider adding tooltips to explain W-values
- **Mobile**: Ensure W-value badges are readable on small screens

## Future Enhancements

1. **Color-coded W-values**: Green for high values, red for low values
2. **W-value heatmap**: Show gradient overlay on board
3. **Value comparison**: Highlight best move vs current selection
4. **Historical W-values**: Show W-values in history navigation
5. **Export W-values**: Download W-value data for analysis
6. **Multiple agents**: Compare W-values from different agents side-by-side
