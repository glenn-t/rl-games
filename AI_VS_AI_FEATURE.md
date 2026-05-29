# AI vs AI Mode - Feature Documentation

## Overview

Added AI vs AI game mode where two AI agents play against each other automatically, with configurable game speed.

## Features

### 1. AI vs AI Game Mode
- Select two different AI agents (one for each player)
- Agents can be Naive or Random
- Game plays automatically without human intervention
- Watch the AI agents compete in real-time

### 2. Game Speed Control
Four speed options available:
- **Slow**: 2 seconds per move (good for learning/analysis)
- **Normal**: 1 second per move (balanced viewing experience)
- **Fast**: 0.5 seconds per move (quick games)
- **Instant**: No delay (immediate completion)

### 3. User Interface
- Separate agent selection for Player X (Blue) and Player O (Red)
- Speed selector dropdown
- Board interaction disabled during AI vs AI games
- Real-time move descriptions displayed
- Automatic game completion detection

## Backend Changes

### Modified Files

**backend/models.py**
- Added `ai_vs_ai` to mode options
- Added `ai_agent_player0` and `ai_agent_player1` fields for AI vs AI mode

**backend/game_manager.py**
- Updated `GameSession` to store both AI agents
- Modified `create_game()` to handle AI vs AI mode
- Updated `make_ai_move()` to work with both modes
- Automatically selects correct AI agent based on current player

**backend/main.py**
- Added validation for AI vs AI mode requirements
- Updated game creation endpoint to accept both AI agents

## Frontend Changes

### Modified Files

**frontend/src/services/api.js**
- Updated `createGame()` to accept AI agents for both players
- Added parameters for `aiAgentPlayer0` and `aiAgentPlayer1`

**frontend/src/components/GameSetup.jsx**
- Added "AI vs AI" option to game mode selector
- Added separate agent selectors for each player
- Added game speed selector (Slow/Normal/Fast/Instant)
- Updated form handling to pass all parameters

**frontend/src/components/GameInfo.jsx**
- Updated to display both AI agents in AI vs AI mode
- Shows which agent controls each player

**frontend/src/App.jsx**
- Added state for AI vs AI mode (`aiAgentPlayer0`, `aiAgentPlayer1`, `gameSpeed`)
- Implemented `runAIvsAI()` function for automatic game loop
- Added `getSpeedDelay()` to convert speed setting to milliseconds
- Disabled board interaction during AI vs AI games
- Automatic game loop continues until game ends

## Usage

### Starting an AI vs AI Game

1. Select "AI vs AI" from game mode dropdown
2. Choose AI agent for Player X (Blue):
   - Naive Agent: Strategic one-step lookahead
   - Random Agent: Random move selection
3. Choose AI agent for Player O (Red):
   - Naive Agent or Random Agent
4. Select game speed:
   - Slow (2s), Normal (1s), Fast (0.5s), or Instant
5. Click "Start Game"
6. Watch the AI agents play automatically!

### Game Flow

```
1. User configures AI vs AI game
2. Frontend calls POST /api/game/new with both AI agents
3. Backend creates game with both agents configured
4. Frontend starts automatic game loop
5. Loop calls POST /api/game/{id}/ai-move repeatedly
6. Backend determines which AI agent's turn and executes move
7. Frontend updates board and waits for configured delay
8. Loop continues until game ends (win or draw)
9. Final result displayed
```

## Technical Details

### Speed Delays
```javascript
const delays = {
  'slow': 2000,    // 2 seconds
  'normal': 1000,  // 1 second
  'fast': 500,     // 0.5 seconds
  'instant': 0     // No delay
};
```

### AI Move Loop
```javascript
const runAIvsAI = async (gameId) => {
  // Make AI move
  const result = await makeAIMove(gameId);
  updateGameState(result);
  
  // Continue if not terminal
  if (!result.is_terminal) {
    setTimeout(() => runAIvsAI(gameId), getSpeedDelay());
  }
};
```

### Backend AI Selection
```python
# In make_ai_move()
if session.mode == "ai_vs_ai":
    if current_player == 0:
        agent = session.ai_agent_player0
    else:
        agent = session.ai_agent_player1
```

## Benefits

1. **Testing**: Quickly test AI agents against each other
2. **Analysis**: Study different agent strategies
3. **Entertainment**: Watch AI battles
4. **Benchmarking**: Compare agent performance
5. **Learning**: Observe game patterns and strategies

## Future Enhancements

- Pause/Resume functionality for AI vs AI games
- Step-by-step mode (manual advance)
- Game statistics (move count, time, patterns)
- Tournament mode (multiple games, win tracking)
- More AI agents (MCTS, Neural Network)
- Replay saved games
- Export game history

## Example Scenarios

### Naive vs Naive
Both agents use strategic lookahead - interesting tactical battles

### Naive vs Random
Strategic agent usually wins - good for testing agent strength

### Random vs Random
Unpredictable games - useful for exploring game tree

### Speed Comparisons
- **Instant**: Complete game in <1 second (good for batch testing)
- **Fast**: ~10-20 seconds per game (quick analysis)
- **Normal**: ~20-40 seconds per game (comfortable viewing)
- **Slow**: ~40-80 seconds per game (detailed observation)

## Testing

To test the AI vs AI feature:

1. Start backend: `cd backend && python main.py`
2. Start frontend: `cd frontend && npm run dev`
3. Open `http://localhost:5173`
4. Select "AI vs AI" mode
5. Choose agents and speed
6. Click "Start Game"
7. Observe automatic gameplay

## Notes

- Board interaction is disabled during AI vs AI games
- Game can be stopped by clicking "New Game"
- All victory conditions work in AI vs AI mode
- Move descriptions show which AI made each move
- Game speed can only be set at game start (not during play)