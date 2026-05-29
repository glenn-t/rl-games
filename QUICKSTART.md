# Quick Start Guide - Dao Game

Follow these steps to get the Dao game running on your local machine.

## Step 1: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

Or if you're using the project's uv:
```bash
cd backend
uv pip install -r requirements.txt
```

## Step 2: Start the Backend Server

```bash
# From the backend directory
python main.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Keep this terminal open!**

Test the backend by visiting: `http://localhost:8000/docs`

## Step 3: Install Frontend Dependencies

Open a **new terminal** and run:

```bash
cd frontend
npm install
```

This will take a minute to download all dependencies.

## Step 4: Start the Frontend Server

```bash
# From the frontend directory
npm run dev
```

You should see:
```
  VITE v5.3.1  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

## Step 5: Play the Game!

1. Open your browser to `http://localhost:5173`
2. You should see the game setup screen
3. Choose your game mode:
   - **Human vs Human**: Play against another person
   - **Human vs AI**: Play against the computer

### Playing Human vs Human

1. Select "Human vs Human"
2. Click "Start Game"
3. Player X (Blue) goes first
4. Click one of your pieces (on the diagonal)
5. Click a direction button (↑ ↓ ← → ↖ ↗ ↙ ↘)
6. The piece slides until it hits a wall or another piece
7. Player O (Red) takes their turn
8. Continue until someone wins!

### Playing Human vs AI

1. Select "Human vs AI"
2. Choose AI agent:
   - **Naive Agent**: Strategic, looks one move ahead
   - **Random Agent**: Makes random moves
3. Choose which player the AI controls (default: Player O)
4. Click "Start Game"
5. If AI is Player X, it will move first automatically
6. Make your move by clicking your piece and selecting a direction
7. AI will automatically make its move after yours
8. Continue until someone wins!

## Victory Conditions

You win by achieving any of these:

1. **Line**: Fill an entire row or column
2. **Square**: Form a 2×2 square
3. **Corners**: Occupy all four corners
4. **Corner Block**: Trap opponent in a corner

## Troubleshooting

### Backend won't start

**Error: "ModuleNotFoundError: No module named 'fastapi'"**
- Solution: Install dependencies with `pip install -r requirements.txt`

**Error: "Address already in use"**
- Solution: Another process is using port 8000. Kill it or use a different port:
  ```bash
  uvicorn main:app --port 8001
  ```
  Then update `frontend/src/services/api.js` to use port 8001

### Frontend won't start

**Error: "command not found: npm"**
- Solution: Install Node.js from https://nodejs.org/

**Error: "Cannot find module"**
- Solution: Delete node_modules and reinstall:
  ```bash
  rm -rf node_modules package-lock.json
  npm install
  ```

### Game won't load

**Blank screen or "Failed to fetch"**
- Check that backend is running on port 8000
- Check browser console (F12) for errors
- Verify CORS is enabled in backend

**AI moves not working**
- Check backend terminal for errors
- Verify the game mode is set to "human_vs_ai"
- Check that the AI agent was selected during setup

## Testing the API Directly

You can test the backend API using the interactive docs:

1. Go to `http://localhost:8000/docs`
2. Try the "POST /api/game/new" endpoint
3. Click "Try it out"
4. Enter:
   ```json
   {
     "mode": "human_vs_human"
   }
   ```
5. Click "Execute"
6. You should get a game_id and initial board state

## Next Steps

- Read `README_DAO_GAME.md` for full documentation
- Check `DAO_FRONTEND_PLAN.md` for implementation details
- Explore the code in `backend/` and `frontend/src/`
- Try modifying the AI agents in `agents/naive.py`

## Need Help?

- Backend API docs: `http://localhost:8000/docs`
- Backend README: `backend/README.md`
- Frontend README: `frontend/README.md`
- Main README: `README_DAO_GAME.md`

Enjoy playing Dao! 🎮