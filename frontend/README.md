# Dao Game Frontend

React frontend for the Dao board game, built with Vite.

## Setup

1. **Install dependencies:**

```bash
cd frontend
npm install
```

2. **Start the development server:**

```bash
npm run dev
```

The frontend will start on `http://localhost:5173`

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Backend server running on `http://localhost:8000`

## Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Preview Production Build

```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Cell.jsx              # Individual board cell
│   │   ├── GameBoard.jsx         # 4x4 game board
│   │   ├── DirectionButtons.jsx  # 8-directional controls
│   │   ├── GameInfo.jsx          # Game status display
│   │   └── GameSetup.jsx         # Game mode selection
│   ├── services/
│   │   └── api.js                # Backend API calls
│   ├── App.jsx                   # Main application
│   ├── App.css                   # Styling
│   └── main.jsx                  # Entry point
├── public/                       # Static assets
├── index.html                    # HTML template
├── vite.config.js                # Vite configuration
└── package.json                  # Dependencies
```

## Features

### Game Modes
- **Human vs Human**: Two players take turns on the same device
- **Human vs AI**: Play against an AI agent (Naive or Random)

### Gameplay
1. Select a piece by clicking on it (only your pieces can be selected)
2. Choose a direction using the 8-directional buttons
3. The piece slides in that direction until it hits a wall or another piece
4. Win by achieving one of four victory conditions

### Victory Conditions
- **Line**: Complete any row or column with your pieces
- **Square**: Form a 2×2 square with your pieces
- **Corners**: Occupy all four corner cells
- **Corner Block**: Trap opponent's piece in a corner

### UI Features
- Visual piece selection with highlighting
- Color-coded players (Blue X for Player 0, Red O for Player 1)
- Real-time game status updates
- AI move automation
- Responsive design for mobile devices

## Technologies

- **React 18**: UI framework
- **Vite**: Build tool and dev server
- **CSS3**: Styling with gradients and animations
- **Fetch API**: Backend communication

## Development

The frontend communicates with the FastAPI backend via REST API calls. Make sure the backend is running before starting the frontend.

### API Endpoints Used
- `POST /api/game/new` - Create new game
- `GET /api/game/{id}` - Get game state
- `POST /api/game/{id}/move` - Make a move
- `POST /api/game/{id}/ai-move` - Make AI move
- `GET /api/agents` - Get available AI agents

## Troubleshooting

### Backend Connection Issues
- Ensure backend is running on `http://localhost:8000`
- Check CORS settings in backend allow `http://localhost:5173`
- Verify no firewall blocking the connection

### Build Issues
- Delete `node_modules` and run `npm install` again
- Clear Vite cache: `rm -rf node_modules/.vite`
- Check Node.js version: `node --version` (should be v16+)

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)