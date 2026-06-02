import React, { useState } from 'react';
import GameSetup from './components/GameSetup';
import GameBoard from './components/GameBoard';
import DirectionButtons from './components/DirectionButtons';
import GameInfo from './components/GameInfo';
import HistoryNav from './components/HistoryNav';
import GameControls from './components/GameControls';
import { createGame, makeMove, makeAIMove, getWValues } from './services/api';
import './App.css';

function App() {
  // Game state
  const [gameId, setGameId] = useState(null);
  const [board, setBoard] = useState(null);
  const [currentPlayer, setCurrentPlayer] = useState(0);
  const [isTerminal, setIsTerminal] = useState(false);
  const [winner, setWinner] = useState(null);
  const [legalActions, setLegalActions] = useState([]);
  const [mode, setMode] = useState(null);
  const [aiAgent, setAiAgent] = useState(null);
  const [aiPlayer, setAiPlayer] = useState(null);
  const [aiAgentPlayer0, setAiAgentPlayer0] = useState(null);
  const [aiAgentPlayer1, setAiAgentPlayer1] = useState(null);

  // History state
  const [history, setHistory] = useState([]);
  const [viewingStep, setViewingStep] = useState(null); // null = live

  // UI state
  const [selectedCell, setSelectedCell] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState('');

  // W-values state
  const [wValues, setWValues] = useState(null);
  const [selectedPieceWValues, setSelectedPieceWValues] = useState(null);

  // Game control state
  const [isPaused, setIsPaused] = useState(true);
  const [autoplaySpeed, setAutoplaySpeed] = useState(null); // null, 'slow', 'fast'
  const [showWValues, setShowWValues] = useState(false); // Toggle for W-values display

  const updateGameState = (gameState, description = '') => {
    setBoard(gameState.board);
    setCurrentPlayer(gameState.current_player);
    setIsTerminal(gameState.is_terminal);
    setWinner(gameState.winner);
    setLegalActions(gameState.legal_actions);
    if (gameState.mode) setMode(gameState.mode);
    if (gameState.ai_agent) setAiAgent(gameState.ai_agent);
    if (gameState.ai_player !== undefined) setAiPlayer(gameState.ai_player);
    if (gameState.ai_agent_player0) setAiAgentPlayer0(gameState.ai_agent_player0);
    if (gameState.ai_agent_player1) setAiAgentPlayer1(gameState.ai_agent_player1);
    setHistory(prev => [...prev, { board: gameState.board, currentPlayer: gameState.current_player, description }]);
  };

  // Fetch W-values for all pieces
  const fetchWValues = async (gId) => {
    if (!gId) return;
    try {
      const values = await getWValues(gId);
      setWValues(values);
    } catch (err) {
      console.error('Failed to fetch W-values:', err);
      setWValues(null);
    }
  };

  // Fetch W-values for selected piece
  const fetchSelectedPieceWValues = async (gId, row, col) => {
    if (!gId) return;
    try {
      const values = await getWValues(gId, row, col);
      console.log('Selected piece W-values:', values);
      setSelectedPieceWValues(values);
    } catch (err) {
      console.error('Failed to fetch selected piece W-values:', err);
      setSelectedPieceWValues(null);
    }
  };

  // Fetch W-values when game state changes (only if showWValues is enabled)
  React.useEffect(() => {
    if (showWValues && gameId && !isTerminal && !viewingStep) {
      fetchWValues(gameId);
    } else if (!showWValues) {
      setWValues(null);
    }
  }, [showWValues, gameId, board, currentPlayer, isTerminal, viewingStep]);

  // Fetch selected piece W-values when piece is selected (only if showWValues is enabled)
  React.useEffect(() => {
    if (showWValues && gameId && selectedCell && !isTerminal) {
      fetchSelectedPieceWValues(gameId, selectedCell.row, selectedCell.col);
    } else {
      setSelectedPieceWValues(null);
    }
  }, [showWValues, gameId, selectedCell, isTerminal]);

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
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const handleStep = async () => {
    if (isTerminal || loading) return;
    
    // In human vs AI mode, only advance if it's AI's turn
    if (mode === 'human_vs_ai' && currentPlayer !== aiPlayer) {
      setMessage("It's your turn! Select a piece and direction.");
      return;
    }
    
    try {
      setLoading(true);
      const result = await makeAIMove(gameId);
      updateGameState(result, result.action_description);
      setMessage(result.action_description);
      
      if (result.is_terminal) {
        setIsPaused(true);
        setAutoplaySpeed(null);
        setMessage(`Game Over! ${result.winner !== null ? `Player ${result.winner === 0 ? 'X' : 'O'} wins!` : 'Draw!'}`);
      } else if (mode === 'human_vs_ai' && result.current_player !== aiPlayer) {
        setMessage("It's your turn! Select a piece and direction.");
      } else {
        setMessage("Click Step to advance, or use autoplay.");
      }
    } catch (err) {
      setError(err.message);
      setMessage(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleAutoplay = (speed) => {
    if (speed === autoplaySpeed) {
      // Toggle off
      setAutoplaySpeed(null);
      setIsPaused(true);
    } else {
      setAutoplaySpeed(speed);
      setIsPaused(false);
    }
  };

  // Effect to run autoplay when autoplaySpeed changes or when it's AI's turn
  React.useEffect(() => {
    if (!autoplaySpeed || isTerminal) return;
    
    // In human vs AI mode, only run if it's AI's turn
    if (mode === 'human_vs_ai' && currentPlayer !== aiPlayer) {
      return;
    }
    
    const delay = autoplaySpeed === 'slow' ? 2000 : 500;
    const timeoutId = setTimeout(async () => {
      await handleStep();
    }, delay);
    
    return () => {
      clearTimeout(timeoutId);
    };
  }, [autoplaySpeed, isTerminal, currentPlayer, mode, aiPlayer, board]);

  const handlePause = () => {
    setIsPaused(true);
    setAutoplaySpeed(null);
  };

  const handleCellClick = (row, col) => {
    if (isTerminal || loading) return;
    
    const cellValue = board[row][col];
    const playerValue = currentPlayer === 0 ? 1 : 2;

    if (cellValue === playerValue) {
      setSelectedCell({ row, col });
      
      // Different messages based on game mode and whose turn it is
      if (mode === 'ai_vs_ai') {
        setMessage(`Viewing AI piece at (${row}, ${col}). Click Step to advance.`);
      } else if (mode === 'human_vs_ai' && currentPlayer === aiPlayer) {
        setMessage(`Viewing AI piece at (${row}, ${col}). Click Step to see AI move.`);
      } else {
        setMessage(`Selected piece at (${row}, ${col}). Choose a direction.`);
      }
    } else {
      setMessage('You can only select pieces belonging to the current player!');
    }
  };

  const calculateActionId = (row, col, direction) => {
    const directionOrder = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'UP LEFT', 'UP RIGHT', 'DOWN LEFT', 'DOWN RIGHT'];
    return (row * 4 + col) * 8 + directionOrder.indexOf(direction);
  };

  const handleDirectionClick = async (direction) => {
    if (!selectedCell || loading || isTerminal) return;

    try {
      setLoading(true);
      setError(null);
      const actionId = calculateActionId(selectedCell.row, selectedCell.col, direction);
      const result = await makeMove(gameId, actionId, currentPlayer);
      updateGameState(result, result.action_description);
      setSelectedCell(null);
      setMessage(result.action_description);
      
      // In human vs AI mode, after human moves, autoplay continues if it was active
      if (!result.is_terminal && mode === 'human_vs_ai' && result.current_player === aiPlayer) {
        if (autoplaySpeed) {
          setMessage(`AI's turn - autoplay continuing (${autoplaySpeed})`);
        } else {
          setMessage("AI's turn - click Step to advance or use autoplay");
        }
      }
    } catch (err) {
      setError(err.message);
      setMessage(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleAIMove = async (gId) => {
    const gameIdToUse = gId || gameId;
    try {
      setLoading(true);
      setMessage('AI is thinking...');
      const result = await makeAIMove(gameIdToUse);
      updateGameState(result, result.action_description);
      setMessage(`AI: ${result.action_description}`);
    } catch (err) {
      setError(err.message);
      setMessage(`AI Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleHistoryNavigate = (step) => {
    setViewingStep(step);
    setSelectedCell(null);
  };

  const handleNewGame = () => {
    setGameId(null);
    setBoard(null);
    setCurrentPlayer(0);
    setIsTerminal(false);
    setWinner(null);
    setLegalActions([]);
    setMode(null);
    setAiAgent(null);
    setAiPlayer(null);
    setSelectedCell(null);
    setError(null);
    setMessage('');
    setHistory([]);
    setViewingStep(null);
  };

  if (!gameId) {
    return (
      <div className="app">
        <GameSetup onStartGame={handleStartGame} />
      </div>
    );
  }

  const isViewingHistory = viewingStep !== null;
  const displayBoard = isViewingHistory ? history[viewingStep]?.board : board;

  return (
    <div className="app">
      <div className="game-container">
        <div className="game-header">
          <h1>Dao Game</h1>
          <button className="new-game-button" onClick={handleNewGame}>New Game</button>
        </div>

        {error && <div className="error-banner">{error}</div>}
        {message && <div className="message-banner">{message}</div>}

        <div className="game-layout">
          <div className="game-left">
            <GameInfo
              currentPlayer={currentPlayer}
              isTerminal={isTerminal}
              winner={winner}
              mode={mode}
              aiAgent={aiAgent}
              aiAgentPlayer0={aiAgentPlayer0}
              aiAgentPlayer1={aiAgentPlayer1}
            />
          </div>

          <div className="game-center">
            <GameBoard
              board={displayBoard}
              currentPlayer={currentPlayer}
              selectedCell={isViewingHistory ? null : selectedCell}
              onCellClick={isViewingHistory ? () => {} : handleCellClick}
              isTerminal={isTerminal || isViewingHistory}
              wValues={isViewingHistory ? null : wValues}
              selectedPieceWValues={isViewingHistory ? null : selectedPieceWValues}
            />

            <div className="game-controls-panel">
              {!isViewingHistory && selectedCell && !isTerminal && (
                <DirectionButtons
                  selectedCell={selectedCell}
                  legalActions={legalActions}
                  onDirectionClick={handleDirectionClick}
                  disabled={loading}
                />
              )}

              <GameControls
                isPaused={isPaused}
                autoplaySpeed={autoplaySpeed}
                onPause={handlePause}
                onStep={handleStep}
                onAutoplay={handleAutoplay}
                disabled={loading || isTerminal}
                mode={mode}
                showWValues={showWValues}
                onToggleWValues={setShowWValues}
              />

              <HistoryNav
                history={history}
                viewingStep={viewingStep}
                onNavigate={handleHistoryNavigate}
              />
            </div>
          </div>
        </div>

        {loading && (
          <div className="loading-overlay">
            <div className="spinner"></div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

// Made with Bob
