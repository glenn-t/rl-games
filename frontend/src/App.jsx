import React, { useState } from 'react';
import GameSetup from './components/GameSetup';
import GameBoard from './components/GameBoard';
import DirectionButtons from './components/DirectionButtons';
import GameInfo from './components/GameInfo';
import HistoryNav from './components/HistoryNav';
import { createGame, makeMove, makeAIMove } from './services/api';
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
  const [gameSpeed, setGameSpeed] = useState('normal');

  // History state
  const [history, setHistory] = useState([]);
  const [viewingStep, setViewingStep] = useState(null); // null = live

  // UI state
  const [selectedCell, setSelectedCell] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState('');
  const [aiVsAiRunning, setAiVsAiRunning] = useState(false);

  const getSpeedDelay = () => {
    const delays = { slow: 2000, normal: 1000, fast: 500, instant: 0 };
    return delays[gameSpeed] || 1000;
  };

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

  const handleStartGame = async (gameMode, selectedAiAgent, selectedAiPlayer, aiAgent0, aiAgent1, speed) => {
    try {
      setLoading(true);
      setError(null);
      setGameSpeed(speed || 'normal');
      setHistory([]);
      setViewingStep(null);
      const gameState = await createGame(gameMode, selectedAiAgent, selectedAiPlayer, aiAgent0, aiAgent1);
      setGameId(gameState.game_id);
      updateGameState(gameState, 'Game started');
      setMessage('Game started!');

      if (gameMode === 'ai_vs_ai') {
        setAiVsAiRunning(true);
        setTimeout(() => runAIvsAI(gameState.game_id), getSpeedDelay());
      } else if (gameMode === 'human_vs_ai' && selectedAiPlayer === 0) {
        setTimeout(() => handleAIMove(gameState.game_id), 500);
      }
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const runAIvsAI = async (gId) => {
    const gameIdToUse = gId || gameId;
    if (!aiVsAiRunning && gId === undefined) return;

    try {
      setLoading(true);
      const result = await makeAIMove(gameIdToUse);
      updateGameState(result, result.action_description);
      setMessage(`${result.action_description}`);
      setLoading(false);

      if (!result.is_terminal) {
        setTimeout(() => runAIvsAI(gameIdToUse), getSpeedDelay());
      } else {
        setAiVsAiRunning(false);
        setMessage(`Game Over! ${result.winner !== null ? `Player ${result.winner === 0 ? 'X' : 'O'} wins!` : 'Draw!'}`);
      }
    } catch (err) {
      console.error('AI vs AI error:', err);
      setError(`AI Error: ${err.message}. Check that backend is running on port 8000.`);
      setMessage(`AI Error: ${err.message}`);
      setAiVsAiRunning(false);
      setLoading(false);
    }
  };

  const handleCellClick = (row, col) => {
    if (isTerminal || loading) return;
    if (mode === 'ai_vs_ai') { setMessage('AI vs AI mode - watch the game play out!'); return; }
    if (mode === 'human_vs_ai' && currentPlayer === aiPlayer) { setMessage("It's the AI's turn!"); return; }

    const cellValue = board[row][col];
    const playerValue = currentPlayer === 0 ? 1 : 2;

    if (cellValue === playerValue) {
      setSelectedCell({ row, col });
      setMessage(`Selected piece at (${row}, ${col}). Choose a direction.`);
    } else {
      setMessage('You can only select your own pieces!');
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

      if (!result.is_terminal && mode === 'human_vs_ai' && result.current_player === aiPlayer) {
        setTimeout(() => handleAIMove(gameId), 1000);
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
