/**
 * GameInfo component - displays game status, current player, and winner
 */
import React from 'react';

const GameInfo = ({ currentPlayer, isTerminal, winner, mode, aiAgent, aiAgentPlayer0, aiAgentPlayer1 }) => {
  const getPlayerName = (player) => {
    if (player === 0) return 'Player X (Blue)';
    if (player === 1) return 'Player O (Red)';
    return 'Unknown';
  };
  
  const getGameStatus = () => {
    if (isTerminal) {
      if (winner !== null && winner !== undefined) {
        return (
          <div className="game-status winner">
            <h2>🎉 {getPlayerName(winner)} Wins! 🎉</h2>
          </div>
        );
      } else {
        return (
          <div className="game-status draw">
            <h2>Game Over - Draw</h2>
          </div>
        );
      }
    }
    
    return (
      <div className="game-status playing">
        <h2>Current Turn: {getPlayerName(currentPlayer)}</h2>
      </div>
    );
  };
  
  const getModeName = () => {
    if (mode === 'human_vs_human') return 'Human vs Human';
    if (mode === 'human_vs_ai') return 'Human vs AI';
    if (mode === 'ai_vs_ai') return 'AI vs AI';
    return mode;
  };
  
  return (
    <div className="game-info">
      {getGameStatus()}
      
      <div className="game-mode-info">
        <p>
          <strong>Mode:</strong> {getModeName()}
        </p>
        {mode === 'human_vs_ai' && aiAgent && (
          <p>
            <strong>AI Agent:</strong> {aiAgent}
          </p>
        )}
        {mode === 'ai_vs_ai' && (
          <>
            <p>
              <strong>Player X AI:</strong> {aiAgentPlayer0 || 'N/A'}
            </p>
            <p>
              <strong>Player O AI:</strong> {aiAgentPlayer1 || 'N/A'}
            </p>
          </>
        )}
      </div>
      
      <div className="victory-conditions">
        <h3>Victory Conditions:</h3>
        <ul>
          <li><strong>Line:</strong> Complete any row or column</li>
          <li><strong>Square:</strong> Form a 2×2 square</li>
          <li><strong>Corners:</strong> Occupy all four corners</li>
          <li><strong>Corner Block:</strong> Trap opponent in a corner</li>
        </ul>
      </div>
    </div>
  );
};

export default GameInfo;

// Made with Bob
