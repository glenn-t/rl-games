/**
 * GameControls component - displays game control buttons (pause, step, autoplay)
 */
import React from 'react';

const GameControls = ({ 
  isPaused, 
  autoplaySpeed,
  onPause,
  onStep,
  onAutoplay,
  disabled = false,
  mode = null
}) => {
  // Only show controls for AI modes
  if (!mode || mode === 'human_vs_human') {
    return null;
  }

  return (
    <div className="game-controls">
      <h3>Game Controls</h3>
      <div className="control-buttons">
        <button
          className={`control-button ${isPaused ? 'active' : ''}`}
          onClick={onPause}
          disabled={disabled}
          title="Pause the game"
        >
          ⏸ Pause
        </button>
        <button
          className="control-button"
          onClick={onStep}
          disabled={disabled || !isPaused}
          title="Advance one move"
        >
          ⏭ Step
        </button>
        <button
          className={`control-button ${autoplaySpeed === 'slow' ? 'active' : ''}`}
          onClick={() => onAutoplay('slow')}
          disabled={disabled}
          title="Autoplay at slow speed (2s per move)"
        >
          ▶ Slow
        </button>
        <button
          className={`control-button ${autoplaySpeed === 'fast' ? 'active' : ''}`}
          onClick={() => onAutoplay('fast')}
          disabled={disabled}
          title="Autoplay at fast speed (0.5s per move)"
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

export default GameControls;

// Made with Bob
