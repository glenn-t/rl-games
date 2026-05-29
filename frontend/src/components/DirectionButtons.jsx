/**
 * DirectionButtons component - displays 8 directional buttons for moving a selected piece
 */
import React from 'react';

const DIRECTIONS = [
  { name: 'UP LEFT', symbol: '↖', key: 'UP LEFT' },
  { name: 'UP', symbol: '↑', key: 'UP' },
  { name: 'UP RIGHT', symbol: '↗', key: 'UP RIGHT' },
  { name: 'LEFT', symbol: '←', key: 'LEFT' },
  { name: '', symbol: '', key: 'center' }, // Center placeholder
  { name: 'RIGHT', symbol: '→', key: 'RIGHT' },
  { name: 'DOWN LEFT', symbol: '↙', key: 'DOWN LEFT' },
  { name: 'DOWN', symbol: '↓', key: 'DOWN' },
  { name: 'DOWN RIGHT', symbol: '↘', key: 'DOWN RIGHT' },
];

const DirectionButtons = ({ selectedCell, legalActions, onDirectionClick, disabled }) => {
  if (!selectedCell) return null;
  
  // Helper to check if a direction is legal
  const isDirectionLegal = (direction) => {
    if (!legalActions || legalActions.length === 0) return false;
    // We'll need to match actions to directions based on the selected cell
    // For now, we'll enable all directions and let the backend validate
    return true;
  };
  
  return (
    <div className="direction-buttons-container">
      <h3>Select Direction</h3>
      <p className="selected-cell-info">
        Selected: ({selectedCell.row}, {selectedCell.col})
      </p>
      <div className="direction-buttons">
        {DIRECTIONS.map((dir, index) => {
          if (dir.key === 'center') {
            return <div key={index} className="direction-button center-placeholder"></div>;
          }
          
          const isLegal = isDirectionLegal(dir.key);
          
          return (
            <button
              key={index}
              className={`direction-button ${!isLegal || disabled ? 'disabled' : ''}`}
              onClick={() => onDirectionClick(dir.key)}
              disabled={!isLegal || disabled}
              title={dir.name}
            >
              <span className="direction-symbol">{dir.symbol}</span>
              <span className="direction-name">{dir.name}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default DirectionButtons;

// Made with Bob
