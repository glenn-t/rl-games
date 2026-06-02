/**
 * Cell component - represents a single cell on the game board
 */
import React from 'react';

const Cell = ({ value, row, col, isSelected, isSelectable, onClick, wValue = null, targetWValue = null }) => {
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

export default Cell;

// Made with Bob
