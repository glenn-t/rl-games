/**
 * Cell component - represents a single cell on the game board
 */
import React from 'react';

const Cell = ({ value, row, col, isSelected, isSelectable, onClick }) => {
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
    </div>
  );
};

export default Cell;

// Made with Bob
