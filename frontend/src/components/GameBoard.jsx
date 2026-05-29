/**
 * GameBoard component - displays the 4x4 game board
 */
import React from 'react';
import Cell from './Cell';

const GameBoard = ({ board, currentPlayer, selectedCell, onCellClick, isTerminal }) => {
  if (!board) {
    return <div className="game-board loading">Loading board...</div>;
  }
  
  // Check if a cell is selectable (belongs to current player and game is not over)
  const isCellSelectable = (row, col) => {
    if (isTerminal) return false;
    const cellValue = board[row][col];
    // Player 0 has value 1, Player 1 has value 2
    const playerValue = currentPlayer === 0 ? 1 : 2;
    return cellValue === playerValue;
  };
  
  // Check if a cell is currently selected
  const isCellSelected = (row, col) => {
    return selectedCell && selectedCell.row === row && selectedCell.col === col;
  };
  
  return (
    <div className="game-board">
      {board.map((row, rowIndex) => (
        <div key={rowIndex} className="board-row">
          {row.map((cell, colIndex) => (
            <Cell
              key={`${rowIndex}-${colIndex}`}
              value={cell}
              row={rowIndex}
              col={colIndex}
              isSelected={isCellSelected(rowIndex, colIndex)}
              isSelectable={isCellSelectable(rowIndex, colIndex)}
              onClick={onCellClick}
            />
          ))}
        </div>
      ))}
    </div>
  );
};

export default GameBoard;

// Made with Bob
