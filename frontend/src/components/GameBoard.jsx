/**
 * GameBoard component - displays the 4x4 game board
 */
import React from 'react';
import Cell from './Cell';

const GameBoard = ({ board, currentPlayer, selectedCell, onCellClick, isTerminal, wValues = null, selectedPieceWValues = null }) => {
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

  // Helper to get W-value for a piece
  const getPieceWValue = (row, col) => {
    if (!wValues || !wValues.has_afterstate_agent) return null;
    const piece = wValues.piece_values?.find(p => p.row === row && p.col === col);
    return piece ? piece.max_w_value : null;
  };
  
  // Helper to get target W-value for a cell
  const getTargetWValue = (row, col) => {
    if (!selectedPieceWValues || !selectedPieceWValues.action_values) return null;
    const action = selectedPieceWValues.action_values.find(
      a => a.target_row === row && a.target_col === col
    );
    return action ? action.w_value : null;
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
              wValue={getPieceWValue(rowIndex, colIndex)}
              targetWValue={getTargetWValue(rowIndex, colIndex)}
            />
          ))}
        </div>
      ))}
    </div>
  );
};

export default GameBoard;

// Made with Bob
