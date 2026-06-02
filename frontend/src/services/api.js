/**
 * API service for communicating with the Dao game backend
 */

const API_BASE = 'http://localhost:8000/api';

/**
 * Create a new game
 * @param {string} mode - "human_vs_human", "human_vs_ai", or "ai_vs_ai"
 * @param {string} aiAgent - "naive" or "random" (optional, for human_vs_ai)
 * @param {number} aiPlayer - 0 or 1 (optional, default 1, for human_vs_ai)
 * @param {string} aiAgentPlayer0 - "naive" or "random" (optional, for ai_vs_ai)
 * @param {string} aiAgentPlayer1 - "naive" or "random" (optional, for ai_vs_ai)
 * @returns {Promise<Object>} Game state
 */
export async function createGame(mode, aiAgent = null, aiPlayer = 1, aiAgentPlayer0 = null, aiAgentPlayer1 = null) {
  const body = { mode };
  
  if (mode === 'human_vs_ai') {
    body.ai_agent = aiAgent;
    body.ai_player = aiPlayer;
  } else if (mode === 'ai_vs_ai') {
    body.ai_agent_player0 = aiAgentPlayer0;
    body.ai_agent_player1 = aiAgentPlayer1;
  }
  
  const response = await fetch(`${API_BASE}/game/new`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to create game');
  }
  
  return response.json();
}

/**
 * Get current game state
 * @param {string} gameId - Game ID
 * @returns {Promise<Object>} Game state
 */
export async function getGameState(gameId) {
  const response = await fetch(`${API_BASE}/game/${gameId}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get game state');
  }
  
  return response.json();
}

/**
 * Make a move
 * @param {string} gameId - Game ID
 * @param {number} action - Action ID
 * @param {number} player - Player ID (0 or 1)
 * @returns {Promise<Object>} Move result
 */
export async function makeMove(gameId, action, player) {
  const response = await fetch(`${API_BASE}/game/${gameId}/move`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ action, player }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to make move');
  }
  
  return response.json();
}

/**
 * Make an AI move
 * @param {string} gameId - Game ID
 * @returns {Promise<Object>} AI move result
 */
export async function makeAIMove(gameId) {
  const response = await fetch(`${API_BASE}/game/${gameId}/ai-move`, {
    method: 'POST',
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to make AI move');
  }
  
  return response.json();
}

/**
 * Get available AI agents
 * @returns {Promise<Object>} List of agents
 */
export async function getAgents() {
  const response = await fetch(`${API_BASE}/agents`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get agents');
  }
  
  return response.json();
}

/**
 * Get action details
 * @param {string} gameId - Game ID
 * @param {number} actionId - Action ID
 * @returns {Promise<Object>} Action details
 */
export async function getActionDetails(gameId, actionId) {
  const response = await fetch(`${API_BASE}/game/${gameId}/action/${actionId}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get action details');
  }
  
  return response.json();
}

/**
 * Delete a game
 * @param {string} gameId - Game ID
 * @returns {Promise<Object>} Deletion result
 */
export async function deleteGame(gameId) {
  const response = await fetch(`${API_BASE}/game/${gameId}`, {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to delete game');
  }
  
  return response.json();
}

// Made with Bob


/**
 * Get W-values for afterstate agent
 * @param {string} gameId - Game ID
 * @param {number} pieceRow - Optional row of selected piece
 * @param {number} pieceCol - Optional column of selected piece
 * @returns {Promise<Object>} W-values data
 */
export async function getWValues(gameId, pieceRow = null, pieceCol = null) {
  const params = new URLSearchParams();
  if (pieceRow !== null) params.append('piece_row', pieceRow);
  if (pieceCol !== null) params.append('piece_col', pieceCol);
  
  const url = `${API_BASE}/game/${gameId}/w-values${params.toString() ? '?' + params.toString() : ''}`;
  const response = await fetch(url);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get W-values');
  }
  
  return response.json();
}

