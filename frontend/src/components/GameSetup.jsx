/**
 * GameSetup component - allows user to configure and start a new game
 */
import React, { useState, useEffect } from 'react';
import { getAgents } from '../services/api';

const GameSetup = ({ onStartGame }) => {
  const [mode, setMode] = useState('human_vs_human');
  const [aiAgent, setAiAgent] = useState('naive');
  const [aiPlayer, setAiPlayer] = useState(1);
  const [aiAgentPlayer0, setAiAgentPlayer0] = useState('naive');
  const [aiAgentPlayer1, setAiAgentPlayer1] = useState('naive');
  const [gameSpeed, setGameSpeed] = useState('normal');
  const [agents, setAgents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    // Fetch available agents
    const fetchAgents = async () => {
      try {
        const data = await getAgents();
        setAgents(data.agents);
        if (data.agents.length > 0) {
          setAiAgent(data.agents[0].id);
          setAiAgentPlayer0(data.agents[0].id);
          setAiAgentPlayer1(data.agents[0].id);
        }
      } catch (err) {
        console.error('Failed to fetch agents:', err);
        setError('Failed to load AI agents');
      }
    };
    
    fetchAgents();
  }, []);
  
  const handleStartGame = async () => {
    setLoading(true);
    setError(null);
    
    try {
      if (mode === 'human_vs_ai') {
        await onStartGame(mode, aiAgent, aiPlayer, null, null, gameSpeed);
      } else if (mode === 'ai_vs_ai') {
        await onStartGame(mode, null, null, aiAgentPlayer0, aiAgentPlayer1, gameSpeed);
      } else {
        await onStartGame(mode, null, null, null, null, gameSpeed);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="game-setup">
      <h1>Dao Game</h1>
      <p className="game-description">
        A strategic board game where pieces slide until blocked. 
        Win by completing a line, forming a square, occupying all corners, or trapping your opponent.
      </p>
      
      <div className="setup-form">
        <div className="form-group">
          <label htmlFor="mode">Game Mode:</label>
          <select
            id="mode"
            value={mode}
            onChange={(e) => setMode(e.target.value)}
            disabled={loading}
          >
            <option value="human_vs_human">Human vs Human</option>
            <option value="human_vs_ai">Human vs AI</option>
            <option value="ai_vs_ai">AI vs AI</option>
          </select>
        </div>
        
        {mode === 'human_vs_ai' && (
          <>
            <div className="form-group">
              <label htmlFor="ai-agent">AI Agent:</label>
              <select
                id="ai-agent"
                value={aiAgent}
                onChange={(e) => setAiAgent(e.target.value)}
                disabled={loading}
              >
                {agents.map((agent) => (
                  <option key={agent.id} value={agent.id}>
                    {agent.name}
                  </option>
                ))}
              </select>
              {agents.find(a => a.id === aiAgent) && (
                <p className="agent-description">
                  {agents.find(a => a.id === aiAgent).description}
                </p>
              )}
            </div>
            
            <div className="form-group">
              <label htmlFor="ai-player">AI Plays As:</label>
              <select
                id="ai-player"
                value={aiPlayer}
                onChange={(e) => setAiPlayer(Number(e.target.value))}
                disabled={loading}
              >
                <option value={0}>Player X (Blue) - Goes First</option>
                <option value={1}>Player O (Red) - Goes Second</option>
              </select>
            </div>
          </>
        )}
        
        {mode === 'ai_vs_ai' && (
          <>
            <div className="form-group">
              <label htmlFor="ai-agent-player0">Player X (Blue) AI:</label>
              <select
                id="ai-agent-player0"
                value={aiAgentPlayer0}
                onChange={(e) => setAiAgentPlayer0(e.target.value)}
                disabled={loading}
              >
                {agents.map((agent) => (
                  <option key={agent.id} value={agent.id}>
                    {agent.name}
                  </option>
                ))}
              </select>
              {agents.find(a => a.id === aiAgentPlayer0) && (
                <p className="agent-description">
                  {agents.find(a => a.id === aiAgentPlayer0).description}
                </p>
              )}
            </div>
            
            <div className="form-group">
              <label htmlFor="ai-agent-player1">Player O (Red) AI:</label>
              <select
                id="ai-agent-player1"
                value={aiAgentPlayer1}
                onChange={(e) => setAiAgentPlayer1(e.target.value)}
                disabled={loading}
              >
                {agents.map((agent) => (
                  <option key={agent.id} value={agent.id}>
                    {agent.name}
                  </option>
                ))}
              </select>
              {agents.find(a => a.id === aiAgentPlayer1) && (
                <p className="agent-description">
                  {agents.find(a => a.id === aiAgentPlayer1).description}
                </p>
              )}
            </div>
            
            <div className="form-group">
              <label htmlFor="game-speed">Game Speed:</label>
              <select
                id="game-speed"
                value={gameSpeed}
                onChange={(e) => setGameSpeed(e.target.value)}
                disabled={loading}
              >
                <option value="slow">Slow (2 seconds per move)</option>
                <option value="normal">Normal (1 second per move)</option>
                <option value="fast">Fast (0.5 seconds per move)</option>
                <option value="instant">Instant (no delay)</option>
              </select>
            </div>
          </>
        )}
        
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}
        
        <button
          className="start-button"
          onClick={handleStartGame}
          disabled={loading}
        >
          {loading ? 'Starting...' : 'Start Game'}
        </button>
      </div>
    </div>
  );
};

export default GameSetup;

// Made with Bob
