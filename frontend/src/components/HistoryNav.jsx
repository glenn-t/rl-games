import React from 'react';

const HistoryNav = ({ history, viewingStep, onNavigate }) => {
  if (!history || history.length <= 1) return null;

  const currentStep = viewingStep !== null ? viewingStep : history.length - 1;
  const isLive = viewingStep === null;

  return (
    <div className="history-panel">
      <h3>Game History</h3>
      <div className="history-step-info">
        <span>Step {currentStep + 1} of {history.length}</span>
        {isLive && <span className="live-badge">LIVE</span>}
      </div>
      {!isLive && history[viewingStep] && (
        <p className="history-description">{history[viewingStep].description}</p>
      )}
      <div className="history-nav">
        <button
          className="history-btn"
          onClick={() => onNavigate(0)}
          disabled={currentStep === 0}
          title="First step"
        >⏮</button>
        <button
          className="history-btn"
          onClick={() => onNavigate(currentStep - 1)}
          disabled={currentStep === 0}
          title="Previous step"
        >◀</button>
        <button
          className="history-btn"
          onClick={() => {
            const next = viewingStep + 1;
            onNavigate(next >= history.length ? null : next);
          }}
          disabled={isLive}
          title="Next step"
        >▶</button>
        <button
          className="history-btn"
          onClick={() => onNavigate(null)}
          disabled={isLive}
          title="Latest step"
        >⏭</button>
      </div>
      {!isLive && (
        <button className="return-live-btn" onClick={() => onNavigate(null)}>
          Return to Live
        </button>
      )}
    </div>
  );
};

export default HistoryNav;

// Made with Bob
