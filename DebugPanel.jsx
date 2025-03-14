// src/components/DebugPanel.jsx
import React from 'react';

const DebugPanel = ({ debugInfo, loading }) => {
  if (loading) {
    return (
      <div className="card shadow-lg h-100">
        <div className="card-header bg-dark text-white">
          <h5 className="mb-0">
            <span className="me-2">🔍</span>
            Pipeline de Traitement
          </h5>
        </div>
        <div className="card-body bg-light d-flex justify-content-center align-items-center">
          <div className="text-center">
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
            <p className="mt-3 text-muted">Traitement en cours...</p>
          </div>
        </div>
      </div>
    );
  }

  if (!debugInfo) {
    return (
      <div className="card shadow-lg h-100">
        <div className="card-header bg-dark text-white">
          <h5 className="mb-0">
            <span className="me-2">🔍</span>
            Pipeline de Traitement
          </h5>
        </div>
        <div className="card-body bg-light">
          <p className="text-muted text-center my-5">Posez une question pour voir les détails du traitement</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card shadow-lg h-100">
      <div className="card-header bg-dark text-white">
        <h5 className="mb-0">
          <span className="me-2">🔍</span>
          Pipeline de Traitement
        </h5>
      </div>
      <div className="card-body bg-light p-0">
        <div className="debug-content p-3" style={{ maxHeight: '600px', overflowY: 'auto', fontSize: '0.85rem' }}>
          {debugInfo.firstDecision && (
            <div className="mb-3">
              <div className="d-flex align-items-center text-primary mb-2">
                <span className="me-2">📡</span>
                <strong>Décision initiale du modèle</strong>
              </div>
              <div className="bg-white p-2 rounded border">
                <pre className="mb-0" style={{ whiteSpace: 'pre-wrap' }}>{debugInfo.firstDecision}</pre>
              </div>
            </div>
          )}

          {debugInfo.functionCalls && debugInfo.functionCalls.length > 0 && (
            <div className="mb-3">
              <div className="d-flex align-items-center text-success mb-2">
                <span className="me-2">🔧</span>
                <strong>Appels de fonction</strong>
              </div>
              {debugInfo.functionCalls.map((call, index) => (
                <div key={index} className="bg-white p-2 rounded border mb-2">
                  <div className="text-muted mb-1">Fonction: <span className="text-dark fw-bold">{call.name}</span></div>
                  <div className="text-muted mb-1">Arguments:</div>
                  <pre className="bg-light p-2 rounded small mb-0" style={{ whiteSpace: 'pre-wrap' }}>{JSON.stringify(call.args, null, 2)}</pre>
                </div>
              ))}
            </div>
          )}

          {debugInfo.executionLogs && debugInfo.executionLogs.length > 0 && (
            <div className="mb-3">
              <div className="d-flex align-items-center text-info mb-2">
                <span className="me-2">📝</span>
                <strong>Logs d'exécution</strong>
              </div>
              <div className="bg-dark text-light p-2 rounded border" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                {debugInfo.executionLogs.map((log, index) => (
                  <div key={index} className="mb-1">
                    <pre className="mb-0 small" style={{ whiteSpace: 'pre-wrap', color: log.type === 'error' ? '#f8d7da' : '#d1e7dd' }}>{log.message}</pre>
                  </div>
                ))}
              </div>
            </div>
          )}

          {debugInfo.finalResponse && (
            <div className="mb-3">
              <div className="d-flex align-items-center text-success mb-2">
                <span className="me-2">✅</span>
                <strong>Réponse finale</strong>
              </div>
              <div className="bg-white p-2 rounded border">
                <pre className="mb-0" style={{ whiteSpace: 'pre-wrap' }}>{debugInfo.finalResponse}</pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DebugPanel;