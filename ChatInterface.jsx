import React, { useState, useRef, useEffect } from 'react';
import Message from './Message';
import UserSelector from './UserSelector';
import SecurityModeSelector from './SecurityModeSelector';
import DebugPanel from './DebugPanel';
import './ChatInterface.css';

const ChatInterface = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [securityMode, setSecurityMode] = useState('SAFE');
  const [debugInfo, setDebugInfo] = useState(null);
  const [showDebugPanel, setShowDebugPanel] = useState(true); // État pour contrôler l'affichage du panel
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading || !currentUser) return;

    setMessages(prev => [...prev, {
      type: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString()
    }]);

    setLoading(true);
    const userInput = input.trim();
    setInput('');

    // Réinitialiser les infos de débogage
    setDebugInfo(null);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userInput,
          template_type: 'sales',
          user: {
            id: currentUser.id,
            first_name: currentUser.first_name,
            last_name: currentUser.last_name
          },
          security_mode: securityMode,
          debug: showDebugPanel // Activer le débogage si le panel est visible
        })
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();

      // Mettre à jour les infos de débogage si elles sont disponibles
      if (data.debug) {
        console.log("Debug info received:", data.debug);
        setDebugInfo(data.debug);
      }

      setMessages(prev => [...prev, {
        type: 'assistant',
        content: data.content,
        timestamp: data.timestamp,
        sqlResults: data.sqlResults
      }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        type: 'assistant',
        content: 'Désolé, une erreur est survenue lors du traitement de votre demande.',
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container-fluid py-5">
      <div className="row justify-content-center">
        <div className={`col-12 ${showDebugPanel ? 'col-lg-6' : 'col-lg-8'} mb-4 mb-lg-0`}>
          <div className="card shadow-lg">
            <div className="card-header chat-header">
              <div className="header-content">
                <div className="header-top">
                  <span className="app-title">💬 Assistant IA Pharmacie</span>
                  <div className="selector-group">
                    <SecurityModeSelector
                      mode={securityMode}
                      onModeChange={setSecurityMode}
                    />
                    <UserSelector
                      onUserChange={setCurrentUser}
                      currentUser={currentUser}
                    />
                  </div>
                </div>

                {currentUser && (
                  <div className="header-bottom">
                    <div className="status-pill user-status">
                      👤 {currentUser.first_name} {currentUser.last_name}
                    </div>
                    <div className={`status-pill mode-status ${securityMode.toLowerCase()}`}>
                      {securityMode === 'SAFE' ? '🔒' : '🔓'} {securityMode}
                    </div>
                    <button
                      className="btn btn-sm btn-outline-light"
                      onClick={() => setShowDebugPanel(!showDebugPanel)}
                    >
                      {showDebugPanel ? '🔍 Masquer débogage' : '🔍 Afficher débogage'}
                    </button>
                  </div>
                )}
              </div>
            </div>

            {!currentUser && (
              <div className="warning-banner">
                ⚠️ Veuillez sélectionner un utilisateur pour commencer
              </div>
            )}

            <div className="card-body chat-body">
              <div
                ref={scrollRef}
                className="messages-container"
              >
                {messages.map((message, idx) => (
                  <Message key={idx} message={message} />
                ))}
                {loading && (
                  <div className="loading-indicator">
                    <div className="spinner-border text-primary" role="status">
                      <span className="visually-hidden">Loading...</span>
                    </div>
                  </div>
                )}
              </div>

              <form onSubmit={handleSubmit} className="chat-input-form">
                <div className="input-group">
                  <input
                    type="text"
                    className="form-control chat-input"
                    placeholder={currentUser ?
                      "Posez votre question sur nos produits..." :
                      "Veuillez d'abord sélectionner un utilisateur"}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    disabled={loading || !currentUser}
                  />
                  <button
                    className={`btn ${currentUser ? 'btn-primary' : 'btn-secondary'} send-button`}
                    type="submit"
                    disabled={loading || !currentUser}
                  >
                    {loading ? (
                      <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                    ) : '➤'}
                    Envoyer
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>

        {/* Panel de débogage */}
        {showDebugPanel && (
          <div className="col-12 col-lg-6">
            <DebugPanel debugInfo={debugInfo} loading={loading} />
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatInterface;