// src/components/Message.jsx
import React from 'react';

const Message = ({ message }) => {
  const isUser = message.type === 'user';

  const SQLResults = ({ data, headers }) => (
    <div className="mt-3 bg-light rounded p-3">
      <div className="d-flex align-items-center text-muted mb-2">
        <span className="me-2">📊</span>
        Produits correspondants
      </div>
      <div className="table-responsive">
        <table className="table table-sm table-striped">
          <thead>
            <tr>
              {headers.map((header, i) => (
                <th key={i} scope="col">{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  return (
    <div className={`d-flex mb-3 ${isUser ? 'justify-content-end' : 'justify-content-start'}`}>
      <div className={`d-flex align-items-start ${isUser ? 'flex-row-reverse' : ''}`}>
        <div className={`rounded-circle bg-${isUser ? 'primary' : 'secondary'} text-white p-2 mx-2`}>
          {isUser ? '👤' : '🤖'}
        </div>
        <div className={`rounded p-3 ${
          isUser ? 'bg-primary text-white' : 'bg-light'
        }`} style={{ maxWidth: '80%' }}>
          <div style={{ whiteSpace: 'pre-line' }}>{message.content}</div>

          {message.sqlResults && (
            <SQLResults
              headers={message.sqlResults.headers}
              data={message.sqlResults.data}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default Message;