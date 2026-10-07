import React from 'react';

export default function MessageItem({ message, onSelectMetadata }) {
  const isUser = message.sender === 'user';

  return (
    <div className={`message-row ${isUser ? 'user' : 'bot'}`}>
      <div className={`avatar ${isUser ? 'user-avatar' : 'bot-avatar'}`}>
        {isUser ? 'U' : 'CL'}
      </div>

      <div className="bubble-wrapper">
        <div className="bubble">
          {message.text}
        </div>

        {!isUser && message.metadata && (
          <div className="bubble-meta">
            <span className={`meta-tag ${message.metadata.fallback ? 'fallback-tag' : ''}`}>
              {message.metadata.layer || 'layer1'}
            </span>
            <span className="meta-tag">
              intent: {message.metadata.intent}
            </span>
            <span className="meta-tag">
              conf: {(message.metadata.confidence * 100).toFixed(0)}%
            </span>
            {message.metadata.processing_time_ms && (
              <span className="meta-tag">
                {message.metadata.processing_time_ms}ms
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
