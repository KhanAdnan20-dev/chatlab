import React, { useState, useEffect, useRef } from 'react';
import MessageItem from './components/MessageItem';
import NlpInspector from './components/NlpInspector';

const API_BASE_URL = 'http://127.0.0.1:8000';

const SUGGESTIONS = [
  'What are your capabilities?',
  'How should I study for exams?',
  'I have a Python exam on Monday.',
  'Teach me recursion',
  'Compare CNN and transformers',
];

export default function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'bot',
      text: 'Hello! I am ChatLab, your modular hybrid conversational AI assistant powered by Layer 1 NLP and Soft-Computing Fuzzy Routing.',
      metadata: {
        layer: 'layer1',
        intent: 'greeting',
        confidence: 0.95,
        entities: [],
        similarity_score: 1.0,
        complexity_score: 0.19,
        entity_density: 0.0,
        activation_score: 0.15,
        routing_reason: 'Fuzzy activation score (0.15 <= 0.50) confirms Layer 1 suitability.',
        processing_time_ms: 4.2,
        fallback: false,
      },
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [showInspector, setShowInspector] = useState(true);
  const [latestMetadata, setLatestMetadata] = useState(messages[0].metadata);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSendMessage = async (textToSend) => {
    const query = (textToSend || inputValue).trim();
    if (!query || isLoading) return;

    setErrorMsg(null);
    setInputValue('');

    const userMessage = {
      id: Date.now(),
      sender: 'user',
      text: query,
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: query }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error (${response.status})`);
      }

      const data = await response.json();

      const botMessage = {
        id: Date.now() + 1,
        sender: 'bot',
        text: data.response,
        metadata: {
          layer: data.layer,
          intent: data.intent,
          confidence: data.confidence,
          entities: data.entities || [],
          similarity_score: data.similarity_score,
          matched_pattern: data.matched_pattern,
          complexity_score: data.complexity_score,
          entity_density: data.entity_density,
          activation_score: data.activation_score,
          routing_reason: data.routing_reason,
          fallback: data.fallback,
          processing_time_ms: data.processing_time_ms,
        },
      };

      setMessages((prev) => [...prev, botMessage]);
      setLatestMetadata(botMessage.metadata);
    } catch (err) {
      console.error('Chat error:', err);
      setErrorMsg(err.message || 'Failed to connect to ChatLab backend.');
      const errorMessage = {
        id: Date.now() + 1,
        sender: 'bot',
        text: '⚠️ Unable to connect to the ChatLab backend. Please ensure the FastAPI server is running on http://127.0.0.1:8000.',
        metadata: {
          layer: 'error',
          intent: 'network_error',
          confidence: 0,
          entities: [],
          activation_score: 1.0,
          routing_reason: 'Network connectivity failure.',
          fallback: true,
        },
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setLatestMetadata(null);
  };

  return (
    <div className="app-container">
      <div className="chat-panel">
        {/* Header */}
        <header className="app-header">
          <div className="brand-section">
            <div className="brand-icon">CL</div>
            <div>
              <h1 className="brand-title">ChatLab</h1>
            </div>
            <span className="brand-badge">Layer 1 + Fuzzy Router Active</span>
          </div>

          <div className="header-actions">
            <button
              className={`btn-icon ${showInspector ? 'active' : ''}`}
              onClick={() => setShowInspector(!showInspector)}
              title="Toggle Soft Computing & NLP Inspector"
            >
              🔬 {showInspector ? 'Hide Inspector' : 'Show Inspector'}
            </button>
            <button
              className="btn-icon"
              onClick={handleClearChat}
              title="Clear current conversation"
            >
              🗑️ Clear
            </button>
          </div>
        </header>

        {/* Message Stream */}
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="welcome-card">
              <h2 className="welcome-title">Welcome to ChatLab</h2>
              <p className="welcome-desc">
                ChatLab is a hybrid conversational AI project demonstrating text preprocessing,
                TF-IDF intent classification, NER, and Mamdani Fuzzy Logic decision routing.
              </p>
              <div className="suggestion-chips">
                {SUGGESTIONS.map((suggestion, idx) => (
                  <button
                    key={idx}
                    className="chip-btn"
                    onClick={() => handleSendMessage(suggestion)}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <MessageItem
                key={msg.id}
                message={msg}
                onSelectMetadata={() => setLatestMetadata(msg.metadata)}
              />
            ))
          )}

          {isLoading && (
            <div className="message-row bot">
              <div className="avatar bot-avatar">CL</div>
              <div className="bubble-wrapper">
                <div className="bubble">
                  <div className="typing-dots">
                    <div className="dot" />
                    <div className="dot" />
                    <div className="dot" />
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="chat-input-area">
          <form
            className="input-form"
            onSubmit={(e) => {
              e.preventDefault();
              handleSendMessage();
            }}
          >
            <input
              type="text"
              className="chat-input"
              placeholder="Type your message (e.g., 'What can you do?', 'Teach me recursion')..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              disabled={isLoading}
            />
            <button
              type="submit"
              className="send-btn"
              disabled={isLoading || !inputValue.trim()}
            >
              Send
            </button>
          </form>
        </div>
      </div>

      {/* Side NLP Inspector */}
      {showInspector && (
        <NlpInspector
          metadata={latestMetadata}
          onClose={() => setShowInspector(false)}
        />
      )}
    </div>
  );
}
