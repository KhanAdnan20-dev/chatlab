import React, { useState, useEffect, useRef } from 'react';
import MessageItem from './components/MessageItem';
import NlpInspector from './components/NlpInspector';
import QuizPanel from './components/QuizPanel';
import RoadmapModal from './components/RoadmapModal';
import RoadmapStudio from './components/RoadmapStudio';
import SyllabusLabModal from './components/SyllabusLabModal';
import LiteratureDrawer from './components/LiteratureDrawer';
import { sendMessage, streamMessage } from './services/api';
import { Compass, GraduationCap, BookMarked, Brain, Sparkles } from 'lucide-react';

const SUGGESTIONS = [
  'Compare Hard Computing vs Soft Computing (Zadeh 1965)',
  'Explain ANFIS 5-Layer Hybrid Learning Architecture',
  'Mamdani Centroid Defuzzification vs Sugeno Weighted Average',
  'Why Minsky proved Single-Layer Perceptrons cannot solve XOR',
  'A* Search Algorithm Admissibility and Heuristic Consistency',
];

const INITIAL_METADATA = {
  layer: 'gemma',
  model_name: 'Gemma 4 E2B Instruct',
  intent: 'greeting',
  confidence: 0.95,
  entities: [],
  similarity_score: 1.0,
  complexity_score: 0.19,
  entity_density: 0.0,
  activation_score: 0.25,
  routing_reason: 'Mamdani FIS derives prompt profile for Gemma 4 E2B Instruct.',
  processing_time_ms: 12.4,
  fallback: false,
  citations: [],
};

export default function App() {
  const [activeMode, setActiveMode] = useState('chat'); // 'chat' | 'roadmap' | 'quiz'
  const [modelChoice, setModelChoice] = useState('gemma'); // 'gemma' | 'deepseek'
  const [isSocratic, setIsSocratic] = useState(false);
  const [depthMode, setDepthMode] = useState('auto'); // 'auto' | 'easy' | 'medium' | 'hard'
  const [isRoadmapOpen, setIsRoadmapOpen] = useState(false);
  const [isSyllabusOpen, setIsSyllabusOpen] = useState(false);
  const [roadmapDomain, setRoadmapDomain] = useState('all');
  const [roadmapStudioQuery, setRoadmapStudioQuery] = useState('AI and Soft Computing midsem syllabus units 1 to 6');
  const [selectedCitation, setSelectedCitation] = useState(null);
  const [citationTriggerEl, setCitationTriggerEl] = useState(null);

  const handleOpenCitation = (citation, triggerEl) => {
    setSelectedCitation(citation);
    setCitationTriggerEl(triggerEl);
  };

  const handleCloseCitation = () => {
    setSelectedCitation(null);
  };

  const handleOpenRoadmapStudio = (targetQuery) => {
    if (targetQuery) {
      setRoadmapStudioQuery(targetQuery);
    }
    setActiveMode('roadmap');
  };
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'bot',
      text: 'Welcome to ChatLab. Query any AI & Soft Computing concept, explore curriculum DAG milestones, or run a viva exam simulation.',
      metadata: INITIAL_METADATA,
    },
  ]);
  const [sessionId, setSessionId] = useState(null);
  const [inputValue, setInputValue] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isWaitingForFirstToken, setIsWaitingForFirstToken] = useState(false);
  const [showInspector, setShowInspector] = useState(true);
  const [latestMetadata, setLatestMetadata] = useState(INITIAL_METADATA);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView?.({ behavior: 'smooth' });
  }, [messages, isGenerating, isWaitingForFirstToken]);

  const handleSendMessage = async (textToSend) => {
    const query = (textToSend || inputValue).trim();
    if (!query || isGenerating) return;

    setInputValue('');

    const userMessage = {
      id: Date.now(),
      sender: 'user',
      text: query,
    };

    const botMessageId = Date.now() + 1;
    const initialBotMessage = {
      id: botMessageId,
      sender: 'bot',
      text: '',
      thinkingText: '',
      isStreaming: true,
      metadata: null,
    };

    setMessages((prev) => [...prev, userMessage, initialBotMessage]);
    setIsGenerating(true);
    setIsWaitingForFirstToken(true);

    const turnRequestId = `turn_${userMessage.id}`;

    try {
      await streamMessage(
        query,
        sessionId,
        (token) => {
          setIsWaitingForFirstToken(false); // First token arrived, hide waiting dots
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === botMessageId
                ? { ...msg, text: msg.text + token }
                : msg
            )
          );
        },
        (meta) => {
          setIsGenerating(false);
          setIsWaitingForFirstToken(false);
          if (meta.session_id) {
            setSessionId(meta.session_id);
          }
          const finalMetadata = {
            layer: meta.layer,
            intent: meta.intent,
            confidence: meta.confidence,
            entities: meta.entities || [],
            similarity_score: meta.similarity_score,
            matched_pattern: meta.matched_pattern,
            complexity_score: meta.complexity_score,
            entity_density: meta.entity_density,
            activation_score: meta.activation_score,
            routing_reason: meta.routing_reason,
            fallback: meta.fallback,
            processing_time_ms: meta.processing_time_ms,
            fuzzy_profile: meta.fuzzy_profile,
            model_name: meta.model_name || (modelChoice === 'deepseek' ? 'DeepSeek-R1 Distill Qwen 1.5B' : 'Gemma 4 E2B Instruct'),
            retried: meta.retried,
            tokens_generated: meta.tokens_generated,
            token_budget: meta.token_budget,
            response_type: meta.response_type,
            demand_score: meta.demand_score,
            citations: meta.citations || [],
            roadmap: meta.roadmap,
            socratic_mode: meta.socratic_mode || isSocratic,
            depth_mode: meta.depth_mode || depthMode,
          };
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === botMessageId
                ? { ...msg, isStreaming: false, metadata: finalMetadata }
                : msg
            )
          );
          setLatestMetadata(finalMetadata);
        },
        async (err) => {
          console.warn('Stream interrupted, fallback to standard request:', err);
          const data = await sendMessage(query, sessionId, isSocratic, turnRequestId, depthMode, modelChoice);
          setIsGenerating(false);
          setIsWaitingForFirstToken(false);
          if (data.session_id) setSessionId(data.session_id);
          const fallbackMeta = {
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
            fuzzy_profile: data.fuzzy_profile,
            model_name: data.model_name || (modelChoice === 'deepseek' ? 'DeepSeek-R1 Distill Qwen 1.5B' : 'Gemma 4 E2B Instruct'),
            retried: data.retried,
            tokens_generated: data.tokens_generated,
            token_budget: data.token_budget,
            response_type: data.response_type,
            demand_score: data.demand_score,
            citations: data.citations || [],
            roadmap: data.roadmap,
            socratic_mode: data.socratic_mode || isSocratic,
            depth_mode: data.depth_mode || depthMode,
          };
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === botMessageId
                ? { ...msg, text: data.response, isStreaming: false, metadata: fallbackMeta }
                : msg
            )
          );
          setLatestMetadata(fallbackMeta);
        },
        isSocratic,
        turnRequestId,
        depthMode,
        modelChoice,
        (thinkToken) => {
          setIsWaitingForFirstToken(false);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === botMessageId
                ? { ...msg, thinkingText: (msg.thinkingText || '') + thinkToken }
                : msg
            )
          );
        }
      );
    } catch (err) {
      console.error('Chat error:', err);
      const errorMessage = {
        id: botMessageId,
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
          model_name: 'Gemma 4 E2B Instruct',
          citations: [],
        },
      };
      setMessages((prev) =>
        prev.map((msg) => (msg.id === botMessageId ? errorMessage : msg))
      );
    } finally {
      setIsGenerating(false);
      setIsWaitingForFirstToken(false);
    }
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
            <div className="system-status-pill">
              <span
                className="system-status-dot"
                style={{ backgroundColor: modelChoice === 'deepseek' ? '#a855f7' : '#10b981' }}
              ></span>
              <span>{modelChoice === 'deepseek' ? 'DeepSeek-R1 • Reasoning' : 'Gemma 4 • Online'}</span>
            </div>
          </div>

          <div className="mode-toggle-group">
            <button
              className={`mode-btn ${activeMode === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveMode('chat')}
            >
              Chat
            </button>
            <button
              className={`mode-btn ${activeMode === 'roadmap' ? 'active' : ''}`}
              onClick={() => setActiveMode('roadmap')}
            >
              Roadmap Studio
            </button>
            <button
              className={`mode-btn ${activeMode === 'quiz' ? 'active' : ''}`}
              onClick={() => setActiveMode('quiz')}
            >
              Viva Exam
            </button>
          </div>

          <div className="header-actions">
            {activeMode === 'chat' && (
              <>
                {/* Dual-Engine LLM Switcher */}
                <div className="model-selector-toggle" title="Select LLM Engine">
                  <button
                    type="button"
                    className={`model-selector-btn ${modelChoice === 'gemma' ? 'active-gemma' : ''}`}
                    onClick={() => setModelChoice('gemma')}
                    title="Gemma 4 E2B Instruct: Curriculum & AI/ML Tutor"
                  >
                    <Sparkles style={{ width: '12px', height: '12px' }} />
                    <span>Gemma 4</span>
                  </button>
                  <button
                    type="button"
                    className={`model-selector-btn ${modelChoice === 'deepseek' ? 'active-deepseek' : ''}`}
                    onClick={() => setModelChoice('deepseek')}
                    title="DeepSeek-R1 Distill Qwen 1.5B: Math & Step-by-Step Reasoning"
                  >
                    <Brain style={{ width: '12px', height: '12px' }} />
                    <span>DeepSeek-R1</span>
                  </button>
                </div>

                {/* Socratic Guidance Toggle */}
                <button
                  className={`btn-icon ${isSocratic ? 'active' : ''}`}
                  onClick={() => setIsSocratic(!isSocratic)}
                  title="Toggle active-recall Socratic guidance"
                >
                  <GraduationCap style={{ width: '13px', height: '13px' }} />
                  <span>{isSocratic ? 'Socratic: ON' : 'Socratic: OFF'}</span>
                </button>

                {/* AI & Soft Computing Syllabus Lab Button */}
                <button
                  className="btn-icon"
                  onClick={() => setIsSyllabusOpen(true)}
                  title="Open Syllabus Master Lab (Units I-VI / CO1-CO6)"
                >
                  <BookMarked style={{ width: '13px', height: '13px', color: '#38bdf8' }} />
                  <span>Syllabus Units</span>
                </button>

                <button
                  className={`btn-icon ${showInspector ? 'active' : ''}`}
                  onClick={() => setShowInspector(!showInspector)}
                  title="Toggle Soft Computing & NLP Inspector"
                >
                  <span>{showInspector ? 'Hide Inspector' : 'Inspector'}</span>
                </button>
              </>
            )}
            <button
              className="btn-icon"
              onClick={() => {
                setMessages([]);
                setSessionId(null);
                setLatestMetadata(null);
              }}
              title="Clear current conversation"
            >
              <span>Clear</span>
            </button>
          </div>
        </header>

        {activeMode === 'chat' ? (
          <>
            {/* Message Stream */}
            <div className="messages-container">
              {messages.length === 0 ? (
                <div className="welcome-card">
                  <h2 className="welcome-title">Welcome to ChatLab</h2>
                  <p className="welcome-desc">
                    ChatLab is an educational conversational AI system powered by Google Gemma 4 E2B Instruct,
                    with prompt orchestration dynamically tuned by a Mamdani Fuzzy Inference System,
                    authoritative RAG literature citations, and interactive curriculum roadmaps.
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
                    onOpenRoadmapStudio={(topic) => handleOpenRoadmapStudio(topic)}
                    onOpenRoadmap={(dom) => {
                      setRoadmapDomain(dom || 'all');
                      setIsRoadmapOpen(true);
                    }}
                    onSelectCitation={handleOpenCitation}
                  />
                ))
              )}

              {isWaitingForFirstToken && (
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
              <div className="chat-input-toolbar">
                <div className="depth-selector-group">
                  <label htmlFor="depth-select" className="depth-label">
                    <svg className="depth-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                    </svg>
                    <span>Pedagogical Depth:</span>
                  </label>
                  <div className="depth-select-wrapper">
                    <select
                      id="depth-select"
                      className="depth-select"
                      value={depthMode}
                      onChange={(e) => setDepthMode(e.target.value)}
                      disabled={isGenerating}
                      title="Select desired pedagogical depth & token budget"
                    >
                      <option value="auto">Auto (Dynamic Mamdani FIS)</option>
                      <option value="easy">Concise (~384 tokens)</option>
                      <option value="medium">Standard (~640 tokens)</option>
                      <option value="hard">Rigorous (~1024 tokens)</option>
                      <option value="ultra">Exhaustive (~2048 tokens)</option>
                    </select>
                  </div>
                  <span className={`depth-badge-pill depth-pill-${depthMode}`}>
                    {depthMode === 'auto' && 'DYNAMIC FIS'}
                    {depthMode === 'easy' && '384 TOKENS'}
                    {depthMode === 'medium' && '640 TOKENS'}
                    {depthMode === 'hard' && '1024 TOKENS'}
                    {depthMode === 'ultra' && '2048 TOKENS'}
                  </span>
                </div>
                <div className="depth-hint-text">
                  {depthMode === 'auto' && 'Evaluates query syntactic complexity, entities, and Mamdani FIS demand.'}
                  {depthMode === 'easy' && 'Definition-oriented intuition and summary overview.'}
                  {depthMode === 'medium' && 'Balanced conceptual analysis with practical examples.'}
                  {depthMode === 'hard' && 'Rigorous mathematical derivations and architectural breakdown.'}
                  {depthMode === 'ultra' && 'Exhaustive monograph with full derivations, code workflows, and proofs.'}
                </div>
              </div>

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
                  placeholder={
                    modelChoice === 'deepseek'
                      ? "DeepSeek-R1 Reasoning: Ask a mathematical proof, algorithm analysis, or logic question..."
                      : isSocratic
                      ? "Socratic Mode: Ask a question and we'll reason step-by-step..."
                      : "Ask any AI/Soft Computing concept, formula derivation, or curriculum topic..."
                  }
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  disabled={isGenerating}
                />
                <button
                  type="submit"
                  className="send-btn"
                  disabled={isGenerating || !inputValue.trim()}
                >
                  Send
                </button>
              </form>
            </div>
          </>
        ) : activeMode === 'roadmap' ? (
          <RoadmapStudio 
            initialQuery={roadmapStudioQuery}
            onAskTopic={(topic) => {
              setActiveMode('chat');
              handleSendMessage(topic);
            }}
            onOpenQuiz={() => setActiveMode('quiz')}
          />
        ) : (
          <QuizPanel />
        )}
      </div>

      {/* Side NLP Inspector (Chat mode only) */}
      {activeMode === 'chat' && showInspector && (
        <NlpInspector
          metadata={latestMetadata}
          onClose={() => setShowInspector(false)}
        />
      )}

      {/* Interactive Curriculum Roadmap Modal */}
      <RoadmapModal
        isOpen={isRoadmapOpen}
        onClose={() => setIsRoadmapOpen(false)}
        onAskTopic={(topic) => handleSendMessage(topic)}
        initialDomain={roadmapDomain}
      />

      {/* AI & Soft Computing Syllabus Master Lab Modal */}
      <SyllabusLabModal
        isOpen={isSyllabusOpen}
        onClose={() => setIsSyllabusOpen(false)}
        onAskPrompt={(prompt) => handleSendMessage(prompt)}
        onOpenRoadmap={(domain) => {
          setRoadmapDomain(domain);
          setIsRoadmapOpen(true);
        }}
        onOpenQuiz={() => setActiveMode('quiz')}
      />

      {/* Authoritative RAG Literature Drawer */}
      <LiteratureDrawer
        isOpen={!!selectedCitation}
        citation={selectedCitation}
        triggerRef={citationTriggerEl}
        onClose={handleCloseCitation}
      />
    </div>
  );
}
