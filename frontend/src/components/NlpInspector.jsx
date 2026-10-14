import React from 'react';

export default function NlpInspector({ metadata, onClose }) {
  if (!metadata) {
    return (
      <aside className="inspector-panel">
        <div className="inspector-header">
          <div className="inspector-title">
            <span>🔬 Mamdani Fuzzy Prompt Inspector</span>
          </div>
          {onClose && (
            <button className="btn-icon" onClick={onClose} title="Close Inspector">
              ✕
            </button>
          )}
        </div>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '1rem' }}>
          Send a message to view the internal Mamdani Fuzzy Inference Prompt Profile and NLP metrics.
        </p>
      </aside>
    );
  }

  const fp = metadata.fuzzy_profile || {
    technical_depth: 0.50,
    beginner_friendliness: 0.85,
    conversationalness: 0.88,
    mathematical_detail: 0.30,
    example_preference: 0.70,
    code_focus: 0.10,
    aiml_relevance: 0.80,
    explanation_depth: 0.50,
    query_complexity: 0.50,
    ambiguity_level: 0.20,
  };

  const confPercent = Math.round((metadata.confidence || 0) * 100);
  const simPercent = Math.round((metadata.similarity_score || 0) * 100);
  const compPercent = Math.round((metadata.complexity_score || fp.query_complexity || 0) * 100);
  const entPercent = Math.round((metadata.entity_density || 0) * 100);

  const engineName = metadata.layer === 'quiz' ? 'Quiz Engine' : (metadata.model_name || 'Gemma 4 E2B Instruct');

  return (
    <aside className="inspector-panel">
      <div className="inspector-header">
        <div className="inspector-title">
          <span>🔬 Mamdani Fuzzy Prompt Inspector</span>
        </div>
        {onClose && (
          <button className="btn-icon" onClick={onClose} title="Close Inspector">
            ✕
          </button>
        )}
      </div>

      {/* Prompt Orchestration & Model */}
      <div className="inspector-card" style={{ borderLeft: '4px solid #f59e0b' }}>
        <div className="card-label">Conversational Engine</div>
        <div className="card-value" style={{ color: '#fcd34d', fontSize: '1.05rem', fontWeight: 'bold' }}>
          {engineName}
        </div>
        {metadata.depth_mode && (
          <div style={{ marginTop: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Depth Mode:</span>
            <span style={{
              fontSize: '0.72rem',
              fontWeight: 'bold',
              textTransform: 'uppercase',
              padding: '1px 7px',
              borderRadius: '4px',
              background: metadata.depth_mode === 'auto' ? 'rgba(99, 102, 241, 0.2)' :
                          metadata.depth_mode === 'easy' ? 'rgba(16, 185, 129, 0.2)' :
                          metadata.depth_mode === 'medium' ? 'rgba(56, 189, 248, 0.2)' :
                          metadata.depth_mode === 'hard' ? 'rgba(236, 72, 153, 0.2)' :
                          'linear-gradient(135deg, rgba(168, 85, 247, 0.3), rgba(236, 72, 153, 0.3))',
              color: metadata.depth_mode === 'auto' ? '#c7d2fe' :
                     metadata.depth_mode === 'easy' ? '#a7f3d0' :
                     metadata.depth_mode === 'medium' ? '#bae6fd' :
                     metadata.depth_mode === 'hard' ? '#fbcfe8' : '#f3e8ff',
              border: `1px solid ${
                metadata.depth_mode === 'auto' ? 'rgba(129, 140, 248, 0.4)' :
                metadata.depth_mode === 'easy' ? 'rgba(52, 211, 153, 0.4)' :
                metadata.depth_mode === 'medium' ? 'rgba(56, 189, 248, 0.4)' :
                metadata.depth_mode === 'hard' ? 'rgba(244, 114, 182, 0.4)' : 'rgba(216, 180, 254, 0.6)'
              }`
            }}>
              {metadata.depth_mode}
            </span>
          </div>
        )}
        <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: '1.4', marginTop: '0.25rem' }}>
          Guided dynamically by Mamdani Fuzzy Inference System & NLP Response Demand Analyzer.
        </div>
      </div>

      {/* Dynamic Token Budget & NLP Response Demand */}
      {metadata.token_budget && (
        <div className="inspector-card" style={{ borderLeft: '4px solid #10b981' }}>
          <div className="card-label">NLP Response Demand & Token Budget</div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.25rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: '600', color: '#6ee7b7' }}>
              {metadata.response_type ? metadata.response_type.replace(/_/g, ' ').toUpperCase() : 'ADAPTIVE'}
            </span>
            <span style={{ fontSize: '0.82rem', fontWeight: 'bold', color: '#10b981', background: 'rgba(16, 185, 129, 0.15)', padding: '2px 8px', borderRadius: '4px' }}>
              {metadata.token_budget} max tokens
            </span>
          </div>
          {metadata.tokens_generated > 0 && (
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.35rem' }}>
              Tokens: <strong>{metadata.tokens_generated}</strong> / {metadata.token_budget} ({Math.round((metadata.tokens_generated / metadata.token_budget) * 100)}% budget used)
            </div>
          )}
          {metadata.demand_score !== undefined && metadata.demand_score !== null && (
            <div style={{ marginTop: '0.35rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                <span>Information Demand Intensity:</span>
                <span>{Math.round(metadata.demand_score * 100)}%</span>
              </div>
              <div className="progress-bar-container" style={{ height: '4px', marginTop: '2px' }}>
                <div className="progress-bar-fill" style={{ width: `${Math.round(metadata.demand_score * 100)}%`, background: '#10b981' }} />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Fuzzy Query Profile */}
      <div className="inspector-card">
        <div className="card-label">Fuzzy Query Profile</div>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
          Mamdani FIS Defuzzified Directives for Gemma 4
        </div>

        {/* Technical Depth */}
        <div style={{ marginTop: '0.3rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            <span>Technical Depth:</span>
            <span style={{ color: '#38bdf8', fontWeight: 'bold' }}>{Math.round(fp.technical_depth * 100)}%</span>
          </div>
          <div className="progress-bar-container" style={{ height: '4px', marginTop: '2px' }}>
            <div className="progress-bar-fill" style={{ width: `${Math.round(fp.technical_depth * 100)}%`, background: '#38bdf8' }} />
          </div>
        </div>

        {/* Beginner Friendly */}
        <div style={{ marginTop: '0.45rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            <span>Beginner Friendly:</span>
            <span style={{ color: '#34d399', fontWeight: 'bold' }}>{Math.round(fp.beginner_friendliness * 100)}%</span>
          </div>
          <div className="progress-bar-container" style={{ height: '4px', marginTop: '2px' }}>
            <div className="progress-bar-fill" style={{ width: `${Math.round(fp.beginner_friendliness * 100)}%`, background: '#34d399' }} />
          </div>
        </div>

        {/* Conversationalness */}
        <div style={{ marginTop: '0.45rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            <span>Conversationalness:</span>
            <span style={{ color: '#f59e0b', fontWeight: 'bold' }}>{Math.round(fp.conversationalness * 100)}%</span>
          </div>
          <div className="progress-bar-container" style={{ height: '4px', marginTop: '2px' }}>
            <div className="progress-bar-fill" style={{ width: `${Math.round(fp.conversationalness * 100)}%`, background: '#f59e0b' }} />
          </div>
        </div>

        {/* Mathematical Detail */}
        <div style={{ marginTop: '0.45rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            <span>Mathematical Detail:</span>
            <span style={{ color: '#a78bfa', fontWeight: 'bold' }}>{Math.round(fp.mathematical_detail * 100)}%</span>
          </div>
          <div className="progress-bar-container" style={{ height: '4px', marginTop: '2px' }}>
            <div className="progress-bar-fill" style={{ width: `${Math.round(fp.mathematical_detail * 100)}%`, background: '#a78bfa' }} />
          </div>
        </div>

        {/* Example Preference */}
        <div style={{ marginTop: '0.45rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            <span>Example Preference:</span>
            <span style={{ color: '#f472b6', fontWeight: 'bold' }}>{Math.round(fp.example_preference * 100)}%</span>
          </div>
          <div className="progress-bar-container" style={{ height: '4px', marginTop: '2px' }}>
            <div className="progress-bar-fill" style={{ width: `${Math.round(fp.example_preference * 100)}%`, background: '#f472b6' }} />
          </div>
        </div>

        {/* Code Focus */}
        <div style={{ marginTop: '0.45rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            <span>Code Focus:</span>
            <span style={{ color: '#fb7185', fontWeight: 'bold' }}>{Math.round(fp.code_focus * 100)}%</span>
          </div>
          <div className="progress-bar-container" style={{ height: '4px', marginTop: '2px' }}>
            <div className="progress-bar-fill" style={{ width: `${Math.round(fp.code_focus * 100)}%`, background: '#fb7185' }} />
          </div>
        </div>

        {/* AIML Relevance */}
        <div style={{ marginTop: '0.45rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            <span>AIML Relevance:</span>
            <span style={{ color: '#2dd4bf', fontWeight: 'bold' }}>{Math.round(fp.aiml_relevance * 100)}%</span>
          </div>
          <div className="progress-bar-container" style={{ height: '4px', marginTop: '2px' }}>
            <div className="progress-bar-fill" style={{ width: `${Math.round(fp.aiml_relevance * 100)}%`, background: '#2dd4bf' }} />
          </div>
        </div>
      </div>

      {/* NLP Feature Antecedents */}
      <div className="inspector-card">
        <div className="card-label">NLP Feature Antecedents</div>

        {/* Intent Confidence */}
        <div style={{ marginTop: '0.35rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            <span>Intent Confidence (C):</span>
            <span style={{ color: '#38bdf8' }}>{confPercent}%</span>
          </div>
          <div className="progress-bar-container" style={{ height: '4px', marginTop: '2px' }}>
            <div className="progress-bar-fill" style={{ width: `${confPercent}%`, background: '#38bdf8' }} />
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '2px' }}>
            Intent: <strong style={{ color: '#93c5fd' }}>{metadata.intent}</strong>
          </div>
        </div>

        {/* Semantic Similarity */}
        <div style={{ marginTop: '0.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            <span>Semantic Similarity (S):</span>
            <span style={{ color: '#2dd4bf' }}>{simPercent}%</span>
          </div>
          <div className="progress-bar-container" style={{ height: '4px', marginTop: '2px' }}>
            <div className="progress-bar-fill" style={{ width: `${simPercent}%`, background: '#2dd4bf' }} />
          </div>
        </div>

        {/* Query Complexity */}
        <div style={{ marginTop: '0.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            <span>Query Complexity (L):</span>
            <span style={{ color: '#a78bfa' }}>{compPercent}%</span>
          </div>
          <div className="progress-bar-container" style={{ height: '4px', marginTop: '2px' }}>
            <div className="progress-bar-fill" style={{ width: `${compPercent}%`, background: '#a78bfa' }} />
          </div>
        </div>

        {/* Entity Density */}
        <div style={{ marginTop: '0.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            <span>Entity Density (E):</span>
            <span style={{ color: '#f472b6' }}>{entPercent}%</span>
          </div>
          <div className="progress-bar-container" style={{ height: '4px', marginTop: '2px' }}>
            <div className="progress-bar-fill" style={{ width: `${entPercent}%`, background: '#f472b6' }} />
          </div>
        </div>
      </div>

      {/* Extracted Named Entities */}
      <div className="inspector-card">
        <div className="card-label">Extracted Entities ({metadata.entities?.length || 0})</div>
        {metadata.entities && metadata.entities.length > 0 ? (
          <div className="entity-badge-list">
            {metadata.entities.map((ent, idx) => (
              <span key={idx} className="entity-chip">
                {ent.text}
                <span className="entity-label-tag">[{ent.label}]</span>
              </span>
            ))}
          </div>
        ) : (
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            No named entities detected
          </div>
        )}
      </div>

      {/* Pipeline Latency */}
      {metadata.processing_time_ms && (
        <div className="inspector-card">
          <div className="card-label">End-to-End Latency</div>
          <div className="card-value" style={{ color: '#34d399', fontSize: '0.85rem' }}>
            {metadata.processing_time_ms} ms
          </div>
        </div>
      )}
    </aside>
  );
}
