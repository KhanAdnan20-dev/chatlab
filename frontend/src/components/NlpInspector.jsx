import React from 'react';

export default function NlpInspector({ metadata, onClose }) {
  if (!metadata) {
    return (
      <aside className="inspector-panel">
        <div className="inspector-header">
          <div className="inspector-title">
            <span>🔬 Soft Computing & NLP Inspector</span>
          </div>
          {onClose && (
            <button className="btn-icon" onClick={onClose} title="Close Inspector">
              ✕
            </button>
          )}
        </div>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '1rem' }}>
          Send a message to view the internal Mamdani Fuzzy Inference routing and Layer 1 NLP metrics.
        </p>
      </aside>
    );
  }

  const confPercent = Math.round((metadata.confidence || 0) * 100);
  const simPercent = Math.round((metadata.similarity_score || 0) * 100);
  const compPercent = Math.round((metadata.complexity_score || 0) * 100);
  const entPercent = Math.round((metadata.entity_density || 0) * 100);
  const actPercent = Math.round((metadata.activation_score || 0) * 100);

  const isLayer1 = metadata.layer === 'layer1';

  return (
    <aside className="inspector-panel">
      <div className="inspector-header">
        <div className="inspector-title">
          <span>🔬 Soft Computing & NLP Inspector</span>
        </div>
        {onClose && (
          <button className="btn-icon" onClick={onClose} title="Close Inspector">
            ✕
          </button>
        )}
      </div>

      {/* Layer & Routing Decision */}
      <div className="inspector-card" style={{ borderLeft: `4px solid ${isLayer1 ? '#10b981' : '#f59e0b'}` }}>
        <div className="card-label">Fuzzy Selected Layer</div>
        <div className="card-value" style={{ color: isLayer1 ? '#34d399' : '#fcd34d', fontSize: '1rem', fontWeight: 'bold' }}>
          {metadata.layer.toUpperCase()}
          {metadata.fallback && ' (Pending LLM)'}
        </div>
        {metadata.routing_reason && (
          <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: '1.4', marginTop: '0.25rem' }}>
            {metadata.routing_reason}
          </div>
        )}
      </div>

      {/* Fuzzy Layer Activation Score */}
      <div className="inspector-card">
        <div className="card-label">Mamdani Activation Score (A)</div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
          <span>Layer 1 (0.0) ◄────────► Layer 2 (1.0)</span>
          <span style={{ fontWeight: 'bold', color: isLayer1 ? '#34d399' : '#fcd34d' }}>{metadata.activation_score}</span>
        </div>
        <div className="progress-bar-container">
          <div
            className="progress-bar-fill"
            style={{
              width: `${actPercent}%`,
              background: actPercent <= 50
                ? 'linear-gradient(90deg, #10b981, #06b6d4)'
                : 'linear-gradient(90deg, #f59e0b, #ef4444)',
            }}
          />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--text-muted)' }}>
          <span>0.00: Deterministic NLP</span>
          <span>0.50 Threshold</span>
          <span>1.00: Generative LLM</span>
        </div>
      </div>

      {/* Four Fuzzy Input Variables */}
      <div className="inspector-card">
        <div className="card-label">Fuzzy Input Antecedents</div>

        {/* 1. Intent Confidence */}
        <div style={{ marginTop: '0.35rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            <span>1. Intent Confidence (C):</span>
            <span style={{ color: '#38bdf8' }}>{confPercent}%</span>
          </div>
          <div className="progress-bar-container" style={{ height: '4px', marginTop: '3px' }}>
            <div className="progress-bar-fill" style={{ width: `${confPercent}%`, background: '#38bdf8' }} />
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '2px' }}>
            Intent: <strong style={{ color: '#93c5fd' }}>{metadata.intent}</strong>
          </div>
        </div>

        {/* 2. Semantic Similarity */}
        <div style={{ marginTop: '0.6rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            <span>2. Semantic Similarity (S):</span>
            <span style={{ color: '#2dd4bf' }}>{simPercent}%</span>
          </div>
          <div className="progress-bar-container" style={{ height: '4px', marginTop: '3px' }}>
            <div className="progress-bar-fill" style={{ width: `${simPercent}%`, background: '#2dd4bf' }} />
          </div>
          {metadata.matched_pattern && (
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px' }}>
              Pattern: "{metadata.matched_pattern}"
            </div>
          )}
        </div>

        {/* 3. Query Complexity */}
        <div style={{ marginTop: '0.6rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            <span>3. Query Complexity (L):</span>
            <span style={{ color: '#a78bfa' }}>{compPercent}%</span>
          </div>
          <div className="progress-bar-container" style={{ height: '4px', marginTop: '3px' }}>
            <div className="progress-bar-fill" style={{ width: `${compPercent}%`, background: '#a78bfa' }} />
          </div>
        </div>

        {/* 4. Entity Density */}
        <div style={{ marginTop: '0.6rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            <span>4. Entity Density (E):</span>
            <span style={{ color: '#f472b6' }}>{entPercent}%</span>
          </div>
          <div className="progress-bar-container" style={{ height: '4px', marginTop: '3px' }}>
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
