import React, { useState, useEffect } from 'react';
import mermaid from 'mermaid';
import CitationChip from './CitationChip';
import ThinkingBox from './ThinkingBox';
import { renderTextWithCitations } from '../utils/citationParser';
import { Compass, Sparkles, GraduationCap } from 'lucide-react';


const InlineMermaid = React.memo(function InlineMermaid({ code, onOpenStudio, topicQuery }) {
  const [svg, setSvg] = useState('');

  useEffect(() => {
    let isMounted = true;
    const id = `chat-mermaid-${Math.random().toString(36).substring(2, 9)}`;

    async function renderChart() {
      try {
        const cleanCode = code.trim();
        if (!cleanCode) return;
        const { svg: renderedSvg } = await mermaid.render(id, cleanCode);
        if (isMounted) setSvg(renderedSvg);
      } catch (err) {
        console.error('Chat mermaid render error:', err);
        const orphan = document.getElementById(`d${id}`) || document.getElementById(id);
        if (orphan) orphan.remove();
      }
    }
    renderChart();
    return () => { isMounted = false; };
  }, [code]);

  if (!svg) return null;

  return (
    <div style={{
      marginTop: '0.85rem',
      marginBottom: '0.85rem',
      borderRadius: '8px',
      backgroundColor: '#090d16',
      border: '1px solid #1e293b',
      overflow: 'hidden'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '6px 12px',
        backgroundColor: '#0d1322',
        borderBottom: '1px solid #1e293b',
        fontSize: '11px',
        fontFamily: 'monospace',
        color: '#94a3b8'
      }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#10b981' }}></span>
          DAG_FLOWCHART :: GRAPH_TD
        </span>
        {onOpenStudio && (
          <button
            onClick={() => onOpenStudio(topicQuery)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              padding: '3px 8px',
              borderRadius: '4px',
              backgroundColor: '#1e293b',
              border: '1px solid #334155',
              color: '#38bdf8',
              fontSize: '11px',
              cursor: 'pointer',
              fontWeight: '500'
            }}
          >
            <span>Open in Roadmap Studio ↗</span>
          </button>
        )}
      </div>
      <div 
        style={{
          padding: '1rem',
          overflowX: 'auto',
          display: 'flex',
          justifyContent: 'center'
        }}
        dangerouslySetInnerHTML={{ __html: svg }}
      />
    </div>
  );
});

export default function MessageItem({
  message,
  onSelectMetadata,
  onOpenRoadmap,
  onOpenRoadmapStudio,
  onSelectCitation,
}) {
  const isUser = message.sender === 'user';
  const meta = message.metadata;
  const citations = meta?.citations || [];
  const roadmap = meta?.roadmap;
  const isSocratic = meta?.socratic_mode;

  // Check specifically for explicitly tagged Mermaid code in text
  const text = message.text || '';
  const mermaidRegex = /```mermaid\s*([\s\S]*?)```/i;
  const mermaidMatch = text.match(mermaidRegex);
  const rawMermaid = mermaidMatch ? mermaidMatch[1].trim() : null;
  const displayText = mermaidMatch ? text.replace(mermaidRegex, '').trim() : text;

  // Render text with interactive inline citation badges and KaTeX math formulas
  const renderedContent = renderTextWithCitations(
    displayText,
    isUser ? [] : citations,
    onSelectCitation
  );

  return (
    <div className={`message-row ${isUser ? 'user' : 'bot'}`}>
      <div className={`avatar ${isUser ? 'user-avatar' : 'bot-avatar'}`}>
        {isUser ? 'U' : 'CL'}
      </div>

      <div className="bubble-wrapper">
        {message.thinkingText && (
          <ThinkingBox
            thinkingText={message.thinkingText}
            isStreaming={message.isStreaming}
            hasAnswerStarted={Boolean(displayText.trim())}
          />
        )}
        <div className="bubble" style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>

          {renderedContent}
          {rawMermaid && (
            <InlineMermaid 
              code={rawMermaid} 
              onOpenStudio={onOpenRoadmapStudio}
              topicQuery={roadmap?.title || displayText.slice(0, 40)}
            />
          )}
          {message.isStreaming && (
            <span style={{ display: 'inline-block', width: '6px', height: '14px', backgroundColor: '#38bdf8', marginLeft: '4px', verticalAlign: 'middle', opacity: 0.8 }}></span>
          )}
        </div>

        {/* Commandable Deep-Link Banner to Roadmap Studio */}
        {(roadmap || rawMermaid) && onOpenRoadmapStudio && (
          <div style={{
            marginTop: '0.65rem',
            padding: '8px 12px',
            borderRadius: '6px',
            backgroundColor: 'rgba(15, 23, 42, 0.9)',
            border: '1px solid #334155',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '8px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Compass style={{ width: '14px', height: '14px', color: '#38bdf8' }} />
              <span style={{ fontSize: '12px', color: '#cbd5e1', fontWeight: '500' }}>
                {roadmap ? `${roadmap.title} (${roadmap.total_nodes} nodes, ${roadmap.phases.length} phases)` : 'Interactive Curriculum DAG Available'}
              </span>
            </div>
            <button
              onClick={() => onOpenRoadmapStudio(roadmap?.title || displayText.slice(0, 40))}
              style={{
                padding: '4px 10px',
                borderRadius: '4px',
                backgroundColor: '#0284c7',
                border: 'none',
                color: '#fff',
                fontSize: '11px',
                fontWeight: '600',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}
            >
              <span>Explore in Studio →</span>
            </button>
          </div>
        )}

        {/* Grounded Literature Citations Footer Chips */}
        {citations.length > 0 && (
          <div className="message-citations-footer">
            <span className="citations-footer-label">
              Verified Literature:
            </span>
            <div className="citations-footer-list">
              {citations.map((cite, idx) => (
                <CitationChip
                  key={idx}
                  citation={cite}
                  onSelect={onSelectCitation}
                  inline={false}
                />
              ))}
            </div>
          </div>
        )}

        {!isUser && meta && (
          <div
            className="bubble-meta"
            onClick={() => onSelectMetadata && onSelectMetadata(meta)}
            style={{ cursor: onSelectMetadata ? 'pointer' : 'default' }}
            title="Click to inspect Mamdani FIS & RAG telemetry for this response"
          >
            {isSocratic && (
              <span className="meta-tag" style={{ borderLeft: '2px solid #8b5cf6', color: '#c084fc' }}>
                <GraduationCap style={{ width: '12px', height: '12px', display: 'inline', marginRight: '4px' }} />
                Socratic Mode
              </span>
            )}
            <span className={`meta-tag ${meta.fallback ? 'fallback-tag' : ''}`}>
              {meta.model_name || 'Gemma 4 E2B Instruct'}
            </span>
            <span className="meta-tag">
              intent: {meta.intent}
            </span>
            <span className="meta-tag">
              conf: {(meta.confidence * 100).toFixed(0)}%
            </span>
            {meta.processing_time_ms && (
              <span className="meta-tag">
                {meta.processing_time_ms}ms
              </span>
            )}
            {meta.tokens_generated ? (
              <span className="meta-tag">
                {meta.tokens_generated}{meta.token_budget ? `/${meta.token_budget}` : ''} tokens
              </span>
            ) : null}
            {meta.response_type && (
              <span className="meta-tag" style={{ borderLeft: '2px solid #10b981' }} title={`Demand score: ${Math.round((meta.demand_score || 0) * 100)}%`}>
                {meta.response_type.replace(/_/g, ' ')}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
