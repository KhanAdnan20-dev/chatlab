import React, { useState } from 'react';
import { Brain, ChevronDown, ChevronRight, Sparkles } from 'lucide-react';

export default function ThinkingBox({ thinkingText, isStreaming, hasAnswerStarted }) {
  // If streaming and answer hasn't started yet, keep it open so user sees reasoning live
  const [isExpanded, setIsExpanded] = useState(true);

  if (!thinkingText) return null;

  const isLiveThinking = isStreaming && !hasAnswerStarted;

  return (
    <div style={{
      marginBottom: '0.85rem',
      borderRadius: '8px',
      backgroundColor: '#0c0a1d',
      border: '1px solid #3730a3',
      overflow: 'hidden',
      transition: 'all 0.2s ease',
    }}>
      {/* Header / Toggle Button */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 12px',
          backgroundColor: '#131034',
          border: 'none',
          borderBottom: isExpanded ? '1px solid #312e81' : 'none',
          color: '#a5b4fc',
          fontSize: '12px',
          fontWeight: '500',
          cursor: 'pointer',
          textAlign: 'left',
          userSelect: 'none',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Brain style={{
            width: '14px',
            height: '14px',
            color: '#818cf8',
            animation: isLiveThinking ? 'pulse 1.5s infinite' : 'none',
          }} />
          <span style={{ color: '#e0e7ff', fontWeight: '600' }}>
            {isLiveThinking ? 'Thinking step-by-step...' : 'Reasoning Process'}
          </span>
          {isLiveThinking && (
            <span style={{
              display: 'inline-block',
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              backgroundColor: '#a855f7',
              animation: 'pulse 1s infinite'
            }} />
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#818cf8', fontSize: '11px' }}>
          <span>{isExpanded ? 'Hide' : 'Show logic'}</span>
          {isExpanded ? (
            <ChevronDown style={{ width: '14px', height: '14px' }} />
          ) : (
            <ChevronRight style={{ width: '14px', height: '14px' }} />
          )}
        </div>
      </button>

      {/* Expanded Reasoning Trace */}
      {isExpanded && (
        <div style={{
          padding: '12px 14px',
          fontSize: '12px',
          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
          lineHeight: '1.6',
          color: '#c7d2fe',
          backgroundColor: '#0c0a1d',
          maxHeight: '320px',
          overflowY: 'auto',
          whiteSpace: 'pre-wrap',
          borderLeft: '2px solid #6366f1',
        }}>
          {thinkingText}
          {isLiveThinking && (
            <span style={{
              display: 'inline-block',
              width: '6px',
              height: '12px',
              backgroundColor: '#818cf8',
              marginLeft: '4px',
              verticalAlign: 'middle',
              animation: 'pulse 0.8s infinite'
            }} />
          )}
        </div>
      )}
    </div>
  );
}
