import React, { useState, useEffect, useRef } from 'react';
import mermaid from 'mermaid';
import { 
  Compass, 
  Download, 
  Copy, 
  Check, 
  X, 
  BookOpen, 
  Sparkles, 
  Layers, 
  Maximize2,
  RefreshCw 
} from 'lucide-react';

mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  securityLevel: 'loose',
  flowchart: {
    useMaxWidth: true,
    htmlLabels: true,
    curve: 'basis'
  }
});

const DOMAINS = [
  { id: 'all', label: 'All AI/ML' },
  { id: 'ai_soft_computing', label: '🎓 AI & Soft Computing (Syllabus)' },
  { id: 'nlp_llms', label: 'NLP & LLMs' },
  { id: 'deep_learning', label: 'Deep Learning' },
  { id: 'computer_vision', label: 'Computer Vision' },
  { id: 'classical_ml', label: 'Classical ML' },
  { id: 'foundations', label: 'Math Foundations' }
];

export default function RoadmapModal({ isOpen, onClose, onAskTopic, initialDomain = 'all' }) {
  const [selectedDomain, setSelectedDomain] = useState(initialDomain);
  const [roadmapData, setRoadmapData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);
  const [svgContent, setSvgContent] = useState('');
  const [selectedNode, setSelectedNode] = useState(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      fetchRoadmap(selectedDomain);
    }
  }, [isOpen, selectedDomain]);

  const fetchRoadmap = async (domain) => {
    setLoading(true);
    setError(null);
    setSelectedNode(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/roadmap/generate?domain=${encodeURIComponent(domain)}`);
      if (!res.ok) throw new Error('Failed to generate roadmap');
      const data = await res.json();
      setRoadmapData(data);
      renderMermaid(data.mermaid_code);
    } catch (err) {
      setError(err.message || 'Error loading roadmap');
    } finally {
      setLoading(false);
    }
  };

  const renderMermaid = async (code) => {
    try {
      const id = `mermaid-svg-${Date.now()}`;
      const { svg } = await mermaid.render(id, code);
      setSvgContent(svg);
    } catch (err) {
      console.error('Mermaid render error:', err);
      setError('Unable to render diagram');
    }
  };

  const handleCopyMermaid = () => {
    if (roadmapData?.mermaid_code) {
      navigator.clipboard.writeText(roadmapData.mermaid_code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleDownloadMarkdown = () => {
    if (!roadmapData?.markdown_content) return;
    const blob = new Blob([roadmapData.markdown_content], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `ChatLab-Roadmap-${roadmapData.domain}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleDownloadSvg = () => {
    if (!svgContent) return;
    const blob = new Blob([svgContent], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `ChatLab-Roadmap-${selectedDomain}.svg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1.25rem',
        backgroundColor: 'rgba(5, 9, 20, 0.85)',
        backdropFilter: 'blur(16px)',
      }}
      onClick={onClose}
    >
      <div
        style={{
          position: 'relative',
          width: '95vw',
          maxWidth: '1280px',
          height: '88vh',
          backgroundColor: '#0a0f1d',
          backgroundImage: 'radial-gradient(circle at 85% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 50%), radial-gradient(circle at 15% 85%, rgba(6, 182, 212, 0.08) 0%, transparent 50%)',
          border: '1px solid rgba(99, 102, 241, 0.35)',
          borderRadius: '20px',
          boxShadow: '0 25px 65px -15px rgba(0, 0, 0, 0.95), 0 0 35px rgba(99, 102, 241, 0.25)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          color: '#f8fafc',
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header Bar */}
        <div style={{
          padding: '1.1rem 1.6rem',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          background: 'linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.95) 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem',
          zIndex: 10,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <div style={{
              width: '42px',
              height: '42px',
              borderRadius: '12px',
              background: 'linear-gradient(135deg, #6366f1 0%, #a855f7 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#fff',
              boxShadow: '0 0 16px rgba(99, 102, 241, 0.4)',
              flexShrink: 0,
            }}>
              <Compass style={{ width: '22px', height: '22px' }} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: '700', color: '#f8fafc', margin: 0 }}>
                  {roadmapData?.title || 'Interactive AI/ML Learning Roadmap'}
                </h2>
                <span style={{
                  fontSize: '0.72rem',
                  fontWeight: '600',
                  padding: '2px 8px',
                  borderRadius: '9999px',
                  backgroundColor: 'rgba(16, 185, 129, 0.18)',
                  color: '#34d399',
                  border: '1px solid rgba(16, 185, 129, 0.35)',
                }}>
                  Topologically Sorted DAG
                </span>
              </div>
              <p style={{ fontSize: '0.8rem', color: '#94a3b8', margin: '3px 0 0 0' }}>
                Prerequisite-aware curriculum generated from verified AI/ML knowledge milestones.
              </p>
            </div>
          </div>

          {/* Action Buttons */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button
              onClick={handleDownloadMarkdown}
              disabled={!roadmapData}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '5px',
                padding: '6px 12px',
                fontSize: '12px',
                fontWeight: '600',
                borderRadius: '8px',
                backgroundColor: '#4f46e5',
                border: '1px solid #6366f1',
                color: '#fff',
                cursor: !roadmapData ? 'not-allowed' : 'pointer',
                opacity: !roadmapData ? 0.5 : 1,
                boxShadow: '0 2px 8px rgba(79, 70, 229, 0.35)',
              }}
              title="Download rich Markdown study guide with checklists and projects"
            >
              <Download style={{ width: '13px', height: '13px' }} />
              <span>Download .MD</span>
            </button>

            <button
              onClick={handleDownloadSvg}
              disabled={!svgContent}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '5px',
                padding: '6px 12px',
                fontSize: '12px',
                fontWeight: '600',
                borderRadius: '8px',
                backgroundColor: '#1e293b',
                border: '1px solid #334155',
                color: '#e2e8f0',
                cursor: !svgContent ? 'not-allowed' : 'pointer',
                opacity: !svgContent ? 0.5 : 1,
              }}
              title="Download high-res vector SVG"
            >
              <Download style={{ width: '13px', height: '13px' }} />
              <span>Download SVG</span>
            </button>

            <button
              onClick={handleCopyMermaid}
              disabled={!roadmapData}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '5px',
                padding: '6px 12px',
                fontSize: '12px',
                fontWeight: '600',
                borderRadius: '8px',
                backgroundColor: '#1e293b',
                border: '1px solid #334155',
                color: '#e2e8f0',
                cursor: !roadmapData ? 'not-allowed' : 'pointer',
                opacity: !roadmapData ? 0.5 : 1,
              }}
              title="Copy Mermaid.js source code"
            >
              {copied ? <Check style={{ width: '13px', height: '13px', color: '#34d399' }} /> : <Copy style={{ width: '13px', height: '13px' }} />}
              <span>{copied ? 'Copied!' : 'Mermaid'}</span>
            </button>

            <button
              onClick={onClose}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '32px',
                height: '32px',
                borderRadius: '8px',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                color: '#94a3b8',
                cursor: 'pointer',
                marginLeft: '6px',
              }}
              title="Close (Esc)"
            >
              <X style={{ width: '18px', height: '18px' }} />
            </button>
          </div>
        </div>

        {/* Domain Filter Bar */}
        <div style={{
          padding: '0.65rem 1.6rem',
          backgroundColor: 'rgba(10, 15, 29, 0.95)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          overflowX: 'auto',
          whiteSpace: 'nowrap',
        }}>
          <span style={{
            fontSize: '11px',
            fontWeight: '700',
            color: '#64748b',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            display: 'flex',
            alignItems: 'center',
            gap: '5px',
            marginRight: '6px',
          }}>
            <Layers style={{ width: '14px', height: '14px', color: '#6366f1' }} />
            Curriculum:
          </span>
          {DOMAINS.map((dom) => {
            const active = selectedDomain === dom.id;
            return (
              <button
                key={dom.id}
                onClick={() => setSelectedDomain(dom.id)}
                style={{
                  padding: '5px 12px',
                  borderRadius: '9999px',
                  fontSize: '12px',
                  fontWeight: active ? '700' : '500',
                  border: active ? '1px solid #818cf8' : '1px solid rgba(255, 255, 255, 0.08)',
                  backgroundColor: active ? '#4f46e5' : '#111827',
                  color: active ? '#ffffff' : '#94a3b8',
                  boxShadow: active ? '0 0 12px rgba(99, 102, 241, 0.45)' : 'none',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                {dom.label}
              </button>
            );
          })}
        </div>

        {/* Main Canvas Body */}
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden', position: 'relative' }}>
          
          {/* Visual Diagram Viewer */}
          <div
            ref={containerRef}
            style={{
              flex: 1,
              overflow: 'auto',
              padding: '1.5rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: '#070b16',
              backgroundImage: 'radial-gradient(circle at 50% 50%, rgba(30, 41, 59, 0.5) 1px, transparent 1px)',
              backgroundSize: '24px 24px',
            }}
          >
            {loading ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem', color: '#94a3b8' }}>
                <RefreshCw style={{ width: '32px', height: '32px', color: '#6366f1', animation: 'spin 1s linear infinite' }} />
                <span style={{ fontSize: '14px', fontWeight: '500' }}>Resolving graph dependencies & rendering flowchart...</span>
              </div>
            ) : error ? (
              <div style={{
                padding: '1rem 1.5rem',
                backgroundColor: 'rgba(127, 29, 29, 0.3)',
                border: '1px solid rgba(239, 68, 68, 0.4)',
                borderRadius: '12px',
                color: '#fca5a5',
                fontSize: '14px',
              }}>
                {error}
              </div>
            ) : (
              <div 
                style={{
                  width: '100%',
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  overflow: 'auto',
                }}
                dangerouslySetInnerHTML={{ __html: svgContent }} 
              />
            )}
          </div>

          {/* Quick Node Navigator / Milestone Sidebar */}
          {roadmapData && (
            <div style={{
              width: '340px',
              borderLeft: '1px solid rgba(255, 255, 255, 0.08)',
              backgroundColor: '#0a0f1e',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}>
              {/* Sidebar Header */}
              <div style={{
                padding: '0.9rem 1.2rem',
                borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
                backgroundColor: '#0d1428',
                fontWeight: '600',
                fontSize: '12px',
                color: '#94a3b8',
                textTransform: 'uppercase',
                letterSpacing: '0.04em',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}>
                <span>Milestone Topics ({roadmapData.total_nodes || roadmapData.nodes?.length || 0})</span>
                <span style={{ color: '#818cf8', fontWeight: '700' }}>{roadmapData.phases?.length || 0} Phases</span>
              </div>

              {/* Scrollable Node Cards */}
              <div style={{ flex: 1, overflowY: 'auto', padding: '0.85rem', display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
                {roadmapData.nodes.map((node) => {
                  const isSelected = selectedNode?.id === node.id;
                  const diffColor =
                    node.difficulty === 'Beginner' ? '#34d399' :
                    node.difficulty === 'Advanced' ? '#c084fc' :
                    '#38bdf8';
                  const diffBg =
                    node.difficulty === 'Beginner' ? 'rgba(16, 185, 129, 0.15)' :
                    node.difficulty === 'Advanced' ? 'rgba(168, 85, 247, 0.15)' :
                    'rgba(56, 189, 248, 0.15)';

                  return (
                    <div
                      key={node.id}
                      onClick={() => setSelectedNode(node)}
                      style={{
                        padding: '10px 12px',
                        borderRadius: '10px',
                        border: isSelected ? '1px solid #6366f1' : '1px solid rgba(255, 255, 255, 0.06)',
                        backgroundColor: isSelected ? 'rgba(99, 102, 241, 0.15)' : '#0d1326',
                        boxShadow: isSelected ? '0 0 14px rgba(99, 102, 241, 0.3)' : 'none',
                        cursor: 'pointer',
                        transition: 'all 0.15s ease',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '6px', marginBottom: '4px' }}>
                        <span style={{ fontWeight: '700', fontSize: '13px', color: '#f1f5f9' }}>{node.label}</span>
                        <span style={{
                          fontSize: '10px',
                          fontFamily: 'monospace',
                          padding: '2px 6px',
                          borderRadius: '4px',
                          color: diffColor,
                          backgroundColor: diffBg,
                          border: `1px solid ${diffColor}40`,
                          fontWeight: '600',
                          flexShrink: 0,
                        }}>
                          {node.difficulty}
                        </span>
                      </div>
                      <p style={{
                        fontSize: '11px',
                        color: '#94a3b8',
                        lineHeight: '1.45',
                        margin: 0,
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                      }}>
                        {node.description}
                      </p>
                    </div>
                  );
                })}
              </div>

              {/* Node Inspector Bottom Drawer if a node is selected */}
              {selectedNode && (
                <div style={{
                  padding: '1rem',
                  borderTop: '1px solid rgba(99, 102, 241, 0.4)',
                  backgroundColor: 'rgba(15, 23, 42, 0.95)',
                }}>
                  <h4 style={{ fontWeight: '700', fontSize: '13px', color: '#818cf8', margin: '0 0 6px 0' }}>
                    {selectedNode.label}
                  </h4>
                  {selectedNode.project && (
                    <div style={{ fontSize: '11px', color: '#cbd5e1', marginBottom: '4px', lineHeight: '1.4' }}>
                      <strong style={{ color: '#94a3b8' }}>Lab Project:</strong> 🛠️ {selectedNode.project}
                    </div>
                  )}
                  {selectedNode.paper && (
                    <div style={{ fontSize: '11px', color: '#cbd5e1', marginBottom: '8px', lineHeight: '1.4' }}>
                      <strong style={{ color: '#94a3b8' }}>Literature:</strong> 📄 {selectedNode.paper}
                    </div>
                  )}
                  <button
                    onClick={() => {
                      if (onAskTopic) {
                        onAskTopic(`Explain the core theoretical intuition, formulas, and architecture for ${selectedNode.label}`);
                        onClose();
                      }
                    }}
                    style={{
                      width: '100%',
                      marginTop: '6px',
                      padding: '8px 12px',
                      borderRadius: '8px',
                      backgroundColor: '#4f46e5',
                      border: '1px solid #6366f1',
                      color: '#ffffff',
                      fontWeight: '600',
                      fontSize: '12px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '6px',
                      boxShadow: '0 2px 10px rgba(79, 70, 229, 0.4)',
                    }}
                  >
                    <Sparkles style={{ width: '14px', height: '14px' }} />
                    <span>Study with Socratic Tutor</span>
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
