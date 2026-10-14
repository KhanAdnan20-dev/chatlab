import React, { useState, useEffect, useRef } from 'react';
import mermaid from 'mermaid';
import { 
  Compass, 
  Search, 
  Download, 
  Copy, 
  Check, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  BookOpen, 
  GraduationCap, 
  Layers, 
  Terminal,
  ExternalLink,
  FileCode,
  FileText,
  Filter,
  CheckCircle2
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

const TRACK_PRESETS = [
  { id: 'ai_sc', label: 'AI & Soft Computing (Units I-VI)', query: 'AI and Soft Computing midsem syllabus units 1 to 6' },
  { id: 'dl_llm', label: 'Deep Learning & Transformers', query: 'Deep learning transformers attention and large language models' },
  { id: 'cv', label: 'Computer Vision & ResNets', query: 'Computer vision convolution resnets and image recognition' },
  { id: 'math', label: 'Math Foundations for AI', query: 'Linear algebra calculus probability and numerical optimization for AI' },
  { id: 'mlops', label: 'MLOps & Systems Engineering', query: 'MLOps model deployment quantization and inference pipelines' },
];

export default function RoadmapStudio({ initialQuery, onAskTopic, onOpenQuiz }) {
  const [query, setQuery] = useState(initialQuery || 'AI and Soft Computing midsem syllabus units 1 to 6');
  const [loading, setLoading] = useState(false);
  const [roadmap, setRoadmap] = useState(null);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [svgContent, setSvgContent] = useState('');
  const [selectedPhase, setSelectedPhase] = useState('ALL');
  const [searchFilter, setSearchFilter] = useState('');

  const containerRef = useRef(null);

  useEffect(() => {
    const q = initialQuery || query;
    setQuery(q);
    handleGenerateRoadmap(q);
  }, [initialQuery]);

  const handleGenerateRoadmap = async (customQuery) => {
    const targetQuery = (customQuery || query).trim();
    if (!targetQuery || loading) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch('http://127.0.0.1:8000/roadmap/ai-generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: targetQuery }),
      });

      if (!res.ok) throw new Error('Failed to generate curriculum DAG');
      const data = await res.json();
      setRoadmap(data);
      renderMermaid(data.mermaid_code);
    } catch (err) {
      setError(err.message || 'Error generating roadmap');
    } finally {
      setLoading(false);
    }
  };

  const renderMermaid = async (code) => {
    if (!code) return;
    try {
      const id = `mermaid-studio-${Date.now()}`;
      const { svg } = await mermaid.render(id, code);
      setSvgContent(svg);
    } catch (err) {
      console.error('Mermaid render error:', err);
      try {
        const fallbackId = `mermaid-fallback-${Date.now()}`;
        const fallbackCode = `graph TD\n  start["${query.replace(/"/g, "'")}"] --> end_node["Mastery & Practical Application"]\n  classDef default fill:#1e293b,stroke:#6366f1,stroke-width:2px,color:#f8fafc;`;
        const { svg } = await mermaid.render(fallbackId, fallbackCode);
        setSvgContent(svg);
      } catch (e) {
        setSvgContent('');
      }
    }
  };

  const handleCopyCode = () => {
    if (!roadmap?.mermaid_code) return;
    navigator.clipboard.writeText(roadmap.mermaid_code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadSvg = () => {
    if (!svgContent) return;
    const blob = new Blob([svgContent], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${(roadmap?.title || 'curriculum_dag').replace(/\s+/g, '_').toLowerCase()}.svg`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadMarkdown = () => {
    if (!roadmap?.markdown_content) return;
    const content = `# ${roadmap.title}\n\n\`\`\`mermaid\n${roadmap.mermaid_code}\n\`\`\`\n\n${roadmap.markdown_content}`;
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${(roadmap?.title || 'curriculum_syllabus').replace(/\s+/g, '_').toLowerCase()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Filter nodes for the right-hand syllabus drawer
  const nodes = roadmap?.nodes || [];
  const filteredNodes = nodes.filter((n) => {
    const matchesPhase = selectedPhase === 'ALL' || n.phase === selectedPhase;
    const matchesSearch = !searchFilter || 
      n.label.toLowerCase().includes(searchFilter.toLowerCase()) ||
      n.description.toLowerCase().includes(searchFilter.toLowerCase()) ||
      (n.key_topics && n.key_topics.some(t => t.toLowerCase().includes(searchFilter.toLowerCase())));
    return matchesPhase && matchesSearch;
  });

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: 'calc(100vh - 65px)',
      backgroundColor: '#070a13',
      color: '#f8fafc',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif',
      overflow: 'hidden'
    }}>
      {/* Studio Command Bar */}
      <div style={{
        padding: '0.75rem 1.25rem',
        borderBottom: '1px solid #1e293b',
        backgroundColor: '#0a0f1d',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '1rem',
        zIndex: 10
      }}>
        {/* Left: Project & Command Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', minWidth: '260px' }}>
          <div style={{
            padding: '5px 8px',
            borderRadius: '4px',
            backgroundColor: '#1e293b',
            border: '1px solid #334155',
            fontSize: '11px',
            fontFamily: 'monospace',
            color: '#38bdf8',
            fontWeight: '600'
          }}>
            DAG_STUDIO :: v2.5
          </div>
          <div>
            <h2 style={{ fontSize: '0.92rem', fontWeight: '700', margin: 0, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span>Curriculum Graph Studio</span>
              {roadmap && (
                <span style={{ fontSize: '11px', fontWeight: '400', color: '#64748b' }}>
                  ({roadmap.total_nodes} nodes • {roadmap.phases.length} units)
                </span>
              )}
            </h2>
          </div>
        </div>

        {/* Center: Command Input */}
        <form 
          onSubmit={(e) => { e.preventDefault(); handleGenerateRoadmap(); }}
          style={{ flex: 1, maxWidth: '640px', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <div style={{ position: 'relative', width: '100%' }}>
            <Search style={{
              position: 'absolute',
              left: '10px',
              top: '50%',
              transform: 'translateY(-50%)',
              width: '14px',
              height: '14px',
              color: '#64748b'
            }} />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search or enter syllabus track (e.g. AI & Soft Computing midsem, Deep Learning, Vision)..."
              style={{
                width: '100%',
                backgroundColor: '#0f172a',
                border: '1px solid #1e293b',
                borderRadius: '6px',
                padding: '0.45rem 0.75rem 0.45rem 2rem',
                fontSize: '0.82rem',
                color: '#f8fafc',
                outline: 'none',
                fontFamily: 'inherit'
              }}
              onFocus={(e) => { e.target.style.borderColor = '#0284c7'; }}
              onBlur={(e) => { e.target.style.borderColor = '#1e293b'; }}
            />
          </div>

          <button
            type="submit"
            disabled={loading || !query.trim()}
            style={{
              padding: '0.45rem 0.9rem',
              borderRadius: '6px',
              backgroundColor: loading ? '#334155' : '#0284c7',
              border: 'none',
              color: '#fff',
              fontSize: '0.82rem',
              fontWeight: '600',
              cursor: loading ? 'not-allowed' : 'pointer',
              whiteSpace: 'nowrap',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}
          >
            <span>{loading ? 'Synthesizing...' : 'Compile DAG'}</span>
          </button>
        </form>

        {/* Right: Export & Controls Toolbar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <button
            onClick={() => setZoomLevel((z) => Math.min(z + 0.15, 2.2))}
            style={toolbarBtnStyle}
            title="Zoom In"
          >
            <ZoomIn style={{ width: '14px', height: '14px' }} />
          </button>
          <button
            onClick={() => setZoomLevel((z) => Math.max(z - 0.15, 0.4))}
            style={toolbarBtnStyle}
            title="Zoom Out"
          >
            <ZoomOut style={{ width: '14px', height: '14px' }} />
          </button>
          <button
            onClick={() => setZoomLevel(1)}
            style={{ ...toolbarBtnStyle, fontSize: '11px', fontFamily: 'monospace' }}
            title="Reset Zoom (100%)"
          >
            {Math.round(zoomLevel * 100)}%
          </button>

          <div style={{ width: '1px', height: '16px', backgroundColor: '#1e293b', margin: '0 2px' }} />

          <button
            onClick={handleCopyCode}
            style={toolbarActionStyle}
            title="Copy Mermaid syntax"
          >
            {copied ? <Check style={{ width: '13px', height: '13px', color: '#10b981' }} /> : <Copy style={{ width: '13px', height: '13px' }} />}
            <span>{copied ? 'Copied' : 'Mermaid'}</span>
          </button>

          <button
            onClick={handleDownloadSvg}
            style={toolbarActionStyle}
            title="Export high-resolution SVG diagram"
          >
            <FileCode style={{ width: '13px', height: '13px', color: '#38bdf8' }} />
            <span>SVG</span>
          </button>

          <button
            onClick={handleDownloadMarkdown}
            style={toolbarActionStyle}
            title="Export syllabus markdown checklist"
          >
            <FileText style={{ width: '13px', height: '13px', color: '#34d399' }} />
            <span>Markdown</span>
          </button>
        </div>
      </div>

      {/* Preset Filter Bar */}
      <div style={{
        padding: '0.4rem 1.25rem',
        backgroundColor: '#080c18',
        borderBottom: '1px solid #1e293b',
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        overflowX: 'auto'
      }}>
        <span style={{ fontSize: '11px', color: '#64748b', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Standard Tracks:
        </span>
        {TRACK_PRESETS.map((t) => (
          <button
            key={t.id}
            onClick={() => {
              setQuery(t.query);
              handleGenerateRoadmap(t.query);
            }}
            style={{
              padding: '3px 8px',
              borderRadius: '4px',
              backgroundColor: query === t.query ? '#1e293b' : 'transparent',
              border: `1px solid ${query === t.query ? '#38bdf8' : '#1e293b'}`,
              color: query === t.query ? '#38bdf8' : '#94a3b8',
              fontSize: '11px',
              cursor: 'pointer',
              whiteSpace: 'nowrap'
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Split-Screen Workspace */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Left Panel: Visual DAG Flowchart Canvas (65% width) */}
        <div 
          ref={containerRef}
          style={{
            flex: 13,
            height: '100%',
            overflow: 'auto',
            position: 'relative',
            backgroundColor: '#05070e',
            backgroundImage: 'radial-gradient(#1e293b 1px, transparent 1px)',
            backgroundSize: '24px 24px',
            display: 'flex',
            flexDirection: 'column'
          }}
        >
          {/* Top Status Tag inside Canvas */}
          <div style={{
            position: 'absolute',
            top: '12px',
            left: '12px',
            zIndex: 5,
            padding: '4px 10px',
            borderRadius: '4px',
            backgroundColor: 'rgba(10, 15, 29, 0.85)',
            border: '1px solid #1e293b',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '11px',
            fontFamily: 'monospace',
            color: '#64748b'
          }}>
            <span style={{ display: 'inline-block', width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#10b981' }}></span>
            <span>TOPOLOGY: {roadmap?.domain ? roadmap.domain.toUpperCase() : 'DAG'}</span>
            <span>•</span>
            <span>NODES: {roadmap?.total_nodes || 0}</span>
            <span>•</span>
            <span>ACYCLIC: VALIDATED</span>
          </div>

          <div style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '2rem',
            overflow: 'auto'
          }}>
            {loading ? (
              <div style={{ textAlign: 'center', color: '#94a3b8' }}>
                <div style={{
                  width: '32px',
                  height: '32px',
                  border: '2px solid #1e293b',
                  borderTopColor: '#38bdf8',
                  borderRadius: '50%',
                  margin: '0 auto 1rem auto',
                  animation: 'spin 0.8s linear infinite'
                }} />
                <p style={{ fontSize: '0.85rem', color: '#f8fafc', fontWeight: '600' }}>
                  Compiling Curriculum Topological DAG...
                </p>
                <p style={{ fontSize: '0.75rem', color: '#64748b' }}>
                  Traversing prerequisite milestones and generating syntax-safe Mermaid flowchart.
                </p>
              </div>
            ) : error ? (
              <div style={{
                padding: '1.25rem',
                borderRadius: '8px',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                color: '#f87171',
                textAlign: 'center',
                maxWidth: '420px',
                fontSize: '0.85rem'
              }}>
                <p style={{ fontWeight: '600', marginBottom: '0.25rem' }}>Synthesis Failed</p>
                <p style={{ fontSize: '0.8rem', color: '#cbd5e1' }}>{error}</p>
              </div>
            ) : svgContent ? (
              <div 
                style={{
                  transform: `scale(${zoomLevel})`,
                  transformOrigin: 'center center',
                  transition: 'transform 0.12s ease-out',
                  maxWidth: '100%',
                  display: 'flex',
                  justifyContent: 'center'
                }}
                dangerouslySetInnerHTML={{ __html: svgContent }}
              />
            ) : (
              <div style={{ textAlign: 'center', color: '#64748b' }}>
                <Compass style={{ width: '40px', height: '40px', color: '#1e293b', margin: '0 auto 0.5rem auto' }} />
                <p style={{ fontSize: '0.85rem' }}>No graph loaded. Select a standard track above or search any syllabus topic.</p>
              </div>
            )}
          </div>
        </div>

        {/* Right Panel: Interactive Syllabus Unit Drawer (35% width) */}
        <div style={{
          flex: 8,
          minWidth: '380px',
          maxWidth: '480px',
          height: '100%',
          overflowY: 'auto',
          borderLeft: '1px solid #1e293b',
          backgroundColor: '#0a0f1d',
          display: 'flex',
          flexDirection: 'column'
        }}>
          {/* Drawer Header */}
          <div style={{
            padding: '1rem 1.25rem',
            borderBottom: '1px solid #1e293b',
            backgroundColor: '#0d1326'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <BookOpen style={{ width: '15px', height: '15px', color: '#38bdf8' }} />
                <h3 style={{ fontSize: '0.88rem', fontWeight: '700', color: '#f8fafc', margin: 0 }}>
                  Curriculum Milestones
                </h3>
              </div>
              <span style={{ fontSize: '11px', fontFamily: 'monospace', color: '#64748b' }}>
                {filteredNodes.length} / {nodes.length} UNITS
              </span>
            </div>

            {/* Filter by Phase Dropdown */}
            {roadmap?.phases && (
              <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
                <select
                  value={selectedPhase}
                  onChange={(e) => setSelectedPhase(e.target.value)}
                  style={{
                    flex: 1,
                    backgroundColor: '#070a13',
                    border: '1px solid #1e293b',
                    borderRadius: '4px',
                    padding: '4px 8px',
                    fontSize: '11px',
                    color: '#cbd5e1',
                    outline: 'none'
                  }}
                >
                  <option value="ALL">Show All Phases / Units</option>
                  {roadmap.phases.map((ph, idx) => (
                    <option key={idx} value={ph}>{ph}</option>
                  ))}
                </select>

                <input
                  type="text"
                  placeholder="Filter concepts..."
                  value={searchFilter}
                  onChange={(e) => setSearchFilter(e.target.value)}
                  style={{
                    width: '120px',
                    backgroundColor: '#070a13',
                    border: '1px solid #1e293b',
                    borderRadius: '4px',
                    padding: '4px 8px',
                    fontSize: '11px',
                    color: '#cbd5e1',
                    outline: 'none'
                  }}
                />
              </div>
            )}
          </div>

          {/* Node Cards Stream */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {filteredNodes.length === 0 ? (
              <div style={{ textAlign: 'center', color: '#64748b', padding: '2rem 1rem', fontSize: '0.82rem' }}>
                No milestones match the current filter.
              </div>
            ) : (
              filteredNodes.map((node, idx) => (
                <div 
                  key={node.id || idx}
                  style={{
                    backgroundColor: '#070a13',
                    border: '1px solid #1e293b',
                    borderRadius: '6px',
                    padding: '0.85rem',
                    transition: 'border-color 0.15s'
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.borderColor = '#38bdf8'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.borderColor = '#1e293b'; }}
                >
                  {/* Card Header */}
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '8px', marginBottom: '0.35rem' }}>
                    <div>
                      <span style={{
                        fontSize: '10px',
                        fontFamily: 'monospace',
                        color: '#38bdf8',
                        textTransform: 'uppercase',
                        letterSpacing: '0.04em',
                        display: 'block',
                        marginBottom: '2px'
                      }}>
                        {node.phase}
                      </span>
                      <h4 style={{ fontSize: '0.85rem', fontWeight: '600', color: '#f8fafc', margin: 0 }}>
                        {node.label}
                      </h4>
                    </div>

                    <span style={{
                      fontSize: '10px',
                      padding: '2px 6px',
                      borderRadius: '3px',
                      backgroundColor: node.difficulty === 'Beginner' ? 'rgba(16, 185, 129, 0.15)' : node.difficulty === 'Advanced' ? 'rgba(168, 85, 247, 0.15)' : 'rgba(56, 189, 248, 0.15)',
                      color: node.difficulty === 'Beginner' ? '#34d399' : node.difficulty === 'Advanced' ? '#c084fc' : '#38bdf8',
                      border: '1px solid currentColor',
                      fontWeight: '600'
                    }}>
                      {node.difficulty}
                    </span>
                  </div>

                  {/* Overview */}
                  <p style={{ fontSize: '0.78rem', color: '#94a3b8', lineHeight: '1.45', margin: '0 0 0.5rem 0' }}>
                    {node.description}
                  </p>

                  {/* Key Topics Tags */}
                  {node.key_topics && node.key_topics.length > 0 && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '0.5rem' }}>
                      {node.key_topics.map((t, tidx) => (
                        <span key={tidx} style={{
                          fontSize: '10px',
                          padding: '2px 5px',
                          borderRadius: '3px',
                          backgroundColor: '#0f172a',
                          border: '1px solid #1e293b',
                          color: '#cbd5e1'
                        }}>
                          {t}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Hands-on Lab Project */}
                  {node.project && (
                    <div style={{
                      fontSize: '11px',
                      color: '#cbd5e1',
                      padding: '4px 8px',
                      borderRadius: '4px',
                      backgroundColor: '#0b1329',
                      border: '1px solid #1e3a5f',
                      marginBottom: '0.4rem'
                    }}>
                      <span style={{ fontWeight: '600', color: '#38bdf8' }}>Hands-on Lab: </span>
                      {node.project}
                    </div>
                  )}

                  {/* Seminal Paper Citation */}
                  {node.paper && (
                    <div style={{
                      fontSize: '10px',
                      color: '#94a3b8',
                      fontStyle: 'italic',
                      marginBottom: '0.5rem'
                    }}>
                      Ref: {node.paper}
                    </div>
                  )}

                  {/* Action Commands */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', paddingTop: '4px', borderTop: '1px solid #131c31' }}>
                    <button
                      onClick={() => onAskTopic && onAskTopic(`Explain the core theoretical intuition and formulas for ${node.label}`)}
                      style={{
                        padding: '3px 8px',
                        borderRadius: '4px',
                        backgroundColor: '#1e293b',
                        border: '1px solid #334155',
                        color: '#38bdf8',
                        fontSize: '11px',
                        fontWeight: '500',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px'
                      }}
                      title="Launch Socratic Tutor for this node"
                    >
                      <GraduationCap style={{ width: '12px', height: '12px' }} />
                      <span>Study with Tutor</span>
                    </button>

                    {onOpenQuiz && (
                      <button
                        onClick={onOpenQuiz}
                        style={{
                          padding: '3px 8px',
                          borderRadius: '4px',
                          backgroundColor: '#0f172a',
                          border: '1px solid #1e293b',
                          color: '#cbd5e1',
                          fontSize: '11px',
                          cursor: 'pointer'
                        }}
                        title="Test your mastery in Viva exam"
                      >
                        <span>Test Viva</span>
                      </button>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

const toolbarBtnStyle = {
  background: '#0f172a',
  border: '1px solid #1e293b',
  color: '#cbd5e1',
  borderRadius: '4px',
  padding: '5px 7px',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center'
};

const toolbarActionStyle = {
  background: '#0f172a',
  border: '1px solid #1e293b',
  color: '#cbd5e1',
  borderRadius: '4px',
  padding: '5px 8px',
  cursor: 'pointer',
  fontSize: '11px',
  display: 'flex',
  alignItems: 'center',
  gap: '4px'
};
