import React, { useEffect, useRef } from 'react';
import {
  BookOpen,
  FileText,
  ExternalLink,
  Sigma,
  X,
  Tag,
  CheckCircle2,
  Activity,
  Layers,
} from 'lucide-react';

/**
 * Validates that an external URL begins with http:// or https://.
 * Rejects javascript:, file:, data:, or relative URLs for security.
 */
export function isValidHttpUrl(string) {
  if (!string || typeof string !== 'string') return false;
  try {
    const url = new URL(string.trim());
    return url.protocol === 'http:' || url.protocol === 'https:';
  } catch {
    return false;
  }
}

/**
 * Accessible Literature Drawer for inspecting verified RAG citations.
 * Slides in from the right on desktop, presents as responsive full-height panel on mobile.
 */
export default function LiteratureDrawer({
  isOpen,
  onClose,
  citation,
  triggerRef,
}) {
  const closeButtonRef = useRef(null);
  const drawerRef = useRef(null);

  // Focus management: focus close button when opened, restore focus to trigger chip when closed
  useEffect(() => {
    if (isOpen) {
      const timer = setTimeout(() => {
        closeButtonRef.current?.focus();
      }, 50);

      const handleKeyDown = (e) => {
        if (e.key === 'Escape') {
          e.preventDefault();
          onClose();
        }
      };

      window.addEventListener('keydown', handleKeyDown);
      return () => {
        clearTimeout(timer);
        window.removeEventListener('keydown', handleKeyDown);
      };
    } else if (triggerRef) {
      if (typeof triggerRef.focus === 'function') {
        triggerRef.focus();
      }
    }
  }, [isOpen, onClose, triggerRef]);

  if (!isOpen || !citation) return null;

  const rawCode = citation.citation_code || 'Ref';
  const cleanCode = rawCode.replace(/[\[\]]/g, '');

  const pageStr = citation.page_start
    ? citation.page_start === citation.page_end
      ? `Page ${citation.page_start}`
      : `Pages ${citation.page_start}–${citation.page_end}`
    : null;

  const hasSafeUrl = isValidHttpUrl(citation.source_url);

  return (
    <div
      className="literature-drawer-overlay"
      onClick={onClose}
      role="presentation"
      data-testid="literature-drawer-overlay"
    >
      <div
        ref={drawerRef}
        className="literature-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="literature-drawer-title"
        aria-describedby="literature-drawer-desc"
        onClick={(e) => e.stopPropagation()}
        data-testid="literature-drawer"
      >
        {/* Header */}
        <div className="drawer-header">
          <div className="drawer-header-brand">
            <span className="drawer-code-badge" aria-label={`Citation tag ${cleanCode}`}>
              {cleanCode}
            </span>
            <div>
              <span className="drawer-eyebrow">Verified Literature Citation</span>
              <h2 id="literature-drawer-title" className="drawer-title">
                {citation.source_title || 'Authoritative Source'}
              </h2>
            </div>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            onClick={onClose}
            className="drawer-close-btn"
            aria-label="Close literature drawer"
            title="Close (Esc)"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Scrollable Content */}
        <div className="drawer-body" id="literature-drawer-desc">
          {/* Metadata Cards Grid */}
          <div className="drawer-meta-grid">
            {/* Physical PDF Provenance Card */}
            <div className="meta-card">
              <div className="meta-card-label">
                <FileText className="w-3.5 h-3.5 text-indigo-400" />
                <span>Physical PDF Document</span>
              </div>
              <div className="meta-card-value font-mono">
                {citation.file_name || 'Corpus Document'}
              </div>
              {pageStr && (
                <div className="meta-card-highlight">
                  <span className="font-semibold text-indigo-300">PDF {pageStr}</span>
                  <span className="meta-card-hint">
                    (Physical PDF page index; matches document PDF viewer counter)
                  </span>
                </div>
              )}
            </div>

            {/* Subject Domain Card */}
            {citation.subject && (
              <div className="meta-card">
                <div className="meta-card-label">
                  <Tag className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Subject Domain</span>
                </div>
                <div className="meta-card-value capitalize">
                  {citation.subject.replace(/_/g, ' ')}
                </div>
              </div>
            )}

            {/* Diagnostic Similarity Score */}
            {citation.score != null && (
              <div className="meta-card">
                <div className="meta-card-label">
                  <Activity className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Retrieval Match</span>
                </div>
                <div className="meta-card-value flex items-center gap-1.5">
                  <span className="font-mono text-emerald-300">
                    {typeof citation.score === 'number'
                      ? citation.score.toFixed(4)
                      : citation.score}
                  </span>
                  <span className="text-[11px] text-slate-400">
                    (dense cosine similarity)
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Supporting Evidence Passage */}
          <div className="drawer-section">
            <div className="drawer-section-title">
              <BookOpen className="w-4 h-4 text-indigo-400" />
              <span>Supporting Literature Excerpt</span>
            </div>
            {citation.content_snippet ? (
              <div className="passage-box">
                <p className="passage-text">"{citation.content_snippet}"</p>
              </div>
            ) : (
              <div className="passage-empty">
                No supporting text snippet available for this citation.
              </div>
            )}
          </div>

          {/* Canonical Mathematical Formulas (if available) */}
          {Array.isArray(citation.formulas) && citation.formulas.length > 0 && (
            <div className="drawer-section">
              <div className="drawer-section-title">
                <Sigma className="w-4 h-4 text-amber-400" />
                <span>Canonical Mathematical Formulation</span>
              </div>
              <div className="formulas-list">
                {citation.formulas.map((formula, idx) => (
                  <div key={idx} className="formula-item">
                    <code>{formula}</code>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* External Source Attribution */}
          <div className="drawer-section">
            <div className="drawer-section-title">
              <ExternalLink className="w-4 h-4 text-slate-400" />
              <span>Publisher & Web Resource</span>
            </div>
            {hasSafeUrl ? (
              <a
                href={citation.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="external-link-btn"
                title={`Visit authoritative resource: ${citation.source_url}`}
              >
                <span className="truncate">{citation.source_url}</span>
                <ExternalLink className="w-3.5 h-3.5 flex-shrink-0" />
              </a>
            ) : (
              <div className="local-source-notice">
                <CheckCircle2 className="w-4 h-4 text-indigo-400" />
                <span>Indexed from local verified PDF textbook repository.</span>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="drawer-footer">
          <div className="drawer-footer-tag">
            <Layers className="w-3.5 h-3.5 text-slate-400" />
            <span>ChatLab Grounded RAG</span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="drawer-action-close"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
