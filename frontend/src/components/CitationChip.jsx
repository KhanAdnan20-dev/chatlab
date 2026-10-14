import React, { useRef } from 'react';
import { BookOpen } from 'lucide-react';

/**
 * Interactive Citation Chip badge for authoritative RAG literature citations.
 * Keyboard accessible, ARIA compliant, and styled to integrate seamlessly into both
 * inline message text and footer citation lists.
 */
export default function CitationChip({
  citation,
  onSelect,
  inline = false,
  label,
  className = '',
}) {
  const chipRef = useRef(null);

  if (!citation) return null;

  // Extract clean source tag, e.g. "[S1]" -> "S1"
  const rawCode = citation.citation_code || 'Ref';
  const displayLabel = label || rawCode.replace(/[\[\]]/g, '');

  const handleClick = (e) => {
    e.stopPropagation();
    if (onSelect) {
      onSelect(citation, chipRef.current);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick(e);
    }
  };

  const pageInfo = citation.page_start
    ? citation.page_start === citation.page_end
      ? `PDF p. ${citation.page_start}`
      : `PDF pp. ${citation.page_start}–${citation.page_end}`
    : '';

  const accessibleName = `Citation ${displayLabel}: ${citation.source_title || 'Unknown Source'}${pageInfo ? `, ${pageInfo}` : ''}`;

  return (
    <button
      ref={chipRef}
      type="button"
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      className={`citation-chip ${inline ? 'citation-chip-inline' : 'citation-chip-pill'} ${className}`}
      aria-haspopup="dialog"
      aria-label={accessibleName}
      title={`${citation.source_title || 'Source'}${pageInfo ? ` (${pageInfo})` : ''} — Click to inspect verified literature`}
    >
      <BookOpen className="chip-icon" aria-hidden="true" />
      <span className="chip-code">{displayLabel}</span>
      {!inline && citation.source_title && (
        <span className="chip-title-preview">
          {citation.source_title.length > 24
            ? `${citation.source_title.slice(0, 24)}...`
            : citation.source_title}
        </span>
      )}
      {!inline && pageInfo && (
        <span className="chip-page-preview">{pageInfo}</span>
      )}
    </button>
  );
}
