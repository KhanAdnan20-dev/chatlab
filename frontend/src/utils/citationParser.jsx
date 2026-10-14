import React from 'react';
import katex from 'katex';
import CitationChip from '../components/CitationChip';

/**
 * Builds a normalized lookup map of citation codes to CitationItem objects.
 * Maps e.g. "S1", "s1", "[S1]" -> citation.
 */
export function buildCitationMap(citations) {
  const map = new Map();
  if (!Array.isArray(citations)) return map;

  citations.forEach((c) => {
    if (c && c.citation_code) {
      const normalized = String(c.citation_code).replace(/[\[\]]/g, '').trim().toUpperCase();
      if (normalized) {
        map.set(normalized, c);
      }
    }
  });

  return map;
}

/**
 * Safely renders LaTeX mathematical notation into KaTeX HTML.
 * Returns null if rendering fails, allowing graceful fallback to raw text.
 */
export function renderKatex(latex, displayMode = false) {
  if (!latex || typeof latex !== 'string') return null;
  const clean = latex.trim();
  if (!clean) return null;

  try {
    return katex.renderToString(clean, {
      displayMode,
      throwOnError: false,
    });
  } catch (err) {
    console.warn('[KaTeX] Render error for expression:', latex, err);
    return null;
  }
}

/**
 * Renders an inline text chunk, handling inline math ($...$ or \(...\))
 * and [S1] citation badges.
 */
function renderInlineContent(text, citationMap, onSelectCitation, blockIdx) {
  if (!text) return null;

  // Match:
  // 1. Inline math: $...$ (at least 1 char between $, no linebreaks)
  // 2. Inline math: \(...\)
  // 3. Citation badges: [S1], [s1], etc.
  const inlineRegex = /(\$(?:[^\$\n\r]+?)\$|\\\((?:[^\)]+?)\\\)|\[[Ss]\d+\])/g;
  const parts = text.split(inlineRegex);

  return parts.map((part, partIdx) => {
    if (!part) return null;

    // Check citation badge
    const citeMatch = part.match(/^\[([Ss]\d+)\]$/);
    if (citeMatch) {
      const code = citeMatch[1].toUpperCase();
      const matchedCitation = citationMap.get(code);
      if (matchedCitation) {
        return (
          <CitationChip
            key={`cite-${blockIdx}-${partIdx}`}
            citation={matchedCitation}
            onSelect={onSelectCitation}
            inline={true}
            label={code}
          />
        );
      }
      return part;
    }

    // Check inline math with $...$
    if (part.startsWith('$') && part.endsWith('$') && part.length > 2) {
      const math = part.slice(1, -1).trim();
      const html = renderKatex(math, false);
      if (html) {
        return (
          <span
            key={`math-inline-${blockIdx}-${partIdx}`}
            className="katex-inline"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        );
      }
    }

    // Check inline math with \(...\)
    if (part.startsWith('\\(') && part.endsWith('\\)') && part.length > 4) {
      const math = part.slice(2, -2).trim();
      const html = renderKatex(math, false);
      if (html) {
        return (
          <span
            key={`math-inline-paren-${blockIdx}-${partIdx}`}
            className="katex-inline"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        );
      }
    }

    // Plain text segment
    return part;
  });
}

/**
 * Parses assistant or user message text:
 * 1. Preserves fenced code blocks verbatim (```...```)
 * 2. Renders display math blocks ($$...$$ and \[...\]) with KaTeX
 * 3. Renders inline math ($...$ and \(...\)) with KaTeX
 * 4. Replaces valid citation markers [S1] with interactive CitationChip components
 */
export function renderTextWithCitations(text, citations, onSelectCitation) {
  if (!text) return null;

  const citationMap = buildCitationMap(citations);

  // Split on fenced code blocks (```...```) AND display math blocks ($$...$$ and \[...\])
  const blockRegex = /(```[\s\S]*?```|\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\])/g;
  const blocks = text.split(blockRegex);

  // Fast-path: If single block with no math syntax and no citations, return plain text
  if (
    blocks.length === 1 &&
    !blocks[0].includes('$') &&
    !blocks[0].includes('\\[') &&
    !blocks[0].includes('\\(') &&
    citationMap.size === 0
  ) {
    return text;
  }

  return blocks.map((block, blockIdx) => {
    if (!block) return null;

    // 1. Fenced code block (preserve verbatim)
    if (block.startsWith('```') && block.endsWith('```')) {
      return <React.Fragment key={`code-${blockIdx}`}>{block}</React.Fragment>;
    }

    // 2. Display math block: $$...$$
    if (block.startsWith('$$') && block.endsWith('$$')) {
      const math = block.slice(2, -2).trim();
      const html = renderKatex(math, true);
      if (html) {
        return (
          <div
            key={`math-display-${blockIdx}`}
            className="katex-display-container"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        );
      }
      return <div key={`math-fallback-${blockIdx}`}>{block}</div>;
    }

    // 3. Display math block: \[...\]
    if (block.startsWith('\\[') && block.endsWith('\\]')) {
      const math = block.slice(2, -2).trim();
      const html = renderKatex(math, true);
      if (html) {
        return (
          <div
            key={`math-display-bracket-${blockIdx}`}
            className="katex-display-container"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        );
      }
      return <div key={`math-fallback-bracket-${blockIdx}`}>{block}</div>;
    }

    // 4. Normal text block with inline math and citations
    return (
      <React.Fragment key={`text-block-${blockIdx}`}>
        {renderInlineContent(block, citationMap, onSelectCitation, blockIdx)}
      </React.Fragment>
    );
  });
}
