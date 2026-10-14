import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import CitationChip from '../components/CitationChip';
import LiteratureDrawer, { isValidHttpUrl } from '../components/LiteratureDrawer';
import { buildCitationMap, renderTextWithCitations } from '../utils/citationParser';
import MessageItem from '../components/MessageItem';

describe('Phase 2B: Citation UI & Literature Drawer Test Suite', () => {
  const sampleCitation1 = {
    citation_code: '[S1]',
    source_title: 'Understanding Deep Learning',
    section: 'PDF p. 229',
    file_name: 'understanding_deep_learning.pdf',
    page_start: 229,
    page_end: 229,
    subject: 'deep_learning',
    score: 0.6961,
    source_url: 'https://udlbook.github.io/udlbook/',
    content_snippet: 'Multi-head attention connects multiple self-attention mechanisms in parallel.',
    formulas: [],
  };

  const sampleCitation2 = {
    citation_code: '[S2]',
    source_title: 'Dive into Deep Learning',
    section: 'PDF p. 221',
    file_name: 'dive_into_deep_learning.pdf',
    page_start: 221,
    page_end: 221,
    subject: 'deep_learning',
    score: 0.6785,
    source_url: null,
    content_snippet: '5.3.3 Backpropagation computes the gradients of the loss function.',
    formulas: ['\\frac{\\partial L}{\\partial W}'],
  };

  // -------------------------------------------------------------------------
  // 1. Citation markers render as chips only when backed by structured metadata
  // -------------------------------------------------------------------------
  it('renders citation markers as chips only when backed by structured metadata', () => {
    const text = 'Multi-head attention connects mechanisms [S1], but [S99] is unknown.';
    const citations = [sampleCitation1];
    const onSelect = vi.fn();

    const { container } = render(<div>{renderTextWithCitations(text, citations, onSelect)}</div>);

    // [S1] should be an interactive button
    const chipButton = screen.getByRole('button', { name: /Citation S1/i });
    expect(chipButton).toBeInTheDocument();
    expect(chipButton).toHaveTextContent('S1');

    // [S99] should remain plain text without a chip
    expect(screen.queryByRole('button', { name: /Citation S99/i })).not.toBeInTheDocument();
    expect(container.textContent).toContain('[S99]');
  });

  // -------------------------------------------------------------------------
  // 2. Clicking a chip opens the correct literature entry
  // -------------------------------------------------------------------------
  it('triggers onSelect with correct citation data when clicked', () => {
    const onSelect = vi.fn();
    render(<CitationChip citation={sampleCitation1} onSelect={onSelect} inline />);

    const chip = screen.getByRole('button', { name: /Citation S1/i });
    fireEvent.click(chip);

    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith(sampleCitation1, expect.anything());
  });

  // -------------------------------------------------------------------------
  // 3. Correct PDF filename and physical page numbers are displayed
  // -------------------------------------------------------------------------
  it('displays correct PDF filename and physical PDF page numbers in drawer', () => {
    const onClose = vi.fn();
    render(
      <LiteratureDrawer
        isOpen={true}
        onClose={onClose}
        citation={sampleCitation1}
      />
    );

    expect(screen.getByText('understanding_deep_learning.pdf')).toBeInTheDocument();
    expect(screen.getByText('PDF Page 229')).toBeInTheDocument();
    expect(screen.getByText(/Physical PDF page index/i)).toBeInTheDocument();
  });

  // -------------------------------------------------------------------------
  // 4. Correct supporting snippet is associated with the selected citation
  // -------------------------------------------------------------------------
  it('displays correct supporting snippet in drawer', () => {
    const onClose = vi.fn();
    render(
      <LiteratureDrawer
        isOpen={true}
        onClose={onClose}
        citation={sampleCitation1}
      />
    );

    expect(screen.getByText(`"${sampleCitation1.content_snippet}"`)).toBeInTheDocument();
  });

  // -------------------------------------------------------------------------
  // 5. Answers without citations render exactly as normal chat messages
  // -------------------------------------------------------------------------
  it('renders answers without citations normally without empty citation sections', () => {
    const msg = {
      id: 1,
      sender: 'bot',
      text: 'Hello, welcome to ChatLab!',
      metadata: { citations: [] },
    };

    const { container } = render(<MessageItem message={msg} />);
    expect(screen.getByText('Hello, welcome to ChatLab!')).toBeInTheDocument();
    expect(container.querySelector('.message-citations-footer')).not.toBeInTheDocument();
  });

  // -------------------------------------------------------------------------
  // 6. Multiple citations are mapped correctly
  // -------------------------------------------------------------------------
  it('maps multiple citations [S1] and [S2] to distinct chips', () => {
    const text = 'Attention [S1] works differently than backpropagation [S2].';
    const citations = [sampleCitation1, sampleCitation2];
    const onSelect = vi.fn();

    render(<div>{renderTextWithCitations(text, citations, onSelect)}</div>);

    const chip1 = screen.getByRole('button', { name: /Citation S1: Understanding Deep Learning/i });
    const chip2 = screen.getByRole('button', { name: /Citation S2: Dive into Deep Learning/i });

    expect(chip1).toBeInTheDocument();
    expect(chip2).toBeInTheDocument();

    fireEvent.click(chip2);
    expect(onSelect).toHaveBeenCalledWith(sampleCitation2, expect.anything());
  });

  // -------------------------------------------------------------------------
  // 7. Citation state remains associated with the correct message after subsequent messages
  // -------------------------------------------------------------------------
  it('associates citations strictly with the corresponding message', () => {
    const onSelect1 = vi.fn();
    const onSelect2 = vi.fn();

    const msg1 = {
      id: 101,
      sender: 'bot',
      text: 'Explanation with [S1].',
      metadata: { citations: [sampleCitation1] },
    };

    const msg2 = {
      id: 102,
      sender: 'bot',
      text: 'Subsequent answer with [S2].',
      metadata: { citations: [sampleCitation2] },
    };

    render(
      <div>
        <MessageItem message={msg1} onSelectCitation={onSelect1} />
        <MessageItem message={msg2} onSelectCitation={onSelect2} />
      </div>
    );

    const chipS1 = screen.getAllByRole('button', { name: /Citation S1/i })[0];
    const chipS2 = screen.getAllByRole('button', { name: /Citation S2/i })[0];

    fireEvent.click(chipS1);
    expect(onSelect1).toHaveBeenCalledWith(sampleCitation1, expect.anything());

    fireEvent.click(chipS2);
    expect(onSelect2).toHaveBeenCalledWith(sampleCitation2, expect.anything());
  });

  // -------------------------------------------------------------------------
  // 8. Unmatched markers and malformed metadata do not crash rendering
  // -------------------------------------------------------------------------
  it('handles unmatched markers and malformed metadata gracefully without crashing', () => {
    const text = 'Unknown marker [S42] and malformed brackets [] or [invalid].';
    const malformedCitations = [null, undefined, { invalid: 'data' }, {}];

    const { container } = render(<div>{renderTextWithCitations(text, malformedCitations, vi.fn())}</div>);
    expect(container.textContent).toContain('[S42]');
    expect(container.textContent).toContain('[] or [invalid]');
  });

  // -------------------------------------------------------------------------
  // 9. Missing and unsafe external URLs are handled safely
  // -------------------------------------------------------------------------
  it('validates HTTP/HTTPS URLs and rejects unsafe or missing URLs', () => {
    expect(isValidHttpUrl('https://example.com')).toBe(true);
    expect(isValidHttpUrl('http://example.com/page')).toBe(true);
    expect(isValidHttpUrl('javascript:alert(1)')).toBe(false);
    expect(isValidHttpUrl('file:///C:/Users/passwords.txt')).toBe(false);
    expect(isValidHttpUrl(null)).toBe(false);
    expect(isValidHttpUrl('')).toBe(false);

    // Render citation with local/missing URL
    render(
      <LiteratureDrawer
        isOpen={true}
        onClose={vi.fn()}
        citation={sampleCitation2}
      />
    );

    expect(screen.getByText(/Indexed from local verified PDF textbook repository/i)).toBeInTheDocument();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });

  // -------------------------------------------------------------------------
  // 10. Drawer close behaviour and keyboard accessibility
  // -------------------------------------------------------------------------
  it('closes drawer on Escape key and close button click, and supports keyboard navigation', () => {
    const onClose = vi.fn();
    const triggerBtn = document.createElement('button');
    document.body.appendChild(triggerBtn);
    const focusSpy = vi.spyOn(triggerBtn, 'focus');

    render(
      <LiteratureDrawer
        isOpen={true}
        onClose={onClose}
        citation={sampleCitation1}
        triggerRef={triggerBtn}
      />
    );

    // 1. Close button click
    const closeBtn = screen.getByRole('button', { name: /Close literature drawer/i });
    fireEvent.click(closeBtn);
    expect(onClose).toHaveBeenCalledTimes(1);

    // 2. Escape key
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(2);

    document.body.removeChild(triggerBtn);
  });

  // -------------------------------------------------------------------------
  // 11. Code blocks preserve verbatim content without citation alteration
  // -------------------------------------------------------------------------
  it('preserves code blocks without converting array brackets to citation chips', () => {
    const text = 'Here is code:\n```python\nx = matrix[S1]\n```\nAnd text [S1].';
    const citations = [sampleCitation1];
    const { container } = render(<div>{renderTextWithCitations(text, citations, vi.fn())}</div>);

    expect(container.textContent).toContain('matrix[S1]');
    // Exactly one citation chip outside the code block
    expect(screen.getAllByRole('button', { name: /Citation S1/i }).length).toBe(1);
  });

  // -------------------------------------------------------------------------
  // 12. Multi-page range display
  // -------------------------------------------------------------------------
  it('displays multi-page bounds correctly as pp. X-Y', () => {
    const multiPageCitation = {
      ...sampleCitation1,
      page_start: 116,
      page_end: 118,
    };

    render(
      <LiteratureDrawer
        isOpen={true}
        onClose={vi.fn()}
        citation={multiPageCitation}
      />
    );

    expect(screen.getByText('PDF Pages 116–118')).toBeInTheDocument();
  });

  // -------------------------------------------------------------------------
  // 13. KaTeX Display Math ($$...$$) Rendering
  // -------------------------------------------------------------------------
  it('renders display math equations ($$...$$) into formatted KaTeX containers', () => {
    const text = 'Here is the output feature formula:\n\n$$\\text{Output Feature Value} = \\sum_{i} (\\text{Weight}_i \\times \\text{Input Pixel}_i)$$\n\nEnd of explanation.';
    const { container } = render(<div>{renderTextWithCitations(text, [], vi.fn())}</div>);

    const displayContainer = container.querySelector('.katex-display-container');
    expect(displayContainer).toBeInTheDocument();
    expect(displayContainer.querySelector('.katex')).toBeInTheDocument();
    expect(displayContainer.querySelector('.katex-display')).toBeInTheDocument();
  });

  // -------------------------------------------------------------------------
  // 14. KaTeX Inline Math ($...$) alongside Citations
  // -------------------------------------------------------------------------
  it('renders inline math ($...$) into KaTeX elements alongside [S1] citation chips', () => {
    const text = 'The query vector $Q$ is scaled by $\\frac{1}{\\sqrt{d_k}}$ as detailed in [S1].';
    const citations = [sampleCitation1];
    const { container } = render(<div>{renderTextWithCitations(text, citations, vi.fn())}</div>);

    // Citations chip present
    expect(screen.getByRole('button', { name: /Citation S1/i })).toBeInTheDocument();

    // Inline math elements present
    const inlineMathElements = container.querySelectorAll('.katex-inline');
    expect(inlineMathElements.length).toBe(2);
  });
});
