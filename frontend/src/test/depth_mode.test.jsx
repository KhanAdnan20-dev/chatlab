import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import App from '../App';

describe('Response Depth Selector & Mamdani Token Decider Suite', () => {
  it('renders the response depth mode dropdown in the chat input area', () => {
    render(<App />);

    const select = screen.getByLabelText(/Response Depth:/i);
    expect(select).toBeInTheDocument();

    // Default value should be auto
    expect(select.value).toBe('auto');

    // Options should include auto, easy, medium, hard, ultra
    const options = screen.getAllByRole('option');
    const optionValues = options.map((opt) => opt.value);
    expect(optionValues).toContain('auto');
    expect(optionValues).toContain('easy');
    expect(optionValues).toContain('medium');
    expect(optionValues).toContain('hard');
    expect(optionValues).toContain('ultra');
  });

  it('updates the depth badge pill and hint text when selecting different depth modes', () => {
    render(<App />);

    const select = screen.getByLabelText(/Response Depth:/i);

    // Change to easy
    fireEvent.change(select, { target: { value: 'easy' } });
    expect(select.value).toBe('easy');
    expect(screen.getByText(/384 Tokens Max/i)).toBeInTheDocument();
    expect(screen.getByText(/Definition-type question with simple real-world intuition/i)).toBeInTheDocument();

    // Change to medium
    fireEvent.change(select, { target: { value: 'medium' } });
    expect(select.value).toBe('medium');
    expect(screen.getByText(/640 Tokens Max/i)).toBeInTheDocument();
    expect(screen.getByText(/Balanced in-depth conceptual analysis with examples/i)).toBeInTheDocument();

    // Change to hard
    fireEvent.change(select, { target: { value: 'hard' } });
    expect(select.value).toBe('hard');
    expect(screen.getByText(/1024 Tokens Max/i)).toBeInTheDocument();
    expect(screen.getByText(/Exhaustive step-by-step mathematical & architectural breakdown/i)).toBeInTheDocument();

    // Change to ultra
    fireEvent.change(select, { target: { value: 'ultra' } });
    expect(select.value).toBe('ultra');
    expect(screen.getByText(/2048 Tokens Max/i)).toBeInTheDocument();
    expect(screen.getByText(/Comprehensive masterclass monograph with complete math derivations, workflows & code/i)).toBeInTheDocument();

    // Change back to auto
    fireEvent.change(select, { target: { value: 'auto' } });
    expect(select.value).toBe('auto');
    expect(screen.getByText(/Dynamic FIS Budget/i)).toBeInTheDocument();
  });
});
