import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import QuizPanel from '../components/QuizPanel';
import MessageItem from '../components/MessageItem';
import * as api from '../services/api';
import mermaid from 'mermaid';

// Mock the API service for QuizPanel tests
vi.mock('../services/api', () => ({
  startQuiz: vi.fn(),
  submitQuizAnswer: vi.fn(),
  nextQuizQuestion: vi.fn(),
  sendMessage: vi.fn(),
  streamMessage: vi.fn(),
}));

// Mock mermaid library for MessageItem tests
vi.mock('mermaid', () => ({
  default: {
    render: vi.fn().mockImplementation((id, code) =>
      Promise.resolve({ svg: `<svg data-testid="mock-mermaid-svg" data-code="${code}"></svg>` })
    ),
  },
}));

describe('Hotfix Batch 1: Frontend Correctness Regressions', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // =========================================================================
  // 1. QuizPanel Component Tests
  // =========================================================================
  describe('QuizPanel component mount and state transitions', () => {
    it('mounts without throwing ReferenceError and renders initial setup state', () => {
      render(<QuizPanel />);

      // Verify header and description
      expect(screen.getByText(/AIML Knowledge Quiz/i)).toBeInTheDocument();
      expect(screen.getByText(/Test your knowledge across Supervised Learning/i)).toBeInTheDocument();

      // Verify controls
      expect(screen.getByRole('button', { name: /Start Quiz/i })).toBeInTheDocument();
      expect(screen.getByDisplayValue('All Topics')).toBeInTheDocument();
      expect(screen.getByDisplayValue('Medium')).toBeInTheDocument();
      expect(screen.getByDisplayValue('5')).toBeInTheDocument();
    });

    it('transitions to active question after successful API startQuiz response', async () => {
      const mockFirstQuestion = {
        question_id: 'q1',
        topic: 'Deep Learning',
        difficulty: 'medium',
        question: 'What is the vanishing gradient problem in deep networks?',
        options: [
          'Gradients become exponentially small in early layers',
          'Weights become infinite',
          'Learning rate drops to zero',
          'Loss function diverges to NaN',
        ],
        current_index: 0,
        total_questions: 3,
      };

      api.startQuiz.mockResolvedValueOnce({
        session_id: 'test_session_123',
        total_questions: 3,
        first_question: mockFirstQuestion,
      });

      render(<QuizPanel />);

      const startButton = screen.getByRole('button', { name: /Start Quiz/i });
      fireEvent.click(startButton);

      // Verify API was called with default settings
      expect(api.startQuiz).toHaveBeenCalledWith('all', 'medium', 5);

      // Verify transition to question display
      await waitFor(() => {
        expect(screen.getByText(mockFirstQuestion.question)).toBeInTheDocument();
      });

      expect(screen.getByText(/Question 1 of 3/i)).toBeInTheDocument();
      expect(screen.getByText(mockFirstQuestion.options[0])).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Submit Answer/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Submit Answer/i })).toBeDisabled();

      // Simulate answer selection and submission
      api.submitQuizAnswer.mockResolvedValueOnce({
        session_id: 'test_session_123',
        question_id: 'q1',
        is_correct: true,
        correct_answer: 0,
        explanation: 'Early layers receive miniscule gradients due to repeated chain-rule multiplications.',
        score: { correct_count: 1, total_questions: 3, score_percent: 33.3 },
      });

      const optionA = screen.getByText(mockFirstQuestion.options[0]);
      fireEvent.click(optionA);

      const submitButton = screen.getByRole('button', { name: /Submit Answer/i });
      expect(submitButton).not.toBeDisabled();
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/✅ Correct!/i)).toBeInTheDocument();
      });

      expect(screen.getByText(/Early layers receive miniscule gradients/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Next Question →/i })).toBeInTheDocument();
    });
  });

  // =========================================================================
  // 2. MessageItem Code Block & Mermaid Rendering Tests
  // =========================================================================
  describe('MessageItem code block and Mermaid parsing', () => {
    it('renders a Python code block visibly without stripping it or invoking Mermaid', () => {
      const pythonMessage = {
        sender: 'bot',
        text: 'Here is how to define a linear layer in PyTorch:\n\n```python\nimport torch.nn as nn\nlayer = nn.Linear(10, 2)\n```\n\nHope this helps!',
      };

      const { container } = render(<MessageItem message={pythonMessage} />);

      // The raw text content should include the Python code
      expect(container.textContent).toContain('import torch.nn as nn');
      expect(container.textContent).toContain('layer = nn.Linear(10, 2)');
      expect(container.textContent).toContain('Here is how to define a linear layer in PyTorch:');
      expect(container.textContent).toContain('Hope this helps!');

      // Mermaid should NOT be invoked
      expect(mermaid.render).not.toHaveBeenCalled();
      expect(screen.queryByTestId('mock-mermaid-svg')).not.toBeInTheDocument();
    });

    it('renders a JavaScript code block visibly without invoking Mermaid', () => {
      const jsMessage = {
        sender: 'bot',
        text: 'Example JavaScript function:\n\n```javascript\nfunction softmax(logits) {\n  const exps = logits.map(Math.exp);\n  return exps;\n}\n```',
      };

      const { container } = render(<MessageItem message={jsMessage} />);

      expect(container.textContent).toContain('function softmax(logits)');
      expect(container.textContent).toContain('const exps = logits.map(Math.exp);');
      expect(mermaid.render).not.toHaveBeenCalled();
    });

    it('renders a Bash code block visibly without invoking Mermaid', () => {
      const bashMessage = {
        sender: 'bot',
        text: 'Run the following command:\n\n```bash\npip install torch transformers faiss-cpu\n```',
      };

      const { container } = render(<MessageItem message={bashMessage} />);

      expect(container.textContent).toContain('pip install torch transformers faiss-cpu');
      expect(mermaid.render).not.toHaveBeenCalled();
    });

    it('handles an explicitly tagged Mermaid block through the Mermaid rendering path', async () => {
      const mermaidMessage = {
        sender: 'bot',
        text: 'Here is the pipeline architecture:\n\n```mermaid\ngraph TD\n  A[Input] --> B[RAG]\n  B --> C[Gemma]\n```',
      };

      render(<MessageItem message={mermaidMessage} />);

      await waitFor(() => {
        expect(mermaid.render).toHaveBeenCalled();
      });

      // Check that the Mermaid SVG rendered
      expect(screen.getByTestId('mock-mermaid-svg')).toBeInTheDocument();
      // Prose should remain
      expect(screen.getByText(/Here is the pipeline architecture:/i)).toBeInTheDocument();
    });

    it('never removes ordinary code blocks from the message display', () => {
      const untaggedCodeMessage = {
        sender: 'bot',
        text: '```\nx = 10\ny = 20\nprint(x + y)\n```',
      };

      const { container } = render(<MessageItem message={untaggedCodeMessage} />);

      expect(container.textContent).toContain('x = 10');
      expect(container.textContent).toContain('y = 20');
      expect(mermaid.render).not.toHaveBeenCalled();
    });

    it('preserves mixed prose and multiple fenced blocks intact', () => {
      const multiBlockMessage = {
        sender: 'bot',
        text: 'First step is installation:\n```bash\npip install scikit-learn\n```\nThen in your Python script:\n```python\nfrom sklearn.svm import LinearSVC\nclf = LinearSVC()\n```\nAll done!',
      };

      const { container } = render(<MessageItem message={multiBlockMessage} />);

      expect(container.textContent).toContain('First step is installation:');
      expect(container.textContent).toContain('pip install scikit-learn');
      expect(container.textContent).toContain('Then in your Python script:');
      expect(container.textContent).toContain('from sklearn.svm import LinearSVC');
      expect(container.textContent).toContain('clf = LinearSVC()');
      expect(container.textContent).toContain('All done!');
      expect(mermaid.render).not.toHaveBeenCalled();
    });

    it('renders both Mermaid flowchart and surrounding Python code when both are present', async () => {
      const combinedMessage = {
        sender: 'bot',
        text: 'System overview:\n```mermaid\ngraph TD\n  A --> B\n```\nImplementation:\n```python\ndef execute():\n    return True\n```',
      };

      const { container } = render(<MessageItem message={combinedMessage} />);

      // Mermaid was triggered for the mermaid block
      await waitFor(() => {
        expect(mermaid.render).toHaveBeenCalled();
      });

      // Python code remains in displayText
      expect(container.textContent).toContain('Implementation:');
      expect(container.textContent).toContain('def execute():');
      expect(container.textContent).toContain('return True');
    });
  });
});
