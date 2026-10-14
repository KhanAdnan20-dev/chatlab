import React, { useState } from 'react';
import QuizQuestion from './QuizQuestion';
import QuizOptions from './QuizOptions';
import QuizScore from './QuizScore';
import { startQuiz, submitQuizAnswer, nextQuizQuestion } from '../services/api';

export default function QuizPanel() {
  const [topic, setTopic] = useState('all');
  const [difficulty, setDifficulty] = useState('medium');
  const [numQuestions, setNumQuestions] = useState(5);

  const [sessionId, setSessionId] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [selectedOpt, setSelectedOpt] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [score, setScore] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleStart = async () => {
    setLoading(true);
    setError(null);
    setScore(null);
    setFeedback(null);
    setSelectedOpt(null);

    try {
      const data = await startQuiz(topic, difficulty, numQuestions);
      setSessionId(data.session_id);
      setCurrentQuestion(data.first_question);
    } catch (err) {
      setError(err.message || 'Failed to start quiz');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitAnswer = async () => {
    if (selectedOpt === null || !sessionId || !currentQuestion) return;
    setLoading(true);

    try {
      const res = await submitQuizAnswer(sessionId, selectedOpt);
      setFeedback(res);
    } catch (err) {
      setError(err.message || 'Error submitting answer');
    } finally {
      setLoading(false);
    }
  };

  const handleNext = async () => {
    if (!sessionId) return;
    setLoading(true);
    setSelectedOpt(null);
    setFeedback(null);

    try {
      const res = await nextQuizQuestion(sessionId);
      if (res.completed) {
        setCurrentQuestion(null);
        setScore(res.score);
      } else {
        setCurrentQuestion(res.question);
      }
    } catch (err) {
      setError(err.message || 'Error advancing question');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="quiz-panel-container">
      {!sessionId || score ? (
        <div className="quiz-setup-card">
          <h2>AIML Knowledge Quiz</h2>
          <p>Test your knowledge across Supervised Learning, Deep Learning, Fuzzy Logic, and NLP.</p>

          {score ? (
            <QuizScore score={score} onRestart={() => setSessionId(null)} />
          ) : (
            <div className="quiz-setup-form">
              <div className="form-group">
                <label>Topic:</label>
                <select value={topic} onChange={(e) => setTopic(e.target.value)}>
                  <option value="all">All Topics</option>
                  <option value="Supervised Learning">Supervised Learning</option>
                  <option value="Deep Learning">Deep Learning</option>
                  <option value="Fuzzy Logic">Fuzzy Logic</option>
                  <option value="NLP">NLP</option>
                  <option value="LLM Fine-tuning">LLM Fine-tuning</option>
                </select>
              </div>

              <div className="form-group">
                <label>Difficulty:</label>
                <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
                  <option value="all">All Difficulties</option>
                  <option value="easy">Easy</option>
                  <option value="medium">Medium</option>
                  <option value="hard">Hard</option>
                </select>
              </div>

              <div className="form-group">
                <label>Questions:</label>
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={numQuestions}
                  onChange={(e) => setNumQuestions(parseInt(e.target.value) || 5)}
                />
              </div>

              {error && <div className="quiz-error">{error}</div>}

              <button className="send-btn" onClick={handleStart} disabled={loading}>
                {loading ? 'Starting...' : 'Start Quiz'}
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="quiz-active-card">
          <QuizQuestion question={currentQuestion} />

          {currentQuestion && (
            <QuizOptions
              options={currentQuestion.options}
              selectedIndex={selectedOpt}
              onSelect={setSelectedOpt}
              disabled={!!feedback || loading}
              feedback={feedback}
            />
          )}

          {feedback && (
            <div className={`quiz-feedback-box ${feedback.is_correct ? 'correct' : 'wrong'}`}>
              <p className="feedback-status">
                {feedback.is_correct ? '✅ Correct!' : '❌ Incorrect'}
              </p>
              <p className="explanation-text">{feedback.explanation}</p>
            </div>
          )}

          {error && <div className="quiz-error">{error}</div>}

          <div className="quiz-actions-row">
            {!feedback ? (
              <button
                className="send-btn"
                onClick={handleSubmitAnswer}
                disabled={selectedOpt === null || loading}
              >
                Submit Answer
              </button>
            ) : (
              <button className="send-btn" onClick={handleNext} disabled={loading}>
                Next Question →
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
