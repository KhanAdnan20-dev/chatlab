import React from 'react';

export default function QuizQuestion({ question }) {
  if (!question) return null;

  return (
    <div className="quiz-question-card">
      <div className="quiz-question-meta">
        <span className="quiz-tag">{question.topic}</span>
        <span className="quiz-tag difficulty">{question.difficulty}</span>
        <span className="quiz-progress-num">
          Question {question.current_index + 1} of {question.total_questions}
        </span>
      </div>
      <h3 className="quiz-question-text">{question.question}</h3>
    </div>
  );
}
