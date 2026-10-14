import React from 'react';

export default function QuizScore({ score, onRestart }) {
  if (!score) return null;

  const pct = score.score_percent;
  let gradeClass = 'score-good';
  if (pct < 50) gradeClass = 'score-low';
  else if (pct < 80) gradeClass = 'score-mid';

  return (
    <div className="quiz-score-card">
      <h2 className="score-title">🎉 Quiz Completed!</h2>
      <div className={`score-circle ${gradeClass}`}>
        <span className="score-num">{pct}%</span>
      </div>
      <p className="score-details">
        You answered <strong>{score.correct_count}</strong> out of <strong>{score.total_questions}</strong> questions correctly.
      </p>
      <button className="send-btn" onClick={onRestart}>
        Take Another Quiz
      </button>
    </div>
  );
}
