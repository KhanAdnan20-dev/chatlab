import React from 'react';

export default function QuizOptions({ options, selectedIndex, onSelect, disabled, feedback }) {
  return (
    <div className="quiz-options-list">
      {options.map((opt, idx) => {
        let btnClass = 'quiz-opt-btn';
        if (selectedIndex === idx) {
          btnClass += ' selected';
        }
        if (feedback) {
          if (idx === feedback.correct_answer) {
            btnClass += ' correct';
          } else if (selectedIndex === idx && !feedback.is_correct) {
            btnClass += ' wrong';
          }
        }
        return (
          <button
            key={idx}
            className={btnClass}
            onClick={() => !disabled && onSelect(idx)}
            disabled={disabled}
          >
            <span className="opt-letter">{String.fromCharCode(65 + idx)}.</span>
            <span className="opt-text">{opt}</span>
          </button>
        );
      })}
    </div>
  );
}
