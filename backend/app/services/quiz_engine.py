"""Quiz Engine — independent state machine for interactive AIML quizzes."""
import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.app.config import settings
from backend.app.exceptions import (
    InvalidAnswer,
    InvalidQuizSession,
    QuestionAlreadyAnswered,
    QuizAlreadyFinished,
)

logger = logging.getLogger(__name__)


@dataclass
class QuestionItem:
    id: str
    question: str
    options: list[str]
    correct_answer: int
    explanation: str
    topic: str
    difficulty: str
    co: Optional[str] = None
    unit: Optional[str] = None


@dataclass
class QuizSessionState:
    session_id: str
    topic: str
    difficulty: str
    questions: list[QuestionItem]
    current_question_index: int = 0
    user_answers: list[int] = field(default_factory=list)
    correct_count: int = 0
    score_percent: float = 0.0
    completed: bool = False
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = None


class QuizEngine:
    """In-memory quiz state machine and scoring engine."""

    def __init__(self, data_path: Optional[Path] = None) -> None:
        self.data_path = Path(data_path or settings.QUIZ_DATA_PATH)
        self.question_bank: list[QuestionItem] = []
        self.active_sessions: dict[str, QuizSessionState] = {}
        self.load_questions()

    def load_questions(self) -> None:
        """Loads quiz question bank from JSON file."""
        if not self.data_path.exists():
            logger.warning("Quiz questions file not found at %s", self.data_path)
            return

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            q_list = raw_data.get("questions", [])
            self.question_bank = [
                QuestionItem(
                    id=item["id"],
                    question=item["question"],
                    options=item["options"],
                    correct_answer=item["correct_answer"],
                    explanation=item["explanation"],
                    topic=item.get("topic", "AIML"),
                    difficulty=item.get("difficulty", "medium"),
                    co=item.get("co"),
                    unit=item.get("unit"),
                )
                for item in q_list
            ]
            logger.info("Loaded %d quiz questions.", len(self.question_bank))
        except Exception as exc:
            logger.error("Failed to load quiz questions: %s", exc)

    def start_quiz(
        self,
        topic: str = "all",
        difficulty: str = "medium",
        num_questions: int = 5,
    ) -> QuizSessionState:
        """Initialize a new quiz session."""
        candidates = list(self.question_bank)

        if topic and topic.lower() != "all":
            filtered = [q for q in candidates if q.topic.lower() == topic.lower()]
            if filtered:
                candidates = filtered

        if difficulty and difficulty.lower() != "all":
            filtered = [q for q in candidates if q.difficulty.lower() == difficulty.lower()]
            if filtered:
                candidates = filtered

        selected = random.sample(candidates, min(len(candidates), num_questions))
        session_id = str(uuid.uuid4())

        state = QuizSessionState(
            session_id=session_id,
            topic=topic,
            difficulty=difficulty,
            questions=selected,
        )
        self.active_sessions[session_id] = state
        logger.info("Started quiz session %s (%d questions)", session_id, len(selected))
        return state

    def get_session(self, session_id: str) -> QuizSessionState:
        """Retrieve session or raise InvalidQuizSession."""
        if session_id not in self.active_sessions:
            raise InvalidQuizSession(f"Quiz session '{session_id}' does not exist.")
        return self.active_sessions[session_id]

    def get_current_question(self, session_id: str) -> Optional[QuestionItem]:
        """Get active question object or None if completed."""
        state = self.get_session(session_id)
        if state.completed or state.current_question_index >= len(state.questions):
            return None
        return state.questions[state.current_question_index]

    def submit_answer(self, session_id: str, answer_index: int) -> dict:
        """Submit answer for current question and evaluate result."""
        state = self.get_session(session_id)
        if state.completed:
            raise QuizAlreadyFinished(f"Quiz '{session_id}' has already been completed.")

        curr_q = self.get_current_question(session_id)
        if not curr_q:
            raise QuizAlreadyFinished("No active question in this session.")

        if answer_index < 0 or answer_index >= len(curr_q.options):
            raise InvalidAnswer(
                f"Answer index {answer_index} is invalid for question with {len(curr_q.options)} options."
            )

        # Check if the current question has already been answered
        if len(state.user_answers) > state.current_question_index:
            prev_answer = state.user_answers[state.current_question_index]
            is_correct = (prev_answer == curr_q.correct_answer)
            # Idempotent return for exact answer retry (e.g. lost response, double-click)
            if answer_index == prev_answer:
                return {
                    "session_id": session_id,
                    "question_id": curr_q.id,
                    "user_answer": prev_answer,
                    "correct_answer": curr_q.correct_answer,
                    "is_correct": is_correct,
                    "explanation": curr_q.explanation,
                    "current_question_index": state.current_question_index,
                    "total_questions": len(state.questions),
                }
            raise QuestionAlreadyAnswered(
                f"Question {state.current_question_index + 1} has already been answered with option {prev_answer}."
            )

        is_correct = answer_index == curr_q.correct_answer
        state.user_answers.append(answer_index)
        if is_correct:
            state.correct_count += 1

        return {
            "session_id": session_id,
            "question_id": curr_q.id,
            "user_answer": answer_index,
            "correct_answer": curr_q.correct_answer,
            "is_correct": is_correct,
            "explanation": curr_q.explanation,
            "current_question_index": state.current_question_index,
            "total_questions": len(state.questions),
        }

    def next_question(self, session_id: str) -> Optional[QuestionItem]:
        """Advance to the next question."""
        state = self.get_session(session_id)
        if state.completed:
            return None

        state.current_question_index += 1
        if state.current_question_index >= len(state.questions):
            self.finish_quiz(session_id)
            return None

        return state.questions[state.current_question_index]

    def get_score(self, session_id: str) -> dict:
        """Return score breakdown for session."""
        state = self.get_session(session_id)
        total = len(state.questions)
        pct = (state.correct_count / total * 100.0) if total > 0 else 0.0
        state.score_percent = round(pct, 1)

        return {
            "session_id": session_id,
            "topic": state.topic,
            "difficulty": state.difficulty,
            "total_questions": total,
            "correct_count": state.correct_count,
            "score_percent": state.score_percent,
            "completed": state.completed,
        }

    def finish_quiz(self, session_id: str) -> dict:
        """Mark quiz as completed and calculate final score."""
        state = self.get_session(session_id)
        state.completed = True
        state.finished_at = datetime.now(timezone.utc).isoformat()
        return self.get_score(session_id)


quiz_engine = QuizEngine()
