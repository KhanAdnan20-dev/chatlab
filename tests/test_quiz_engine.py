import pytest
from backend.app.exceptions import (
    InvalidAnswer,
    InvalidQuizSession,
    QuestionAlreadyAnswered,
    QuizAlreadyFinished,
)
from backend.app.services.quiz_engine import QuizEngine


def test_quiz_engine_start_and_flow():
    engine = QuizEngine()
    state = engine.start_quiz(topic="all", difficulty="all", num_questions=3)
    assert state.session_id is not None
    assert len(state.questions) <= 3
    assert state.completed is False

    curr_q = engine.get_current_question(state.session_id)
    assert curr_q is not None

    # Submit a valid answer automatically (use the correct answer)
    curr_q = engine.get_current_question(state.session_id)
    correct_idx = curr_q.correct_answer if curr_q else 0
    res = engine.submit_answer(state.session_id, correct_idx)
    assert "is_correct" in res
    assert res["user_answer"] == correct_idx

    # Advance
    next_q = engine.next_question(state.session_id)
    # Could be second question or finished
    score = engine.get_score(state.session_id)
    assert "score_percent" in score


def test_quiz_invalid_session():
    engine = QuizEngine()
    with pytest.raises(InvalidQuizSession):
        engine.get_session("non_existent_id")


def test_quiz_invalid_answer_range():
    engine = QuizEngine()
    state = engine.start_quiz(num_questions=1)
    with pytest.raises(InvalidAnswer):
        engine.submit_answer(state.session_id, 999)


def test_quiz_answer_idempotency_correct_answer_repeated():
    """Correct answer submitted once vs repeatedly without advancing."""
    engine = QuizEngine()
    state = engine.start_quiz(num_questions=3)
    curr_q = engine.get_current_question(state.session_id)
    correct_idx = curr_q.correct_answer

    # 1. Correct answer submitted once
    res1 = engine.submit_answer(state.session_id, correct_idx)
    assert res1["is_correct"] is True
    assert state.correct_count == 1

    # 2. Correct answer submitted repeatedly without advancing (e.g. 5 duplicate clicks)
    for _ in range(5):
        res_repeat = engine.submit_answer(state.session_id, correct_idx)
        assert res_repeat["is_correct"] is True
        assert res_repeat["user_answer"] == correct_idx
        # Score must NOT be incremented more than once!
        assert state.correct_count == 1

    score = engine.get_score(state.session_id)
    assert score["correct_count"] == 1


def test_quiz_answer_idempotency_incorrect_answer_repeated():
    """Incorrect answer submitted repeatedly without advancing."""
    engine = QuizEngine()
    state = engine.start_quiz(num_questions=3)
    curr_q = engine.get_current_question(state.session_id)
    wrong_idx = (curr_q.correct_answer + 1) % len(curr_q.options)

    # 1. Incorrect answer submitted
    res1 = engine.submit_answer(state.session_id, wrong_idx)
    assert res1["is_correct"] is False
    assert state.correct_count == 0

    # 2. Incorrect answer resubmitted
    for _ in range(3):
        res_repeat = engine.submit_answer(state.session_id, wrong_idx)
        assert res_repeat["is_correct"] is False
        assert state.correct_count == 0


def test_quiz_advance_and_answer_normally_after_retry():
    """Advancing to next question and answering it normally after retries."""
    engine = QuizEngine()
    state = engine.start_quiz(num_questions=2)
    
    # Question 0: answer correct, retry twice
    q0 = engine.get_current_question(state.session_id)
    engine.submit_answer(state.session_id, q0.correct_answer)
    engine.submit_answer(state.session_id, q0.correct_answer)
    assert state.correct_count == 1

    # Advance to question 1
    q1 = engine.next_question(state.session_id)
    assert q1 is not None
    assert state.current_question_index == 1

    # Question 1: answer correctly
    engine.submit_answer(state.session_id, q1.correct_answer)
    assert state.correct_count == 2

    # Finish quiz
    next_q = engine.next_question(state.session_id)
    assert next_q is None
    assert state.completed is True

    # Final score never exceeds total questions
    score = engine.get_score(state.session_id)
    assert score["correct_count"] == 2
    assert score["total_questions"] == 2
    assert score["score_percent"] == 100.0


def test_quiz_retry_after_simulated_lost_response():
    """Simulated network timeout/retry: client resends exact request and gets identical payload."""
    engine = QuizEngine()
    state = engine.start_quiz(num_questions=2)
    q0 = engine.get_current_question(state.session_id)

    # Attempt 1: server receives and processes
    first_response = engine.submit_answer(state.session_id, q0.correct_answer)

    # Attempt 2: client never got HTTP 200 due to network drop, retries with exact payload
    retry_response = engine.submit_answer(state.session_id, q0.correct_answer)

    assert first_response == retry_response
    assert state.correct_count == 1


def test_quiz_conflicting_answer_rejected():
    """Submitting a conflicting answer to an already answered question raises QuestionAlreadyAnswered."""
    engine = QuizEngine()
    state = engine.start_quiz(num_questions=2)
    q0 = engine.get_current_question(state.session_id)
    wrong_idx = (q0.correct_answer + 1) % len(q0.options)

    # Answer submitted initially
    engine.submit_answer(state.session_id, wrong_idx)

    # Attempt to change answer to correct one after seeing explanation
    with pytest.raises(QuestionAlreadyAnswered):
        engine.submit_answer(state.session_id, q0.correct_answer)

