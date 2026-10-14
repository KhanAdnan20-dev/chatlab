"""Thin FastAPI Router for Quiz Engine endpoints."""
from fastapi import APIRouter, status
from backend.app.exceptions import ChatLabError, to_http
from backend.app.schemas import (
    AnswerSubmitRequest,
    AnswerSubmitResponse,
    PublicQuizQuestion,
    QuizScoreResponse,
    QuizStartRequest,
    QuizStartResponse,
)
from backend.app.services.quiz_engine import quiz_engine

router = APIRouter(prefix="/quiz", tags=["Quiz"])


def _to_public_q(q_item, index: int, total: int) -> PublicQuizQuestion:
    return PublicQuizQuestion(
        id=q_item.id,
        question=q_item.question,
        options=q_item.options,
        topic=q_item.topic,
        difficulty=q_item.difficulty,
        current_index=index,
        total_questions=total,
    )


@router.get(
    "/questions",
    summary="Get All Available Quiz Questions",
    description="Returns the full question bank with topic, difficulty, CO tag, and options.",
)
def get_all_questions():
    questions = [
        {
            "id": q.id,
            "question": q.question,
            "options": q.options,
            "correct_answer": q.correct_answer,
            "explanation": q.explanation,
            "topic": q.topic,
            "difficulty": q.difficulty,
            "co": q.co,
            "unit": q.unit,
        }
        for q in quiz_engine.question_bank
    ]
    return {"total": len(questions), "questions": questions}


@router.post(
    "/start",
    response_model=QuizStartResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start New Quiz Session",
)
def start_quiz(payload: QuizStartRequest) -> QuizStartResponse:
    state = quiz_engine.start_quiz(
        topic=payload.topic,
        difficulty=payload.difficulty,
        num_questions=payload.num_questions,
    )
    first_q = quiz_engine.get_current_question(state.session_id)
    public_q = _to_public_q(first_q, 0, len(state.questions)) if first_q else None

    return QuizStartResponse(
        session_id=state.session_id,
        topic=state.topic,
        difficulty=state.difficulty,
        total_questions=len(state.questions),
        first_question=public_q,
    )


@router.post(
    "/{session_id}/answer",
    response_model=AnswerSubmitResponse,
    summary="Submit Question Answer",
)
def submit_answer(session_id: str, payload: AnswerSubmitRequest) -> AnswerSubmitResponse:
    try:
        res = quiz_engine.submit_answer(session_id, payload.answer_index)
        return AnswerSubmitResponse(**res)
    except ChatLabError as exc:
        raise to_http(exc)


@router.post(
    "/{session_id}/next",
    response_model=dict,
    summary="Advance to Next Question",
)
def next_question(session_id: str):
    try:
        q_next = quiz_engine.next_question(session_id)
        state = quiz_engine.get_session(session_id)
        if not q_next:
            return {"completed": True, "question": None, "score": quiz_engine.get_score(session_id)}
        public_q = _to_public_q(q_next, state.current_question_index, len(state.questions))
        return {"completed": False, "question": public_q.model_dump(), "score": None}
    except ChatLabError as exc:
        raise to_http(exc)


@router.get(
    "/{session_id}/score",
    response_model=QuizScoreResponse,
    summary="Get Quiz Score",
)
def get_score(session_id: str) -> QuizScoreResponse:
    try:
        res = quiz_engine.get_score(session_id)
        return QuizScoreResponse(**res)
    except ChatLabError as exc:
        raise to_http(exc)


@router.post(
    "/{session_id}/finish",
    response_model=QuizScoreResponse,
    summary="Finish Quiz Session",
)
def finish_quiz(session_id: str) -> QuizScoreResponse:
    try:
        res = quiz_engine.finish_quiz(session_id)
        return QuizScoreResponse(**res)
    except ChatLabError as exc:
        raise to_http(exc)
