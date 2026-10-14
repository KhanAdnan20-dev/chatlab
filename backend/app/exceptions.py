"""Application-level exceptions mapped to HTTP error codes at the API boundary."""
from fastapi import HTTPException, status


class ChatLabError(Exception):
    """Base ChatLab exception."""


class ModelUnavailable(ChatLabError):
    """Raised when a requested model is not loaded."""


class InvalidQuizSession(ChatLabError):
    """Raised when a quiz session_id does not exist."""


class QuizAlreadyFinished(ChatLabError):
    """Raised when an action targets a completed quiz session."""


class InvalidAnswer(ChatLabError):
    """Raised when an answer index is out of bounds."""


class QuestionAlreadyAnswered(ChatLabError):
    """Raised when an answer is submitted for an already answered question with a conflicting index."""


def to_http(exc: ChatLabError) -> HTTPException:
    """Convert ChatLabError subclass to a FastAPI HTTPException."""
    if isinstance(exc, ModelUnavailable):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc) or "Requested model is currently unavailable.",
        )
    if isinstance(exc, InvalidQuizSession):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc) or "Quiz session not found.",
        )
    if isinstance(exc, QuizAlreadyFinished):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc) or "This quiz session has already finished.",
        )
    if isinstance(exc, InvalidAnswer):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc) or "Invalid answer index submitted.",
        )
    if isinstance(exc, QuestionAlreadyAnswered):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc) or "Question has already been answered.",
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An unexpected internal error occurred.",
    )
