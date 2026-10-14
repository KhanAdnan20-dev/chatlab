"""Conversation Memory — in-memory session manager for multi-turn conversations."""
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class ConversationSession:
    """Single conversation session with rolling history window and turn lifecycle tracking."""

    def __init__(self, session_id: str, max_turns: int = 20) -> None:
        self.session_id = session_id
        self.max_turns = max_turns
        self._history: list[dict] = []
        self._last_user_request_id: Optional[str] = None
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at

    def add_user_message(self, content: str, request_id: Optional[str] = None) -> bool:
        """
        Records a user turn. If this is a retry of the currently uncompleted user turn
        (matching request_id or matching text while awaiting an assistant reply),
        the duplicate entry is skipped while preserving the existing turn.
        Returns True if a new message was added, False if deduplicated.
        """
        # Check if the last turn is an uncompleted user message awaiting an assistant response
        if self._history and self._history[-1]["role"] == "user":
            # Scenario A: Explicit request_id match
            if request_id and self._last_user_request_id and request_id == self._last_user_request_id:
                logger.info("Deduplicated retry for request_id '%s' in session '%s'", request_id, self.session_id)
                return False
            # Scenario B: Uncompleted user message with identical content (e.g. streaming failure fallback)
            if self._history[-1]["content"] == content:
                logger.info("Deduplicated uncompleted user message turn in session '%s'", self.session_id)
                if request_id:
                    self._last_user_request_id = request_id
                return False

        self._history.append({
            "role": "user",
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
        })
        self._last_user_request_id = request_id

        cap = self.max_turns * 2
        if len(self._history) > cap:
            self._history = self._history[-cap:]
        self.updated_at = datetime.now(timezone.utc)
        return True

    def add_assistant_message(self, content: str) -> None:
        self._history.append({
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._last_user_request_id = None
        cap = self.max_turns * 2
        if len(self._history) > cap:
            self._history = self._history[-cap:]
        self.updated_at = datetime.now(timezone.utc)

    def add_message(self, role: str, content: str) -> None:
        if role == "user":
            self.add_user_message(content)
        else:
            self.add_assistant_message(content)

    def get_history(self, n_turns: Optional[int] = None) -> list[dict]:
        if n_turns is None:
            return list(self._history)
        return list(self._history[-(n_turns * 2):])

    def clear(self) -> None:
        self._history.clear()
        self._last_user_request_id = None


class ConversationMemory:
    """Registry of active in-memory conversation sessions."""

    def __init__(self, max_turns: int = 20) -> None:
        self._sessions: dict[str, ConversationSession] = {}
        self.max_turns = max_turns

    def get_or_create(self, session_id: Optional[str]) -> ConversationSession:
        if not session_id:
            session_id = str(uuid.uuid4())
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(session_id, self.max_turns)
            logger.debug("New conversation session created: %s", session_id)
        return self._sessions[session_id]

    def add_user_message(self, session_id: str, content: str, request_id: Optional[str] = None) -> bool:
        return self.get_or_create(session_id).add_user_message(content, request_id=request_id)

    def add_assistant_message(self, session_id: str, content: str) -> None:
        self.get_or_create(session_id).add_assistant_message(content)

    def get_history(self, session_id: str, n_turns: Optional[int] = None) -> list[dict]:
        if session_id not in self._sessions:
            return []
        return self._sessions[session_id].get_history(n_turns)

    def clear_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].clear()

    def session_count(self) -> int:
        return len(self._sessions)


from backend.app.config import settings
conversation_memory = ConversationMemory(max_turns=settings.MAX_HISTORY_TURNS)
