"""Tests for Multi-turn Conversational Memory & Context Resolution with Gemma 4."""
import pytest
from backend.app.services.chatbot import chatbot_service
from backend.app.services.conversation_memory import conversation_memory
from backend.app.services.model_manager import model_manager

model_manager.set_test_mode(True)


def test_conversation_memory_multi_turn_flow():
    session_id = "test-session-overfitting-flow"
    conversation_memory.clear_session(session_id)

    # Turn 1: "What is overfitting?"
    r1 = chatbot_service.process("What is overfitting?", session_id=session_id)
    assert r1.session_id == session_id
    assert len(r1.response) > 0

    # Verify history has turn 1
    history = conversation_memory.get_history(session_id)
    assert len(history) == 2  # user + assistant

    # Turn 2: Follow-up query "Why does it happen?"
    r2 = chatbot_service.process("Why does it happen?", session_id=session_id)
    assert r2.session_id == session_id
    history_after_2 = conversation_memory.get_history(session_id)
    assert len(history_after_2) == 4

    # Turn 3: "Explain it more simply."
    r3 = chatbot_service.process("Explain it more simply.", session_id=session_id)
    assert r3.session_id == session_id
    assert r3.fuzzy_profile is not None
    # ELI5/simple query should yield high beginner friendliness
    assert r3.fuzzy_profile.beginner_friendliness >= 0.70

    # Ensure separate sessions do not mix
    other_session_id = "test-session-isolated"
    conversation_memory.clear_session(other_session_id)
    r_other = chatbot_service.process("Hello", session_id=other_session_id)
    other_history = conversation_memory.get_history(other_session_id)
    assert len(other_history) == 2
    assert other_history[0]["content"] == "Hello"
