"""Tests for DeepSeekEngine reasoning parser, hot-swap lifecycle, and streaming."""
from unittest.mock import MagicMock, patch
import pytest

from backend.app.schemas import ChatRequest
from backend.app.services.chat_engine import DeepSeekEngine, EngineRequest, deepseek_engine
from backend.app.services.chatbot import chatbot_service
from backend.app.services.model_manager import model_manager


def test_deepseek_engine_test_mode_generate():
    model_manager.set_test_mode(True)
    req = EngineRequest(message="Solve x^2 - 4 = 0", intent="code_implementation", confidence=0.90)
    res = deepseek_engine.generate(req)
    assert res.layer == "deepseek"
    assert "<think>" in res.text
    assert res.available is True


def test_deepseek_engine_test_mode_stream():
    model_manager.set_test_mode(True)
    req = EngineRequest(message="Explain backpropagation", intent="concept_definition", confidence=0.90)

    chunks = list(deepseek_engine.generate_stream(req))

    chunk_types = [c["type"] for c in chunks]
    assert "think" in chunk_types
    assert "token" in chunk_types
    assert "done" in chunk_types

    done_chunk = [c for c in chunks if c["type"] == "done"][0]
    assert done_chunk["layer"] == "deepseek"


def test_chatbot_process_stream_routes_to_deepseek():
    model_manager.set_test_mode(True)
    chunks = list(chatbot_service.process_stream(
        message="Prove that the square root of 2 is irrational.",
        model_choice="deepseek",
    ))
    types = [c["type"] for c in chunks]
    assert "think" in types
    assert "token" in types
    assert "metadata" in types

    meta = [c for c in chunks if c["type"] == "metadata"][0]
    assert meta["layer"] == "deepseek"
    assert "DeepSeek" in meta["model_name"]


def test_model_manager_switch_model():
    model_manager.set_test_mode(True)
    assert model_manager.switch_model("deepseek") is True
    assert model_manager.switch_model("gemma") is True
