"""Tests for VLLMEngine and High-Throughput Serving Integration."""
import json
from unittest.mock import MagicMock, patch
import pytest

from backend.app.schemas import FuzzyQueryProfile
from backend.app.services.chat_engine import EngineRequest, EngineResponse, VLLMEngine


def test_vllm_engine_not_available_fallback():
    engine = VLLMEngine(base_url="http://127.0.0.1:9999/v1")
    assert engine.is_available() is False

    req = EngineRequest(
        message="What is machine learning?",
        intent="concept_definition",
        confidence=0.85,
    )
    # When unavailable, generate() should seamlessly delegate without crashing
    with patch("backend.app.services.chat_engine.gemma_engine.generate") as mock_gemma:
        mock_gemma.return_value = EngineResponse(text="Gemma reply", layer="gemma")
        res = engine.generate(req)
        assert res.layer == "gemma"
        assert res.text == "Gemma reply"
        mock_gemma.assert_called_once()


def test_vllm_engine_generate_when_available():
    engine = VLLMEngine(base_url="http://127.0.0.1:8001/v1", model_name="google/gemma-4-E2B-it")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "vLLM high-speed explanation"}}],
        "usage": {"completion_tokens": 12},
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    req = EngineRequest(
        message="Explain supervised learning",
        intent="concept_definition",
        confidence=0.90,
    )

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with patch.object(engine, "is_available", return_value=True):
            res = engine.generate(req)
            assert res.layer == "vllm"
            assert "vLLM high-speed explanation" in res.text
            assert res.tokens_generated == 12
