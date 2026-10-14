"""Unit tests for ModelManager lifecycle and Gemma readiness."""
import pytest
from backend.app.services.model_manager import model_manager
from backend.app.config import settings


def test_model_manager_components():
    assert model_manager.intent_classifier is not None
    assert model_manager.semantic_engine is not None
    assert model_manager.entity_extractor is not None


def test_model_manager_status():
    status = model_manager.get_status()
    assert "gemma_loaded" in status
    assert status["model_name"] == "Gemma 4 E2B Instruct"
    assert "intent_classifier_loaded" in status
    assert status["intent_classifier_loaded"] is True
    assert "vram_allocated_mb" in status


def test_gemma_configuration():
    assert settings.GEMMA_MODEL_ID == "google/gemma-4-E2B-it"
    assert settings.USE_4BIT_QUANTIZATION is True
    assert settings.DEVICE_MAP == "auto"
    assert settings.TOKEN_BUDGET_SIMPLE == 96
    assert settings.TOKEN_BUDGET_NORMAL == 192
    assert settings.TOKEN_BUDGET_COMPLEX == 384
