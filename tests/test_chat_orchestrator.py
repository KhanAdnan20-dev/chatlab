from backend.app.services.chatbot import chatbot_service
from backend.app.services.model_manager import model_manager

model_manager.set_test_mode(True)


def test_orchestrator_process_greeting():
    res = chatbot_service.process("hello chatlab")
    assert res.response is not None
    assert res.layer in ["layer1", "layer2", "layer2_pending", "fallback", "gemma", "vllm"]
    assert res.confidence >= 0.0
    assert res.session_id != ""
    assert res.fuzzy_profile is not None


def test_orchestrator_process_capabilities():
    res = chatbot_service.process("what are your capabilities?")
    assert res.intent in ["capabilities", "help", "unknown"]
    assert res.layer in ["layer1", "layer2", "gemma", "vllm"]
    assert res.fallback is False
    assert res.fuzzy_profile is not None
