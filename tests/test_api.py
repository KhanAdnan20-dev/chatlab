from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


from backend.app.services.model_manager import model_manager
model_manager.set_test_mode(True)


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "ChatLab backend is running"
    assert "version" in data
    assert "docs_url" in data


def test_get_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["intents_count"] > 0


def test_post_chat_greeting():
    response = client.post("/chat", json={"message": "Hello, how are you?"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["layer"] in ["layer1", "layer2", "fallback", "layer2_pending", "layer1_fallback", "gemma", "vllm"]
    assert data["intent"] in ["greeting", "small_talk"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["entities"], list)
    assert isinstance(data["processing_time_ms"], float)
    # Verify fuzzy logic routing fields
    assert "activation_score" in data
    assert 0.0 <= data["activation_score"] <= 1.0
    assert "complexity_score" in data
    assert "entity_density" in data
    assert "routing_reason" in data


def test_post_chat_known_intent_capabilities():
    response = client.post("/chat", json={"message": "What can you do?"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] in ["capabilities", "help", "unknown"]
    assert data["layer"] in ["layer1", "layer2", "gemma", "vllm"]
    assert data["fallback"] is False
    assert "fuzzy_profile" in data


def test_post_chat_with_entities():
    response = client.post("/chat", json={"message": "I have a Python exam on Monday"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["entities"]) > 0
    entity_texts = [e["text"].lower() for e in data["entities"]]
    assert "python" in entity_texts or "monday" in entity_texts
    assert data["entity_density"] > 0.0


def test_post_chat_low_confidence_unknown():
    response = client.post(
        "/chat",
        json={"message": "qwertyuiop zxcvbnm 99887766 random unmatched string"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["fallback"] is True or data["intent"] == "unknown"
    assert data["layer"] in ["layer2", "layer2_pending", "layer1_fallback", "fallback", "gemma", "vllm"]


def test_post_chat_empty_message():
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 422


def test_post_chat_malformed_payload():
    response = client.post("/chat", json={"wrong_key": 1234})
    assert response.status_code == 422


def test_stream_failure_and_sync_retry_deduplication():
    from backend.app.services.conversation_memory import conversation_memory

    session_id = "e2e-stream-retry-session"
    turn_id = "turn-e2e-retry-101"
    prompt = "Explain stochastic gradient descent"

    from backend.app.services.chatbot import chatbot_service

    # 1. Stream starts: process_stream registers user message and yields initial tokens
    stream_gen = chatbot_service.process_stream(
        message=prompt,
        session_id=session_id,
        request_id=turn_id,
    )
    first_chunk = next(stream_gen)
    assert first_chunk["type"] == "token"

    # Simulate network drop / client abort mid-stream: generator terminates before completion
    stream_gen.close()

    # The user message was registered in memory, but assistant reply was never completed
    history_mid = conversation_memory.get_history(session_id)
    assert len(history_mid) >= 1
    assert history_mid[-1]["role"] == "user"

    # 2. Client encounters stream failure and retries synchronously with same turn_id
    sync_resp = client.post(
        "/chat",
        json={"message": prompt, "session_id": session_id, "request_id": turn_id},
    )
    assert sync_resp.status_code == 200
    sync_data = sync_resp.json()
    assert sync_data["response"] != ""

    # 3. Verify conversation memory contains strictly ONE user turn and ONE assistant turn
    history_final = conversation_memory.get_history(session_id)
    assert len(history_final) == 2
    assert history_final[0]["role"] == "user"
    assert history_final[0]["content"] == prompt
    assert history_final[1]["role"] == "assistant"

    # 4. Verify that a subsequent legitimate user turn with the same text is NOT rejected
    turn2_resp = client.post(
        "/chat",
        json={"message": prompt, "session_id": session_id, "request_id": "turn-e2e-new-102"},
    )
    assert turn2_resp.status_code == 200
    history_turn2 = conversation_memory.get_history(session_id)
    assert len(history_turn2) == 4
    assert [h["role"] for h in history_turn2] == ["user", "assistant", "user", "assistant"]

