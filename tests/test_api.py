from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


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
    assert data["layer"] in ["layer1", "fallback", "layer2_pending"]
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
    assert data["intent"] == "capabilities"
    assert data["layer"] == "layer1"
    assert data["fallback"] is False
    assert data["activation_score"] <= 0.50


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
    assert data["layer"] in ["layer2_pending", "fallback"]


def test_post_chat_empty_message():
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 422


def test_post_chat_malformed_payload():
    response = client.post("/chat", json={"wrong_key": 1234})
    assert response.status_code == 422
