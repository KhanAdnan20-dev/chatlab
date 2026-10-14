"""API tests for Roadmap and Mastery endpoints."""
from fastapi.testclient import TestClient
from backend.main import app
from backend.app.services.model_manager import model_manager

model_manager.set_test_mode(True)
client = TestClient(app)


def test_api_generate_roadmap():
    response = client.get("/roadmap/generate?domain=nlp")
    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "nlp_llms"
    assert "graph TD" in data["mermaid_code"]
    assert data["total_nodes"] > 5
    assert len(data["phases"]) > 0
    assert "```mermaid" in data["markdown_content"]


def test_api_get_curriculum():
    response = client.get("/roadmap/curriculum")
    assert response.status_code == 200
    data = response.json()
    assert data["total_nodes"] >= 20
    assert data["total_edges"] >= 20
    assert len(data["nodes"]) == data["total_nodes"]


def test_api_mastery_lifecycle():
    session_id = "api-test-session-99"

    # 1. Fetch initial mastery
    get_res = client.get(f"/mastery/{session_id}")
    assert get_res.status_code == 200
    m_data = get_res.json()
    assert m_data["session_id"] == session_id
    assert m_data["nodes"]["py_prog"]["status"] == "not_started"

    # 2. Update mastery
    post_res = client.post(
        "/mastery/update",
        json={
            "session_id": session_id,
            "node_id": "py_prog",
            "status": "mastered",
            "quiz_score": 1.0,
        },
    )
    assert post_res.status_code == 200
    u_data = post_res.json()
    assert u_data["status"] == "success"
    assert u_data["new_status"] == "mastered"

    # 3. Verify next recommendation updated
    updated_res = client.get(f"/mastery/{session_id}")
    up_data = updated_res.json()
    assert up_data["nodes"]["py_prog"]["status"] == "mastered"
    assert up_data["recommended_next_node"]["id"] == "lin_alg"


def test_api_generate_syllabus_roadmap():
    """Verify the AI & Soft Computing syllabus track generates all 13 milestone nodes across Units I-VI."""
    response = client.get("/roadmap/generate?domain=ai_soft_computing")
    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "ai_soft_computing"
    assert "AI & Soft Computing (Syllabus Units I-VI)" in data["title"]
    assert "graph TD" in data["mermaid_code"]
    assert data["total_nodes"] >= 13
    assert len(data["phases"]) >= 6

    node_ids = {n["id"] for n in data["nodes"]}
    assert "ai_intro_sc" in node_ids
    assert "search_informed" in node_ids
    assert "inference_fopl" in node_ids
    assert "fuzzy_reasoning_fis" in node_ids
    assert "perceptron_backprop" in node_ids
    assert "hybrid_anfis" in node_ids


def test_rag_syllabus_retrieval():
    """Verify that RAG engine retrieves syllabus concepts and formulas with academic citations."""
    from backend.app.services.rag_engine import rag_engine

    # 1. Soft Computing vs Hard Computing
    docs_sc = rag_engine.retrieve("What is the difference between hard computing and soft computing?", top_k=2)
    assert len(docs_sc) > 0
    sc_doc = docs_sc[0]
    assert "kb_ai_intro_sc" in [d.id for d in docs_sc]
    assert "Zadeh" in sc_doc.source or "Soft Computing" in sc_doc.title

    # 2. ANFIS 5 layers
    docs_anfis = rag_engine.retrieve("Explain the 5 layers in ANFIS and hybrid learning algorithm", top_k=2)
    assert len(docs_anfis) > 0
    anfis_doc = next(d for d in docs_anfis if d.id == "kb_hybrid_anfis")
    assert "Jang" in anfis_doc.source
    assert any("Layer" in f or "Hybrid" in f for f in anfis_doc.formulas)

    # 3. Mamdani vs Sugeno FIS
    docs_fis = rag_engine.retrieve("Mamdani fuzzy inference system centroid defuzzification vs Sugeno", top_k=2)
    assert len(docs_fis) > 0
    assert any(d.id in ["kb_fuzzy_fis", "kb_fuzzy_sets"] for d in docs_fis)


def test_syllabus_quiz_questions():
    """Verify syllabus questions for CO1-CO6 exist and have valid structure."""
    response = client.get("/quiz/questions")
    assert response.status_code == 200
    data = response.json()
    questions = data["questions"]
    assert len(questions) >= 13

    co_questions = [q for q in questions if "co" in q]
    cos = {q["co"] for q in co_questions}
    assert {"CO1", "CO2", "CO3", "CO4", "CO5", "CO6"}.issubset(cos)


def test_api_ai_generate_roadmap():
    """Verify the AI dynamic roadmap endpoint generates syllabus and valid Mermaid code."""
    payload = {"query": "Quantum Computing and Grover Algorithm"}
    response = client.post("/roadmap/ai-generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "graph TD" in data["mermaid_code"]
    assert len(data["phases"]) > 0
    assert data["total_nodes"] > 0
    assert len(data["markdown_content"]) > 50
