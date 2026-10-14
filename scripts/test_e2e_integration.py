"""End-to-End Integration Verification Script for ChatLab."""
import sys
import time
import json
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.main import app
from backend.app.services.model_manager import model_manager

def run_e2e_tests():
    client = TestClient(app)
    print("=" * 80)
    print("CHATLAB END-TO-END SYSTEM INTEGRATION TEST")
    print("=" * 80)

    # 1. Health check & Model Loading
    print("\n[1/5] Testing GET / and GET /health...")
    r_root = client.get("/")
    assert r_root.status_code == 200, f"Root failed: {r_root.text}"
    print(f"  [OK] GET / -> {r_root.json()}")

    r_health = client.get("/health")
    assert r_health.status_code == 200, f"Health failed: {r_health.text}"
    health_data = r_health.json()
    print(f"  [OK] GET /health -> status={health_data['status']}, layer2_enabled={health_data['layer2_enabled']}")

    print("\n[Ensuring Qwen + QLoRA model loaded into GPU memory...]")
    loaded = model_manager.ensure_qwen_loaded()
    print(f"  [OK] Qwen model ready: {loaded}")

    # 2. Chat Routing & Layer 1 vs Layer 2 Verification
    print("\n[2/5] Testing Chat Queries across Mamdani Fuzzy Router & Layer 1 / Layer 2...")
    test_questions = [
        ("What is supervised learning?", "concept_definition"),
        ("What is the F1 score formula?", "formula_request"),
        ("What is the difference between precision and recall?", "concept_comparison"),
        ("What is CNN?", "concept_definition"),
        ("Explain the bias variance tradeoff with an intuitive example.", "example_request"),
        ("Why can increasing model complexity reduce bias but increase variance?", "concept_explanation"),
    ]

    routing_results = []

    for q, expected_intent in test_questions:
        start = time.perf_counter()
        res = client.post("/chat", json={"message": q})
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        assert res.status_code == 200, f"Chat failed for '{q}': {res.text}"
        data = res.json()

        item = {
            "question": q,
            "intent": data["intent"],
            "confidence": round(data["confidence"], 4),
            "similarity": round(data.get("similarity_score") or 0.0, 4),
            "complexity": round(data.get("complexity_score") or 0.0, 4),
            "activation": round(data.get("activation_score") or 0.0, 4),
            "selected_layer": data["layer"],
            "processing_time_ms": elapsed,
            "response_preview": data["response"][:120].replace("\n", " ") + "..."
        }
        routing_results.append(item)

        print(f"\n  Query: '{q}'")
        print(f"    Intent: {item['intent']} (conf={item['confidence']:.2f}) | Sim Score: {item['similarity']:.2f}")
        print(f"    Fuzzy Activation: {item['activation']:.2f} -> Selected Layer: [{item['selected_layer'].upper()}] ({elapsed} ms)")
        print(f"    Answer: {item['response_preview']}")

    # 3. Multi-turn Session Memory
    print("\n[3/5] Testing Multi-Turn Conversation Memory...")
    r1 = client.post("/chat", json={"message": "What is overfitting?"})
    assert r1.status_code == 200
    d1 = r1.json()
    sess_id = d1["session_id"]
    print(f"  Turn 1: 'What is overfitting?' -> sess_id={sess_id}, layer={d1['layer']}")

    r2 = client.post("/chat", json={"message": "How can I reduce it?", "session_id": sess_id})
    assert r2.status_code == 200
    d2 = r2.json()
    print(f"  Turn 2: 'How can I reduce it?' -> sess_id={d2['session_id']}, layer={d2['layer']}")
    print(f"  Answer: {d2['response'][:120]}...")
    assert d2["session_id"] == sess_id, "Session ID mismatch in multi-turn"

    # 4. Quiz Engine Full Lifecycle
    print("\n[4/5] Testing Quiz System Lifecycle...")
    # Start quiz
    r_qstart = client.post("/quiz/start", json={"topic": "all", "difficulty": "medium", "num_questions": 3})
    assert r_qstart.status_code in (200, 201), f"Quiz start failed: {r_qstart.text}"
    qdata = r_qstart.json()
    qsess = qdata["session_id"]
    print(f"  [OK] POST /quiz/start -> session_id={qsess}, total_questions={qdata['total_questions']}")
    print(f"       First Question: {qdata['first_question']['question']}")

    # Submit answer
    r_ans = client.post(f"/quiz/{qsess}/answer", json={"answer_index": 0})
    assert r_ans.status_code == 200, f"Quiz answer failed: {r_ans.text}"
    adata = r_ans.json()
    print(f"  [OK] POST /quiz/{qsess}/answer -> is_correct={adata['is_correct']}, explanation='{adata['explanation'][:60]}...'")

    # Next question
    r_next = client.post(f"/quiz/{qsess}/next")
    assert r_next.status_code == 200, f"Quiz next failed: {r_next.text}"
    ndata = r_next.json()
    next_index = ndata["question"]["current_index"] if ndata.get("question") else "completed"
    print(f"  [OK] POST /quiz/{qsess}/next -> next_question_index={next_index}")

    # Get score
    r_score = client.get(f"/quiz/{qsess}/score")
    assert r_score.status_code == 200, f"Quiz score failed: {r_score.text}"
    sdata = r_score.json()
    print(f"  [OK] GET /quiz/{qsess}/score -> correct={sdata['correct_count']}/{sdata['total_questions']} ({sdata['score_percent']}%)")

    # Finish quiz
    r_finish = client.post(f"/quiz/{qsess}/finish")
    assert r_finish.status_code == 200, f"Quiz finish failed: {r_finish.text}"
    fdata = r_finish.json()
    print(f"  [OK] POST /quiz/{qsess}/finish -> completed={fdata['completed']}")

    # 5. Summary Report
    print("\n" + "=" * 80)
    print("ROUTING SUMMARY MATRIX")
    print("=" * 80)
    print(f"{'Question':<45} | {'Intent':<20} | {'Conf':<6} | {'Sim':<6} | {'Fuzzy Act':<9} | {'Layer':<10}")
    print("-" * 110)
    for r in routing_results:
        print(f"{r['question'][:44]:<45} | {r['intent']:<20} | {r['confidence']:<6.2f} | {r['similarity']:<6.2f} | {r['activation']:<9.2f} | {r['selected_layer']:<10}")
    print("=" * 80)
    print("[SUCCESS] ALL END-TO-END INTEGRATION TESTS PASSED CLEANLY!")

    return routing_results

if __name__ == "__main__":
    run_e2e_tests()
