#!/usr/bin/env python3
"""
ChatLab Extreme Audit — Cross-Component Integration Journeys (A through I).
Exercises realistic user journeys via FastAPI TestClient and records exact evidence.
"""

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from backend.main import app
from backend.app.services.model_manager import model_manager

# Ensure test mode for fast deterministic evaluation across all 9 journeys
model_manager._test_mode = True

client = TestClient(app)

results = {}

def run_journey_a():
    """Journey A: AIML question with citations, then follow-up."""
    print("--- Journey A: AIML question + citation + follow-up ---")
    r1 = client.post("/chat", json={"message": "Explain backpropagation in deep neural networks"})
    d1 = r1.json()
    sess_id = d1.get("session_id")
    citations = d1.get("citations", [])
    
    # Send follow-up
    r2 = client.post("/chat", json={"message": "give an example", "session_id": sess_id})
    d2 = r2.json()
    
    res = {
        "status_code_1": r1.status_code,
        "intent_1": d1.get("intent"),
        "citations_count_1": len(citations),
        "citation_1_codes": [c.get("citation_code") for c in citations],
        "status_code_2": r2.status_code,
        "intent_2": d2.get("intent"),
        "routing_reason_2": d2.get("routing_reason"),
        "session_preserved": sess_id == d2.get("session_id"),
    }
    print(res)
    return res

def run_journey_b():
    """Journey B: General question without textbook retrieval."""
    print("--- Journey B: General question (bypass RAG) ---")
    r = client.post("/chat", json={"message": "What is the capital of France?"})
    d = r.json()
    res = {
        "status_code": r.status_code,
        "intent": d.get("intent"),
        "citations_count": len(d.get("citations", [])),
        "routing_reason": d.get("routing_reason"),
    }
    print(res)
    return res

def run_journey_c():
    """Journey C: Arithmetic question."""
    print("--- Journey C: Arithmetic question ---")
    r = client.post("/chat", json={"message": "what is 2 + 2?"})
    d = r.json()
    res = {
        "status_code": r.status_code,
        "intent": d.get("intent"),
        "citations_count": len(d.get("citations", [])),
        "response": d.get("response")[:60] if d.get("response") else None,
    }
    print(res)
    return res

def run_journey_d():
    """Journey D: Technical question with grounded citations."""
    print("--- Journey D: Technical question grounding ---")
    r = client.post("/chat", json={"message": "How does multi-head attention project queries, keys, and values into subspaces?"})
    d = r.json()
    cites = d.get("citations", [])
    res = {
        "status_code": r.status_code,
        "intent": d.get("intent"),
        "citations_count": len(cites),
        "first_citation": cites[0] if cites else None,
    }
    print(res)
    return res

def run_journey_e():
    """Journey E: Controlled retrieval failure with graceful degradation."""
    print("--- Journey E: Controlled retrieval failure ---")
    with patch("backend.app.services.rag_retrieval_service.search_index", side_effect=RuntimeError("FAISS index IO failure")):
        r = client.post("/chat", json={"message": "Explain backpropagation in deep networks"})
        d = r.json()
        res = {
            "status_code": r.status_code,
            "citations_count": len(d.get("citations", [])),
            "response_available": bool(d.get("response")),
            "fallback": d.get("fallback"),
        }
        print(res)
        return res

def run_journey_f():
    """Journey F: Streaming response and client interrupt simulation."""
    print("--- Journey F: Streaming response ---")
    r = client.post("/chat/stream", json={"message": "Explain attention mechanism"})
    chunks = []
    metadata_chunk = None
    for line in r.iter_lines():
        if line.startswith("data: "):
            chunk = json.loads(line[6:])
            chunks.append(chunk)
            if chunk.get("type") == "metadata":
                metadata_chunk = chunk
    res = {
        "status_code": r.status_code,
        "total_chunks": len(chunks),
        "token_chunks": sum(1 for c in chunks if c.get("type") == "token"),
        "has_metadata": metadata_chunk is not None,
        "citations_in_meta": len(metadata_chunk.get("citations", [])) if metadata_chunk else 0,
    }
    print(res)
    return res

def run_journey_g():
    """Journey G: Complete a quiz through API."""
    print("--- Journey G: Quiz workflow ---")
    # Start quiz
    r_start = client.post("/quiz/start", json={"topic": "all", "difficulty": "medium", "num_questions": 3})
    d_start = r_start.json()
    sess_id = d_start.get("session_id")
    total_q = d_start.get("total_questions")
    
    # Answer question 1
    r_ans1 = client.post(f"/quiz/{sess_id}/answer", json={"answer_index": 0})
    d_ans1 = r_ans1.json()
    
    # Advance to question 2
    r_next1 = client.post(f"/quiz/{sess_id}/next")
    
    # Answer question 2
    r_ans2 = client.post(f"/quiz/{sess_id}/answer", json={"answer_index": 1})
    
    # Advance to question 3
    r_next2 = client.post(f"/quiz/{sess_id}/next")
    
    # Answer question 3
    r_ans3 = client.post(f"/quiz/{sess_id}/answer", json={"answer_index": 2})
    
    # Finish / advance to end
    r_next3 = client.post(f"/quiz/{sess_id}/next")
    d_final = r_next3.json()
    
    # Get score
    r_score = client.get(f"/quiz/{sess_id}/score")
    d_score = r_score.json()
    
    res = {
        "start_status": r_start.status_code,
        "session_id": sess_id,
        "total_questions": total_q,
        "final_completed": d_final.get("completed"),
        "score_percent": d_score.get("score_percent"),
        "correct_count": d_score.get("correct_count"),
    }
    print(res)
    return res

def run_journey_h():
    """Journey H: Multi-turn sequential messages with isolated citations."""
    print("--- Journey H: Multi-turn sequence ---")
    # Turn 1
    r1 = client.post("/chat", json={"message": "What is gradient descent?"})
    d1 = r1.json()
    sess_id = d1.get("session_id")
    c1 = d1.get("citations", [])
    
    # Turn 2
    r2 = client.post("/chat", json={"message": "What is the capital of Japan?", "session_id": sess_id})
    d2 = r2.json()
    c2 = d2.get("citations", [])
    
    # Turn 3
    r3 = client.post("/chat", json={"message": "How does Adam optimizer adapt the learning rate?", "session_id": sess_id})
    d3 = r3.json()
    c3 = d3.get("citations", [])
    
    res = {
        "turn_1_citations": len(c1),
        "turn_2_citations": len(c2),
        "turn_3_citations": len(c3),
        "turn_2_properly_empty_citations": len(c2) == 0,
        "session_preserved": sess_id == d2.get("session_id") == d3.get("session_id"),
    }
    print(res)
    return res

def run_journey_i():
    """Journey I: Edge cases, malformed requests, invalid sessions."""
    print("--- Journey I: Edge cases & invalid inputs ---")
    # Empty message
    r_empty = client.post("/chat", json={"message": "   "})
    # Nonexistent quiz session
    r_bad_quiz = client.post("/quiz/invalid-session-uuid/answer", json={"answer_index": 0})
    # Out of range quiz answer
    r_start = client.post("/quiz/start", json={"num_questions": 2})
    quiz_sess = r_start.json().get("session_id")
    r_out_of_bounds = client.post(f"/quiz/{quiz_sess}/answer", json={"answer_index": 999})
    
    res = {
        "empty_message_status": r_empty.status_code,
        "invalid_quiz_session_status": r_bad_quiz.status_code,
        "out_of_bounds_answer_status": r_out_of_bounds.status_code,
    }
    print(res)
    return res

if __name__ == "__main__":
    results["Journey_A"] = run_journey_a()
    results["Journey_B"] = run_journey_b()
    results["Journey_C"] = run_journey_c()
    results["Journey_D"] = run_journey_d()
    results["Journey_E"] = run_journey_e()
    results["Journey_F"] = run_journey_f()
    results["Journey_G"] = run_journey_g()
    results["Journey_H"] = run_journey_h()
    results["Journey_I"] = run_journey_i()
    
    out_file = ROOT_DIR / "data" / "rag" / "evaluation" / "audit_journeys_result.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll 9 journeys executed. Results saved to {out_file}")
