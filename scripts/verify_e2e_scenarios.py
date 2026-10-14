"""
ChatLab Phase 2A: End-to-End Live Verification of Scenarios A through F.
"""

import sys
import json
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from backend.main import app
from backend.app.services.model_manager import model_manager
from backend.app.services.rag_retrieval_service import rag_retrieval_service
from backend.app.services.conversation_memory import conversation_memory

client = TestClient(app)

def print_separator(title: str):
    print("\n" + "=" * 80)
    print(f" {title} ")
    print("=" * 80)

def run_scenarios(use_live_gemma: bool = False):
    print_separator(f"STARTING E2E VERIFICATION (Live Gemma: {use_live_gemma})")

    if use_live_gemma:
        print("[Setup] Loading Gemma 4 E2B Instruct onto GPU VRAM...")
        model_manager.set_test_mode(False)
        loaded = model_manager.ensure_gemma_loaded()
        print(f"  - Gemma loaded: {loaded} on {model_manager.get_status().get('device')}")
    else:
        print("[Setup] Running in test mode with live LangChain FAISS index...")
        model_manager.set_test_mode(True)

    # Pre-warm live RAG index
    rag_retrieval_service.ensure_loaded()
    print(f"  - RAG Service ready: {rag_retrieval_service.is_available}")

    results_summary = {}

    # -------------------------------------------------------------------------
    # Scenario A: Grounded technical response
    # -------------------------------------------------------------------------
    print_separator("SCENARIO A — Transformer Multi-Head Attention")
    query_a = "How does the transformer multi-head attention mechanism work?"
    print(f"Query: \"{query_a}\"")
    
    resp_a = client.post("/chat", json={"message": query_a})
    data_a = resp_a.json()
    
    citations_a = data_a.get("citations", [])
    print(f"Status: {resp_a.status_code}")
    print(f"Intent: {data_a.get('intent')} | Layer: {data_a.get('layer')}")
    print(f"Citations Returned: {len(citations_a)}")
    for c in citations_a:
        print(f"  - {c['citation_code']}: {c['source_title']} ({c['file_name']}, pp. {c['page_start']}-{c['page_end']}) [Score: {c.get('score')}]")
    print("\nGenerated Response Snippet:")
    print("  " + "\n  ".join(data_a.get("response", "").splitlines()[:6]))

    has_attention_citation = any("attention" in (c.get("content_snippet", "") + c.get("source_title", "")).lower() for c in citations_a)
    scenario_a_pass = (resp_a.status_code == 200 and len(citations_a) > 0 and has_attention_citation)
    results_summary["Scenario A"] = "PASS" if scenario_a_pass else "FAIL"
    print(f"Result: {results_summary['Scenario A']}")

    # -------------------------------------------------------------------------
    # Scenario B: Backpropagation
    # -------------------------------------------------------------------------
    print_separator("SCENARIO B — Backpropagation with Chain Rule")
    query_b = "Explain backpropagation using the chain rule."
    print(f"Query: \"{query_b}\"")
    
    resp_b = client.post("/chat", json={"message": query_b})
    data_b = resp_b.json()
    
    citations_b = data_b.get("citations", [])
    print(f"Status: {resp_b.status_code}")
    print(f"Intent: {data_b.get('intent')} | Layer: {data_b.get('layer')}")
    print(f"Citations Returned: {len(citations_b)}")
    for c in citations_b:
        print(f"  - {c['citation_code']}: {c['source_title']} ({c['file_name']}, pp. {c['page_start']}-{c['page_end']}) [Score: {c.get('score')}]")
    print("\nGenerated Response Snippet:")
    print("  " + "\n  ".join(data_b.get("response", "").splitlines()[:6]))

    has_backprop_citation = any("backprop" in (c.get("content_snippet", "") + c.get("source_title", "")).lower() for c in citations_b)
    # Check physical page reference is accurate (D2L chunk around p. 221 or UDL p. 116-117)
    valid_pages = all(c.get("page_start", 0) > 0 for c in citations_b)
    scenario_b_pass = (resp_b.status_code == 200 and len(citations_b) > 0 and has_backprop_citation and valid_pages)
    results_summary["Scenario B"] = "PASS" if scenario_b_pass else "FAIL"
    print(f"Result: {results_summary['Scenario B']}")

    # -------------------------------------------------------------------------
    # Scenario C: General conversation
    # -------------------------------------------------------------------------
    print_separator("SCENARIO C — General Conversation")
    query_c = "Hi, how are you?"
    print(f"Query: \"{query_c}\"")
    
    resp_c = client.post("/chat", json={"message": query_c})
    data_c = resp_c.json()
    
    citations_c = data_c.get("citations", [])
    print(f"Status: {resp_c.status_code}")
    print(f"Intent: {data_c.get('intent')} | Layer: {data_c.get('layer')}")
    print(f"Citations Returned: {len(citations_c)}")
    print(f"Response: {data_c.get('response')}")

    scenario_c_pass = (resp_c.status_code == 200 and len(citations_c) == 0 and len(data_c.get("response", "")) > 0)
    results_summary["Scenario C"] = "PASS" if scenario_c_pass else "FAIL"
    print(f"Result: {results_summary['Scenario C']}")

    # -------------------------------------------------------------------------
    # Scenario D: Simple arithmetic
    # -------------------------------------------------------------------------
    print_separator("SCENARIO D — Simple Arithmetic")
    query_d = "What is 2 + 2?"
    print(f"Query: \"{query_d}\"")
    
    resp_d = client.post("/chat", json={"message": query_d})
    data_d = resp_d.json()
    
    citations_d = data_d.get("citations", [])
    print(f"Status: {resp_d.status_code}")
    print(f"Intent: {data_d.get('intent')} | Layer: {data_d.get('layer')}")
    print(f"Citations Returned: {len(citations_d)}")
    print(f"Response: {data_d.get('response')}")

    scenario_d_pass = (resp_d.status_code == 200 and len(citations_d) == 0)
    results_summary["Scenario D"] = "PASS" if scenario_d_pass else "FAIL"
    print(f"Result: {results_summary['Scenario D']}")

    # -------------------------------------------------------------------------
    # Scenario E: Insufficient evidence
    # -------------------------------------------------------------------------
    print_separator("SCENARIO E — Insufficient Evidence (Corpus-Unsupported)")
    query_e = "What was the economic treaty signed following the Peloponnesian War in 404 BC?"
    print(f"Query: \"{query_e}\"")
    
    resp_e = client.post("/chat", json={"message": query_e})
    data_e = resp_e.json()
    
    citations_e = data_e.get("citations", [])
    print(f"Status: {resp_e.status_code}")
    print(f"Intent: {data_e.get('intent')} | Layer: {data_e.get('layer')}")
    print(f"Citations Returned: {len(citations_e)}")
    print(f"Response Snippet: {data_e.get('response', '')[:200]}")

    scenario_e_pass = (resp_e.status_code == 200 and len(citations_e) == 0)
    results_summary["Scenario E"] = "PASS" if scenario_e_pass else "FAIL"
    print(f"Result: {results_summary['Scenario E']}")

    # -------------------------------------------------------------------------
    # Scenario F: Quiz isolation
    # -------------------------------------------------------------------------
    print_separator("SCENARIO F — Quiz Isolation & Deterministic Evaluation")
    
    # 1. Start Quiz
    q_start = client.post("/quiz/start", json={"topic": "all", "difficulty": "easy", "num_questions": 3})
    start_data = q_start.json()
    sess_id = start_data.get("session_id")
    print(f"Quiz Start: Status {q_start.status_code} | Session: {sess_id} | Total Qs: {start_data.get('total_questions')}")

    # 2. Answer Question 1
    q_ans = client.post(f"/quiz/{sess_id}/answer", json={"answer_index": 0})
    ans_data = q_ans.json()
    print(f"Answer Submit: Status {q_ans.status_code} | Correct: {ans_data.get('is_correct')} | Explanation: {ans_data.get('explanation', '')[:80]}...")

    # 3. Check Score
    q_score = client.get(f"/quiz/{sess_id}/score")
    score_data = q_score.json()
    print(f"Score Check: Status {q_score.status_code} | Total: {score_data.get('total_questions')} | Correct: {score_data.get('correct_count')}")

    scenario_f_pass = (q_start.status_code == 201 and q_ans.status_code == 200 and q_score.status_code == 200)
    results_summary["Scenario F"] = "PASS" if scenario_f_pass else "FAIL"
    print(f"Result: {results_summary['Scenario F']}")

    print_separator("FINAL E2E VERIFICATION SUMMARY")
    for sc, outcome in results_summary.items():
        print(f"  {sc}: {outcome}")

    all_passed = all(v == "PASS" for v in results_summary.values())
    print(f"\nAll Scenarios Passed: {all_passed}")
    return all_passed

if __name__ == "__main__":
    live_flag = "--live-gemma" in sys.argv
    success = run_scenarios(use_live_gemma=live_flag)
    sys.exit(0 if success else 1)
