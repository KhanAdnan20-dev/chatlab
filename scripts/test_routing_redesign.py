"""Routing Redesign & Qwen-First Fuzzy Prompt Orchestration Verification Script for ChatLab."""
import sys
import time
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.main import app
from backend.app.services.model_manager import model_manager

CODE_SAMPLE = """import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.fc1 = nn.Linear(32 * 26 * 26, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc1(x)
"""

def run_redesign_tests():
    client = TestClient(app)
    print("=" * 80)
    print("CHATLAB QWEN-FIRST FUZZY PROMPT ORCHESTRATION VERIFICATION")
    print("=" * 80)

    print("\n[Ensuring Qwen3 + QLoRA model is ready in GPU memory...]")
    model_manager.ensure_qwen_loaded()

    def safe_print_resp(resp_str: str):
        safe_str = resp_str[:140].encode('ascii', errors='replace').decode('ascii')
        print(f"  Response: {safe_str}...")

    # Case 1: "hi" -> Qwen natural greeting
    print("\n--- CASE 1: 'hi' ---")
    r1 = client.post("/chat", json={"message": "hi"})
    assert r1.status_code == 200
    d1 = r1.json()
    print(f"  Layer: [{d1['layer'].upper()}] | Intent: {d1['intent']}")
    safe_print_resp(d1['response'])
    assert 'fuzzy_profile' in d1 and d1['fuzzy_profile'] is not None
    assert d1['layer'] == 'layer2', f"FAIL: Expected layer2, got {d1['layer']}"
    print("  [OK] CASE 1 PASSED!")

    # Case 2: "what is 2+2?" -> Qwen general question
    print("\n--- CASE 2: 'what is 2+2?' ---")
    r2 = client.post("/chat", json={"message": "what is 2+2?"})
    assert r2.status_code == 200
    d2 = r2.json()
    print(f"  Layer: [{d2['layer'].upper()}] | Intent: {d2['intent']}")
    safe_print_resp(d2['response'])
    assert "4" in d2['response']
    print("  [OK] CASE 2 PASSED!")

    # Case 3: "teach me recursion like I'm five" -> High beginner friendliness fuzzy profile
    print("\n--- CASE 3: 'teach me recursion like I'm five' ---")
    r3 = client.post("/chat", json={"message": "teach me recursion like I'm five"})
    assert r3.status_code == 200
    d3 = r3.json()
    fp3 = d3['fuzzy_profile']
    print(f"  Fuzzy Profile: Beginner Friendly={fp3['beginner_friendliness']*100:.0f}%, Example Pref={fp3['example_preference']*100:.0f}%")
    safe_print_resp(d3['response'])
    assert fp3['beginner_friendliness'] >= 0.70
    print("  [OK] CASE 3 PASSED!")

    # Case 4: "what is supervised learning?" -> AIML Explanation
    print("\n--- CASE 4: 'what is supervised learning?' ---")
    r4 = client.post("/chat", json={"message": "what is supervised learning?"})
    assert r4.status_code == 200
    d4 = r4.json()
    print(f"  Layer: [{d4['layer'].upper()}] | Intent: {d4['intent']}")
    safe_print_resp(d4['response'])
    assert d4['layer'] == 'layer2'
    print("  [OK] CASE 4 PASSED!")

    # Case 5: "why is overfitting bad?" -> Conversational AIML explanation
    print("\n--- CASE 5: 'why is overfitting bad?' ---")
    r5 = client.post("/chat", json={"message": "why is overfitting bad?"})
    assert r5.status_code == 200
    d5 = r5.json()
    print(f"  Layer: [{d5['layer'].upper()}] | Intent: {d5['intent']}")
    safe_print_resp(d5['response'])
    assert d5['layer'] == 'layer2'
    print("  [OK] CASE 5 PASSED!")

    # Case 6: "what should I learn after neural networks?" -> Learning advice
    print("\n--- CASE 6: 'what should I learn after neural networks?' ---")
    r6 = client.post("/chat", json={"message": "what should I learn after neural networks?"})
    assert r6.status_code == 200
    d6 = r6.json()
    print(f"  Layer: [{d6['layer'].upper()}] | Intent: {d6['intent']}")
    safe_print_resp(d6['response'])
    assert d6['layer'] == 'layer2'
    print("  [OK] CASE 6 PASSED!")

    # Case 7: "give me an ML roadmap" -> Structured roadmap
    print("\n--- CASE 7: 'give me an ML roadmap' ---")
    r7 = client.post("/chat", json={"message": "give me an ML roadmap"})
    assert r7.status_code == 200
    d7 = r7.json()
    print(f"  Layer: [{d7['layer'].upper()}] | Intent: {d7['intent']}")
    safe_print_resp(d7['response'])
    assert d7['layer'] == 'layer2'
    print("  [OK] CASE 7 PASSED!")

    # Case 8: "What is CNN?" then "explain" -> Multi-turn Context Resolution
    print("\n--- CASE 8: 'What is CNN?' then 'explain' ---")
    r8a = client.post("/chat", json={"message": "What is CNN?"})
    r8b = client.post("/chat", json={"message": "explain", "session_id": r8a.json()['session_id']})
    assert r8b.status_code == 200
    d8b = r8b.json()
    print(f"  Layer: [{d8b['layer'].upper()}] | Intent: {d8b['intent']}")
    safe_print_resp(d8b['response'])
    assert d8b['layer'] == 'layer2'
    print("  [OK] CASE 8 PASSED!")

    # Case 9: "What is overfitting?" then "why?" -> Multi-turn Context Resolution
    print("\n--- CASE 9: 'What is overfitting?' then 'why?' ---")
    r9a = client.post("/chat", json={"message": "What is overfitting?"})
    r9b = client.post("/chat", json={"message": "why?", "session_id": r9a.json()['session_id']})
    assert r9b.status_code == 200
    d9b = r9b.json()
    print(f"  Layer: [{d9b['layer'].upper()}] | Intent: {d9b['intent']}")
    safe_print_resp(d9b['response'])
    assert d9b['layer'] == 'layer2'
    print("  [OK] CASE 9 PASSED!")

    # Case 10: "compare CNN and RNN" -> Comparison
    print("\n--- CASE 10: 'compare CNN and RNN' ---")
    r10 = client.post("/chat", json={"message": "compare CNN and RNN"})
    assert r10.status_code == 200
    d10 = r10.json()
    print(f"  Layer: [{d10['layer'].upper()}] | Intent: {d10['intent']}")
    safe_print_resp(d10['response'])
    assert d10['layer'] == 'layer2'
    print("  [OK] CASE 10 PASSED!")

    # Case 11: Code + "explain this" -> High Code Focus
    print("\n--- CASE 11: PyTorch Code + 'explain this' ---")
    r11 = client.post("/chat", json={"message": f"explain this\n\n{CODE_SAMPLE}"})
    assert r11.status_code == 200
    d11 = r11.json()
    fp11 = d11['fuzzy_profile']
    print(f"  Fuzzy Profile: Code Focus={fp11['code_focus']*100:.0f}%")
    safe_print_resp(d11['response'])
    assert fp11['code_focus'] >= 0.80
    assert d11['layer'] == 'layer2'
    print("  [OK] CASE 11 PASSED!")

    # Case 12: Code + "debug this" -> Code Debugging
    print("\n--- CASE 12: PyTorch Code + 'debug this' ---")
    r12 = client.post("/chat", json={"message": f"debug this\n\n{CODE_SAMPLE}"})
    assert r12.status_code == 200
    d12 = r12.json()
    print(f"  Layer: [{d12['layer'].upper()}] | Intent: {d12['intent']}")
    assert d12['layer'] == 'layer2'
    print("  [OK] CASE 12 PASSED!")

    # Case 13: "start a quiz" -> Quiz Engine
    print("\n--- CASE 13: 'start a quiz' ---")
    r13 = client.post("/quiz/start", json={"topic": "all", "difficulty": "medium", "num_questions": 3})
    assert r13.status_code in (200, 201)
    print(f"  [OK] Quiz started cleanly: session_id={r13.json()['session_id']}")
    print("  [OK] CASE 13 PASSED!")

    print("\n" + "=" * 80)
    print("[SUCCESS] ALL REQUIRED QWEN-FIRST TEST CASES PASSED CLEANLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_redesign_tests()
