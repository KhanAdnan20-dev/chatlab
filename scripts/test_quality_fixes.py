"""Quality Fixes Verification Script for ChatLab (Tests A - J)."""
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

CODE_PASTE_NN = """import numpy as np

class Layer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros((1, output_dim))
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        if self.activation == 'relu':
            return np.maximum(0, self.output)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-self.output))
        return self.output

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
"""

def run_quality_tests():
    client = TestClient(app)
    print("=" * 80)
    print("CHATLAB QUALITY FIXES VERIFICATION (TESTS A - J)")
    print("=" * 80)

    from backend.app.config import settings
    if not settings.VLLM_ENABLED:
        print("\n[Ensuring Gemma model is ready in GPU memory...]")
        model_manager.ensure_gemma_loaded()
    else:
        print("\n[Using active vLLM engine on port 8001...]")

    # TEST A: "roadmap to learn ml"
    print("\n--- TEST A: 'roadmap to learn ml' ---")
    res_a = client.post("/chat", json={"message": "roadmap to learn ml"})
    assert res_a.status_code == 200
    data_a = res_a.json()
    print(f"  Intent: {data_a['intent']} (conf={data_a['confidence']:.2f}) | Layer: {data_a['layer']}")
    print(f"  Response: {data_a['response'][:140]}...")
    assert data_a['intent'] != 'capabilities', "FAIL: 'roadmap to learn ml' mapped to capabilities!"
    assert any(kw in data_a['response'].lower() for kw in ["roadmap", "phase", "step", "learn", "python", "scikit", "math"]), "FAIL: Response is not a roadmap!"
    print("  [OK] TEST A PASSED!")

    # TEST B: "explain" with previous "roadmap to learn ml"
    print("\n--- TEST B: Context Follow-Up 'explain' ---")
    sess_b = data_a['session_id']
    res_b = client.post("/chat", json={"message": "explain", "session_id": sess_b})
    assert res_b.status_code == 200
    data_b = res_b.json()
    print(f"  Intent: {data_b['intent']} | Layer: {data_b['layer']}")
    print(f"  Response: {data_b['response'][:140]}...")
    assert "capabilities" not in data_b['response'].lower(), "FAIL: Follow-up 'explain' produced generic capabilities!"
    print("  [OK] TEST B PASSED!")

    # TEST C: Overfitting follow-up "explain with an example"
    print("\n--- TEST C: 'What is overfitting?' -> 'explain with an example' ---")
    r_c1 = client.post("/chat", json={"message": "What is overfitting?"})
    sess_c = r_c1.json()['session_id']
    r_c2 = client.post("/chat", json={"message": "explain with an example", "session_id": sess_c})
    data_c2 = r_c2.json()
    print(f"  Turn 2 Response: {data_c2['response'][:140]}...")
    assert any(kw in data_c2['response'].lower() for kw in ["example", "overfit", "train", "test", "data", "model"]), "FAIL: Follow-up did not explain overfitting with an example!"
    print("  [OK] TEST C PASSED!")

    # TEST D: "What is CNN?" -> "compare it with RNN"
    print("\n--- TEST D: 'What is CNN?' -> 'compare it with RNN' ---")
    r_d1 = client.post("/chat", json={"message": "What is CNN?"})
    sess_d = r_d1.json()['session_id']
    r_d2 = client.post("/chat", json={"message": "compare it with RNN", "session_id": sess_d})
    data_d2 = r_d2.json()
    print(f"  Turn 2 Response: {data_d2['response'][:140]}...")
    assert any(kw in data_d2['response'].lower() for kw in ["contrast", "comparison", "compare", "rnn", "cnn", "difference", "versus"]), "FAIL: CNN vs RNN comparison missing!"
    print("  [OK] TEST D PASSED!")

    # TEST E: Neural Network code paste
    print("\n--- TEST E: Large Neural Network Code Paste ---")
    res_e = client.post("/chat", json={"message": f"Explain this code:\n\n{CODE_PASTE_NN}"})
    assert res_e.status_code == 200
    data_e = res_e.json()
    print(f"  Intent: {data_e['intent']} | Layer: {data_e['layer']}")
    print(f"  Response: {data_e['response'][:160]}...")
    assert data_e['intent'] == 'code_explanation', f"FAIL: Expected code_explanation intent, got {data_e['intent']}"
    assert "linear regression" not in data_e['response'].lower() or "neural network" in data_e['response'].lower(), "FAIL: Response substituted Linear Regression!"
    assert any(kw in data_e['response'].lower() for kw in ["layer", "neural", "weight", "forward", "activation", "relu"]), "FAIL: Response does not analyze Neural Network code!"
    print("  [OK] TEST E PASSED!")

    # TEST F: "debug this code" with code
    print("\n--- TEST F: 'debug this code' with Neural Network Code ---")
    res_f = client.post("/chat", json={"message": f"debug this code\n\n{CODE_PASTE_NN}"})
    assert res_f.status_code == 200
    data_f = res_f.json()
    print(f"  Intent: {data_f['intent']} | Layer: {data_f['layer']}")
    assert data_f['intent'] == 'problem_solving', f"FAIL: Expected problem_solving, got {data_f['intent']}"
    print("  [OK] TEST F PASSED!")

    # TEST G: "what algorithm does this code implement?"
    print("\n--- TEST G: Algorithm Identification for Code ---")
    res_g = client.post("/chat", json={"message": f"what algorithm does this code implement?\n\n{CODE_PASTE_NN}"})
    assert res_g.status_code == 200
    data_g = res_g.json()
    print(f"  Intent: {data_g['intent']} | Layer: {data_g['layer']}")
    print(f"  Response: {data_g['response'][:140]}...")
    assert any(kw in data_g['response'].lower() for kw in ["neural network", "perceptron", "deep learning", "layer", "forward"]), "FAIL: Algorithm identification failed!"
    print("  [OK] TEST G PASSED!")

    # TEST H: "give me the F1 score formula"
    print("\n--- TEST H: 'give me the F1 score formula' ---")
    res_h = client.post("/chat", json={"message": "give me the F1 score formula"})
    assert res_h.status_code == 200
    data_h = res_h.json()
    print(f"  Intent: {data_h['intent']} | Layer: {data_h['layer']}")
    assert data_h['intent'] == 'formula_request'
    print("  [OK] TEST H PASSED!")

    # TEST I: "explain bias variance tradeoff intuitively"
    print("\n--- TEST I: 'explain bias variance tradeoff intuitively' ---")
    res_i = client.post("/chat", json={"message": "explain bias variance tradeoff intuitively"})
    assert res_i.status_code == 200
    data_i = res_i.json()
    print(f"  Intent: {data_i['intent']} | Layer: {data_i['layer']}")
    print("  [OK] TEST I PASSED!")

    # TEST J: Random gibberish
    print("\n--- TEST J: Random Gibberish ('zzzz blah random unrelated text') ---")
    res_j = client.post("/chat", json={"message": "zzzz blah random unrelated text"})
    assert res_j.status_code == 200
    data_j = res_j.json()
    print(f"  Intent: {data_j['intent']} (conf={data_j['confidence']:.2f}) | Layer: {data_j['layer']}")
    print(f"  Response: {data_j['response'][:120]}...")
    assert data_j['intent'] != 'capabilities', "FAIL: Gibberish mapped to capabilities!"
    print("  [OK] TEST J PASSED!")

    print("\n" + "=" * 80)
    print("[SUCCESS] ALL QUALITY FIXES TESTS (A - J) PASSED CLEANLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_quality_tests()
