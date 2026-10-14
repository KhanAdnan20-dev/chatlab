"""Local Gemma 4 E2B Instruct Verification and Benchmark Script."""
import os
import sys
import time

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from backend.app.config import settings
from backend.app.services.model_manager import model_manager
from backend.app.services.chatbot import chatbot_service
from backend.app.services.conversation_memory import conversation_memory

def run_benchmark():
    print("=" * 70)
    print("CHATLAB — GEMMA 4 E2B INSTRUCT LOCAL HARDWARE INFERENCE BENCHMARK")
    print("=" * 70)

    # 1. Hardware Inspection
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Device: {device_name}")
        print(f"Total VRAM: {total_vram_gb:.2f} GB")
    else:
        print("Running on CPU.")

    # Disable mock/test mode to ensure genuine local Gemma inference
    model_manager.set_test_mode(False)

    # 2. Model Loading
    print("\n--- 1. LOADING GEMMA 4 E2B INSTRUCT (4-bit NF4) ---")
    t0 = time.time()
    loaded = model_manager.ensure_gemma_loaded()
    load_time = time.time() - t0

    if not loaded or not model_manager.gemma_available:
        print(f"[FAIL] Could not load Gemma 4: {model_manager._load_error}")
        return False

    status = model_manager.get_status()
    print(f"[OK] Model loaded in {load_time:.2f} seconds.")
    print(f"VRAM Allocated: {status['vram_allocated_mb']:.2f} MB")
    print(f"VRAM Reserved:  {status['vram_reserved_mb']:.2f} MB")

    # 3. Test Cases
    test_cases = [
        {
            "name": "AIML Conceptual Query",
            "prompt": "What is overfitting in machine learning and how can it be prevented?",
            "session_id": "bench-sess-aiml",
        },
        {
            "name": "General Query (ELI5 / Analogy)",
            "prompt": "Teach me recursion like I'm five.",
            "session_id": "bench-sess-eli5",
        },
        {
            "name": "Follow-up Query (Memory & Resolution)",
            "prompt": "Explain it more simply with a fun toy example.",
            "session_id": "bench-sess-eli5",
        },
        {
            "name": "Code Analysis Query",
            "prompt": (
                "Explain how this binary search algorithm works:\n"
                "```python\n"
                "def binary_search(arr, target):\n"
                "    low, high = 0, len(arr) - 1\n"
                "    while low <= high:\n"
                "        mid = (low + high) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            low = mid + 1\n"
                "        else:\n"
                "            high = mid - 1\n"
                "    return -1\n"
                "```"
            ),
            "session_id": "bench-sess-code",
        },
        {
            "name": "Mathematical Reasoning Query",
            "prompt": "Calculate precision, recall, and F1-score for TP=72, FP=18, FN=10 step-by-step.",
            "session_id": "bench-sess-math",
        },
    ]

    results = []

    print("\n--- 2. RUNNING INFERENCE BENCHMARKS ---")
    for tc in test_cases:
        print(f"\n>> Running: {tc['name']}")
        t_start = time.perf_counter()
        res = chatbot_service.process(tc["prompt"], session_id=tc["session_id"])
        latency_sec = time.perf_counter() - t_start

        tokens = res.tokens_generated or max(1, len(res.response.split()))
        throughput = tokens / latency_sec if latency_sec > 0 else 0

        fp = res.fuzzy_profile
        fp_str = (
            f"TechDepth={fp.technical_depth:.2f}, BegFriendly={fp.beginner_friendliness:.2f}, "
            f"Conv={fp.conversationalness:.2f}, Math={fp.mathematical_detail:.2f}"
            if fp else "None"
        )

        print(f"Latency: {latency_sec:.2f}s ({latency_sec*1000:.1f}ms)")
        print(f"Tokens: {tokens} | Throughput: {throughput:.2f} tok/s")
        print(f"Model: {res.model_name} | Layer: {res.layer} | Retried: {res.retried}")
        print(f"Fuzzy Profile: {fp_str}")
        print(f"Preview: {res.response[:160]}...")

        results.append({
            "name": tc["name"],
            "latency_sec": latency_sec,
            "tokens": tokens,
            "throughput": throughput,
            "response": res.response,
        })

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Model Loading Time: {load_time:.2f}s")
    print(f"GPU VRAM Allocated: {status['vram_allocated_mb']:.2f} MB ({status['vram_allocated_mb']/1024:.2f} GB)")
    print(f"Average Generation Latency: {sum(r['latency_sec'] for r in results)/len(results):.2f}s")
    print(f"Average Throughput: {sum(r['throughput'] for r in results)/len(results):.2f} tokens/s")
    print("All inference benchmarks completed successfully!")
    return True

if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
