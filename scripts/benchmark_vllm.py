"""vLLM vs Local Engine Benchmark Script for ChatLab."""
import json
import os
import sys
import time
import urllib.request

VLLM_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
CHATLAB_URL = "http://127.0.0.1:8000"

TEST_PROMPTS = [
    ("Concept (Concise)", "Explain quantum superposition in 2 sentences.", "ultra_concise"),
    ("Code Implementation", "Write a python function to compute fibonacci numbers iteratively.", "code_implementation"),
    ("Comparison", "Compare TCP and UDP in terms of reliability, speed, and typical use cases.", "structured_comparison"),
]


def test_vllm_direct(prompt: str, max_tokens: int = 256):
    print(f"\n--- [Direct vLLM Benchmark] '{prompt}' ---")
    url = f"{VLLM_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": "google/gemma-4-E2B-it",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "ChatLab-Bench/1.0"},
        method="POST",
    )

    t0 = time.time()
    first_token_time = None
    tokens = []

    try:
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        if first_token_time is None:
                            first_token_time = time.time()
                        tokens.append(delta)
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                except Exception:
                    pass
    except Exception as e:
        print(f"\n[Error connecting to vLLM]: {e}")
        return None

    t1 = time.time()
    total_time = t1 - t0
    ttft = (first_token_time - t0) * 1000 if first_token_time else 0.0
    num_tokens = len(tokens)
    gen_time = t1 - (first_token_time or t0)
    tok_per_sec = (num_tokens / gen_time) if gen_time > 0 else 0.0

    print(f"\n\nMetrics:")
    print(f"  Total Latency:     {total_time:.2f} s")
    print(f"  TTFT:              {ttft:.1f} ms")
    print(f"  Tokens Generated:  {num_tokens} tokens")
    print(f"  Generation Speed:  {tok_per_sec:.1f} tokens/sec")
    return {"latency": total_time, "ttft": ttft, "tokens": num_tokens, "speed": tok_per_sec}


def test_chatlab_stream(prompt: str):
    print(f"\n--- [ChatLab End-to-End SSE Benchmark] '{prompt}' ---")
    url = f"{CHATLAB_URL}/chat/stream"
    payload = {"message": prompt, "history": []}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "ChatLab-Bench/1.0"},
        method="POST",
    )

    t0 = time.time()
    first_token_time = None
    tokens = []
    meta = {}

    try:
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:].strip()
                try:
                    data = json.loads(data_str)
                    if data.get("type") == "token":
                        t = data.get("content", "")
                        if first_token_time is None:
                            first_token_time = time.time()
                        tokens.append(t)
                        sys.stdout.write(t)
                        sys.stdout.flush()
                    elif data.get("type") in ("meta", "metadata"):
                        meta = data
                except Exception:
                    pass
    except Exception as e:
        print(f"\n[Error connecting to ChatLab]: {e}")
        return None

    t1 = time.time()
    total_time = t1 - t0
    ttft = (first_token_time - t0) * 1000 if first_token_time else 0.0
    num_tokens = len(tokens)
    gen_time = t1 - (first_token_time or t0)
    tok_per_sec = (num_tokens / gen_time) if gen_time > 0 else 0.0

    print(f"\n\nMetrics:")
    print(f"  Layer Used:        {meta.get('layer', 'unknown')}")
    print(f"  Response Type:     {meta.get('response_type', 'unknown')}")
    print(f"  Token Budget:      {meta.get('token_budget', 'unknown')}")
    print(f"  Total Latency:     {total_time:.2f} s")
    print(f"  TTFT:              {ttft:.1f} ms")
    print(f"  Tokens Streamed:   {num_tokens} tokens")
    print(f"  Generation Speed:  {tok_per_sec:.1f} tokens/sec")
    return {"latency": total_time, "ttft": ttft, "tokens": num_tokens, "speed": tok_per_sec, "meta": meta}


if __name__ == "__main__":
    print("========================================")
    print("ChatLab vLLM Inference Benchmark")
    print("========================================")
    for title, prompt, exp_type in TEST_PROMPTS:
        test_chatlab_stream(prompt)
