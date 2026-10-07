"""
ChatLab - Layer 2 Local Qwen Base Model Inference Sanity Check.
Loads the local 4-bit quantized Qwen model and runs 5 sample AIML questions.
"""

import sys
import time
from pathlib import Path

# Enforce UTF-8 encoding for Windows console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import settings

SAMPLE_QUESTIONS = [
    "What is the difference between supervised and unsupervised learning?",
    "Explain the bias-variance tradeoff simply.",
    "What is the mathematical equation for F1-score?",
    "How does a convolutional neural network extract visual features?",
    "Why do we use the Rectified Linear Unit (ReLU) activation function?"
]


def run_qwen_inference_test():
    print("=" * 80)
    print("         ChatLab Layer 2: Local Qwen Base Model Inference Sanity Test         ")
    print("=" * 80)

    model_path = settings.QWEN_MODEL_PATH
    print(f"Model Path: {model_path}")

    if not model_path.exists():
        print(f"[ERROR] Model path {model_path} does not exist.")
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        print(f"[ERROR] Missing transformers/torch/bitsandbytes: {e}")
        return

    print("\n[1/3] Loading local tokenizer and 4-bit model...")
    t0 = time.perf_counter()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        load_time = time.perf_counter() - t0
        print(f"  [OK] Model loaded successfully in {load_time:.2f} seconds.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"  • GPU Memory Allocated: {vram_allocated:.2f} GB")
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None

    print("\n[2/3] Executing 5 Sample AIML Inference Prompts...")
    print("-" * 80)

    for idx, prompt in enumerate(SAMPLE_QUESTIONS, 1):
        formatted = (
            f"<|im_start|>system\n"
            f"You are ChatLab, an intelligent AIML educational tutor. Provide concise, clear explanations.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        start_gen = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - start_gen

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        print(f"\n[Question {idx}] \"{prompt}\"")
        print(f"  Generated ({len(generated_tokens)} tokens in {elapsed:.2f}s, {len(generated_tokens)/elapsed:.1f} tok/s):")
        print(f"  \"{response_text}\"")

    print("\n" + "=" * 80)
    print("                     Local Qwen Inference Sanity Test PASSED                    ")
    print("=" * 80)


if __name__ == "__main__":
    run_qwen_inference_test()
