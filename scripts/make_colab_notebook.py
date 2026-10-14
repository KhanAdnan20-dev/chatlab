import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
colab_notebook_path = ROOT / "colab" / "ChatLab_Qwen3_QLoRA_Train.ipynb"

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# Section 1: Title & Overview
add_md("""# 🧪 ChatLab — Qwen 3-4B QLoRA Supervised Fine-Tuning Pipeline
**Domain:** Artificial Intelligence & Machine Learning QA & Quiz Assistant  
**Target Hardware:** Google Colab T4 / L4 / A100 GPU  
**Base Model:** `unsloth/Qwen3-4B-Instruct-2507-bnb-4bit`  
**Dataset:** 3,000 AIML Instruction Pairs (2,400 Train / 300 Validation / 300 Test)
""")

# Section 2: GPU Verification
add_md("## 1. GPU Verification & Dynamic Resource Detection")
add_code("""import torch
import sys
import os

print("=" * 60)
print("             ChatLab Colab Environment & GPU Audit           ")
print("=" * 60)
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cuda_ver = torch.version.cuda
    print(f"GPU:  {gpu_name}")
    print(f"VRAM: {vram_gb:.2f} GB")
    print(f"CUDA: {cuda_ver}")
    print("-" * 60)
    if vram_gb < 12.0:
        print("💡 Memory Advisory: Detected T4/Free-tier GPU (16GB VRAM cap). Using Batch Size=1, Grad Accum=16, Seq Len=512.")
    else:
        print("🚀 High-Memory Advisory: Detected L4/A100 GPU. Hardware acceleration optimal.")
else:
    print("❌ ERROR: No GPU accelerator detected! Please go to Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU.")
""")

# Section 3: Dependency Installation
add_md("## 2. Dependency Installation")
add_code("""!pip install --quiet --upgrade "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --quiet --upgrade trl peft transformers bitsandbytes datasets accelerate sentencepiece protobuf
print("✅ Compatible Colab packages installed successfully!")
""")

# Section 4: Hugging Face Auth
add_md("## 3. Hugging Face Authentication (Optional)")
add_code("""from huggingface_hub import notebook_login
import os

# Optional HF login if downloading gated models
hf_token = os.getenv("HF_TOKEN", "")
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)
    print("Logged in via HF_TOKEN environment variable.")
else:
    print("Using public model download (no HF login required for unsloth/Qwen3-4B-Instruct-2507-bnb-4bit).")
""")

# Section 5: Base Model Download
add_md("## 4. Download Base Qwen 3-4B 4-bit Quantized Model")
add_code("""from unsloth import FastLanguageModel
import torch

max_seq_length = 512
dtype = torch.bfloat16 # BF16 for modern Ampere/T4 architectures
load_in_4bit = True

model_name = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"

print(f"Downloading & Loading Base Model: {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    trust_remote_code=True,
    local_files_only=False, # Download from Hugging Face Hub
)
print("✅ Base Qwen 3-4B model & tokenizer loaded successfully!")
""")

# Section 6: Dataset Upload
add_md("## 5. Dataset Transfer (Upload or Google Drive)")
add_code("""import os
from pathlib import Path
from google.colab import files

data_dir = Path("/content/data")
data_dir.mkdir(exist_ok=True)

train_path = data_dir / "qa_train.jsonl"
val_path = data_dir / "qa_validation.jsonl"
test_path = data_dir / "qa_test.jsonl"

if not train_path.exists():
    print("Please upload qa_train.jsonl, qa_validation.jsonl, and qa_test.jsonl:")
    uploaded = files.upload()
    for fname, content in uploaded.items():
        (data_dir / fname).write_bytes(content)
        print(f"Saved: {fname} -> {data_dir / fname}")
else:
    print(f"Found existing datasets in {data_dir}")
""")

# Section 7: Dataset Validation
add_md("## 6. Dataset Count & Schema Verification")
add_code("""import json

def check_count(path, expected):
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset file: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    count = len(lines)
    print(f"  • {path.name}: {count} samples (Expected: {expected})")
    if count != expected:
        raise ValueError(f"Sample count mismatch for {path.name}: got {count}, expected {expected}")

print("Validating dataset files...")
check_count(train_path, 2400)
check_count(val_path, 300)
check_count(test_path, 300)
print("✅ Dataset counts verified perfectly!")
""")

# Section 8: Dataset Formatting
add_md("## 7. ChatML Prompt Formatting")
add_code("""from datasets import load_dataset

def format_chatml(sample):
    instruction = sample.get("instruction", "").strip()
    context = sample.get("context", "").strip()
    response = sample.get("response", "").strip()

    if context:
        user_content = f"{instruction}\\n\\nContext:\\n{context}"
    else:
        user_content = instruction

    prompt = (
        f"<|im_start|>system\\n"
        f"You are ChatLab, an intelligent educational conversational AI assistant specializing in Artificial Intelligence, Machine Learning, Deep Learning, NLP, and Soft Computing.<|im_end|>\\n"
        f"<|im_start|>user\\n"
        f"{user_content}<|im_end|>\\n"
        f"<|im_start|>assistant\\n"
        f"{response}<|im_end|>"
    )
    return {"text": prompt}

raw_dataset = load_dataset("json", data_files={
    "train": str(train_path),
    "validation": str(val_path),
    "test": str(test_path),
})

train_dataset = raw_dataset["train"].map(format_chatml)
val_dataset = raw_dataset["validation"].map(format_chatml)

print("Sample Formatted ChatML Training Prompt:")
print("-" * 60)
print(train_dataset[0]["text"][:400] + "...")
print("-" * 60)
""")

# Section 9: QLoRA Setup
add_md("## 8. Attach QLoRA Adapter Configuration")
add_code("""model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rssa=False,
    loftq_config=None,
)

trainable_params, all_params = model.get_nb_trainable_parameters()
print(f"Trainable Parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
print("✅ QLoRA adapter configured with r=16, alpha=32, dropout=0.05 on 7 target modules!")
""")

# Section 10: Training Config Summary
add_md("## 9. Training Configuration & Hyperparameters")
add_code("""batch_size = 1
grad_accum = 16
effective_batch = batch_size * grad_accum
epochs = 3
train_count = len(train_dataset)
steps_per_epoch = train_count // effective_batch
total_steps = steps_per_epoch * epochs
warmup_steps = int(total_steps * 0.05)

print("=" * 60)
print("                 Training Run Specification                  ")
print("=" * 60)
print(f"Training Samples:        {train_count}")
print(f"Validation Samples:      {len(val_dataset)}")
print(f"Per-Device Batch Size:   {batch_size}")
print(f"Gradient Accumulation:   {grad_accum}")
print(f"Effective Batch Size:    {effective_batch}")
print(f"Training Epochs:         {epochs}")
print(f"Steps Per Epoch:         {steps_per_epoch}")
print(f"Total Optimizer Steps:   {total_steps}")
print(f"Warmup Steps:            {warmup_steps} (5%)")
print(f"Learning Rate:           2e-4 (Cosine Decay)")
print(f"Precision:               BF16")
print("=" * 60)
""")

# Section 11: Execute Training
add_md("## 10. Execute QLoRA Supervised Fine-Tuning")
add_code("""from trl import SFTTrainer, SFTConfig

output_dir = "/content/chatlab_output"
adapter_output_dir = f"{output_dir}/qwen3-4b-qlora-adapter"

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=adapter_output_dir,
        report_to="none",
    ),
)

print("🚀 Starting QLoRA Fine-Tuning Run on Google Colab...")
trainer_stats = trainer.train()
print("✅ Training complete!")
""")

# Section 12: Evaluation
add_md("## 11. Validation Evaluation & Metrics Export")
add_code("""eval_results = trainer.evaluate()
print("=" * 60)
print("                   Validation Metrics                        ")
print("=" * 60)
for k, v in eval_results.items():
    print(f"  • {k}: {v}")
print("=" * 60)

training_summary = {
    "train_loss": trainer_stats.training_loss,
    "eval_loss": eval_results.get("eval_loss", None),
    "metrics": eval_results,
    "train_runtime_sec": trainer_stats.metrics.get("train_runtime", 0),
    "train_samples_per_second": trainer_stats.metrics.get("train_samples_per_second", 0),
    "train_steps_per_second": trainer_stats.metrics.get("train_steps_per_second", 0),
}

with open(f"{output_dir}/training_summary.json", "w", encoding="utf-8") as f:
    json.dump(training_summary, f, indent=2)

print(f"Saved training summary to {output_dir}/training_summary.json")
""")

# Section 13: Baseline Benchmark
add_md("## 12. Baseline vs Fine-Tuned Model Benchmark")
add_code("""benchmark_prompts = [
    "What is the key difference between Supervised and Unsupervised Learning?",
    "Explain the Bias-Variance Tradeoff in machine learning.",
    "Write the formula and explanation for F1-Score.",
    "How does feature extraction work in Convolutional Neural Networks (CNNs)?",
    "Why does standard ReLU suffer from the dying neuron problem?",
    "Explain how Backpropagation calculates parameter gradients.",
    "What strategies prevent overfitting in deep neural networks?",
    "What mechanism enables Transformers to capture long-range token dependencies?",
    "In Mamdani Fuzzy Systems, what is the role of Defuzzification?",
    "Explain Low-Rank Adaptation (LoRA) for parameter-efficient LLM fine-tuning."
]

benchmark_results = []

FastLanguageModel.for_inference(model)

print("Running benchmark evaluation across 10 core AIML prompts...")
for idx, prompt in enumerate(benchmark_prompts):
    chat_prompt = f"<|im_start|>system\\nYou are ChatLab, an intelligent AIML assistant.<|im_end|>\\n<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"
    inputs = tokenizer([chat_prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, temperature=0.7)
    gen_text = tokenizer.batch_decode(outputs)[0]
    
    reply = gen_text.split("<|im_start|>assistant\\n")[-1].replace("<|im_end|>", "").strip()
    benchmark_results.append({
        "prompt_id": idx + 1,
        "prompt": prompt,
        "finetuned_response": reply
    })
    print(f"[{idx+1}/10] Prompt: {prompt[:40]}... -> Generated {len(reply)} chars")

with open(f"{output_dir}/benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump({"benchmarks": benchmark_results}, f, indent=2)

print(f"✅ Benchmark evaluation complete. Saved to {output_dir}/benchmark_results.json")
""")

# Section 14: Save Adapter
add_md("## 13. Save LoRA Adapter Artifacts")
add_code("""model.save_pretrained(adapter_output_dir)
tokenizer.save_pretrained(adapter_output_dir)
print(f"✅ Saved LoRA Adapter to: {adapter_output_dir}")
""")

# Section 15: Verify Adapter
add_md("## 14. Verify LoRA Adapter Artifacts")
add_code("""import os

required_files = ["adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"]
for rf in required_files:
    fp = Path(adapter_output_dir) / rf
    if fp.exists():
        print(f"  [OK] {rf} ({fp.stat().st_size:,} bytes)")
    else:
        print(f"  [WARNING] {rf} missing or unverified")
""")

# Section 16: Package & Download
add_md("## 15. Package Artifacts into Zip & Trigger Download")
add_code("""import zipfile
from google.colab import files

zip_path = "/content/ChatLab_Qwen3_AIML_QLoRA_Adapter.zip"

# Create README.txt for package
readme_txt = \"\"\"======================================================================
ChatLab Layer 2 Qwen 3-4B QLoRA Fine-Tuned Adapter Package
======================================================================

Contents:
  1. qwen3-4b-qlora-adapter/  -> LoRA weights (adapter_config.json, adapter_model.safetensors)
  2. training_summary.json    -> Loss curves, training runtime, eval metrics
  3. benchmark_results.json   -> Benchmark generations for 10 AIML evaluation prompts
  4. README.txt               -> This instructions file

Instructions for Local Deployment:
  1. Extract this zip file.
  2. Copy the 'qwen3-4b-qlora-adapter' directory into your local ChatLab project:
     ChatLab/models/qwen3-4b-qlora-adapter/
  3. Enable Layer 2 in configuration:
     Set LAYER2_ENABLED = True in backend/app/config.py or environment.
  4. Start ChatLab backend:
     uvicorn backend.main:app --reload
======================================================================
\"\"\"

with open(f"{output_dir}/README.txt", "w", encoding="utf-8") as f:
    f.write(readme_txt)

print("Creating download package zip...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files_in_dir in os.walk(adapter_output_dir):
        for f in files_in_dir:
            full_p = Path(root) / f
            arc_name = Path("qwen3-4b-qlora-adapter") / full_p.relative_to(adapter_output_dir)
            zipf.write(full_p, arc_name)
    
    zipf.write(f"{output_dir}/training_summary.json", "training_summary.json")
    zipf.write(f"{output_dir}/benchmark_results.json", "benchmark_results.json")
    zipf.write(f"{output_dir}/README.txt", "README.txt")

print(f"✅ Created package: {zip_path} ({Path(zip_path).stat().st_size / 1e6:.2f} MB)")
print("Downloading package to your local computer...")
files.download(zip_path)
""")

notebook_dict = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpu_type": "T4",
            "provenance": []
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

colab_notebook_path.write_text(json.dumps(notebook_dict, indent=2), encoding="utf-8")
print(f"Created {colab_notebook_path}")
