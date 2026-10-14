import json
import zipfile
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
colab_dir = ROOT / "colab"
colab_dir.mkdir(exist_ok=True)

def w(path, content):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.strip() + "\n", encoding="utf-8")
    print("  Created:", path)

# 1. colab/project_manifest.json
project_manifest = {
    "project": "ChatLab",
    "domain": "AIML Question-Answering and Quiz Chatbot",
    "version": "0.3.0",
    "python_version": "3.11.9",
    "os": "Windows-10-10.0.26200-SP0",
    "transformers_version": "5.17.0",
    "peft_version": "0.21.2",
    "trl_version": "1.13.0",
    "unsloth_version": "2026.2",
    "bitsandbytes_version": "0.48.0",
    "pytorch_version": "2.14.1+cu126",
    "cuda_version": "12.6",
    "model_info": {
        "base_model_name": "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit",
        "local_model_path": "models/qwen3-4b-bnb-4bit",
        "quantization_format": "4-bit NF4",
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True
        },
        "tokenizer_config": {
            "use_fast": True,
            "trust_remote_code": True,
            "chat_template": "ChatML"
        }
    },
    "qlora_info": {
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "learning_rate": 0.0002,
        "lr_scheduler_type": "cosine",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "effective_batch_size": 16,
        "max_seq_length": 512,
        "warmup_steps": 23,
        "warmup_ratio": 0.05,
        "optimizer": "adamw_torch",
        "precision": "bf16",
        "gradient_checkpointing": True,
        "seed": 42
    },
    "dataset_info": {
        "full_path": "data/qa_dataset.jsonl",
        "train_path": "data/qa_train.jsonl",
        "validation_path": "data/qa_validation.jsonl",
        "test_path": "data/qa_test.jsonl",
        "sample_counts": {
            "full": 3000,
            "train": 2400,
            "validation": 300,
            "test": 300
        },
        "schema": {
            "instruction": "str",
            "context": "str (optional)",
            "response": "str"
        },
        "category_distribution": {
            "canonical_aiml_qa": 2000,
            "aiml_quiz_multiple_choice": 500,
            "aiml_code_conceptual_explanations": 500
        }
    }
}
w("colab/project_manifest.json", json.dumps(project_manifest, indent=2))

# 2. colab/chatlab_export_manifest.json
export_manifest = {
    "project": "ChatLab",
    "export_date": "2026-10-07",
    "files": [
        {"path": "backend/main.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/config.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/router.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/schemas.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/exceptions.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/api/chat_router.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/api/quiz_router.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/services/chatbot.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/services/chat_engine.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/services/conversation_memory.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/services/entity_extractor.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/services/fuzzy_router.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/services/intent_classifier.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/services/model_manager.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/services/preprocessing.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/services/quiz_engine.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/services/response_engine.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/services/response_postprocessor.py", "classification": "CORE_RUNTIME"},
        {"path": "backend/app/services/semantic_similarity.py", "classification": "CORE_RUNTIME"},
        {"path": "scripts/train_qwen_qlora.py", "classification": "TRAINING"},
        {"path": "scripts/validate_dataset.py", "classification": "TRAINING"},
        {"path": "scripts/build_intents_dataset.py", "classification": "TRAINING"},
        {"path": "data/qa_dataset.jsonl", "classification": "DATA"},
        {"path": "data/qa_train.jsonl", "classification": "DATA"},
        {"path": "data/qa_validation.jsonl", "classification": "DATA"},
        {"path": "data/qa_test.jsonl", "classification": "DATA"},
        {"path": "data/quiz_questions.json", "classification": "DATA"},
        {"path": "data/intents.json", "classification": "DATA"},
        {"path": "data/aiml_topics.json", "classification": "DATA"},
        {"path": "frontend/src/App.jsx", "classification": "FRONTEND"},
        {"path": "frontend/src/services/api.js", "classification": "FRONTEND"},
        {"path": "tests/test_api.py", "classification": "TEST"},
        {"path": "tests/test_chat_orchestrator.py", "classification": "TEST"},
        {"path": "tests/test_conversation_memory.py", "classification": "TEST"},
        {"path": "tests/test_quiz_engine.py", "classification": "TEST"},
        {"path": "docs/system_architecture.md", "classification": "DOCUMENTATION"},
        {"path": "docs/service_architecture.md", "classification": "DOCUMENTATION"},
        {"path": "docs/api.md", "classification": "DOCUMENTATION"}
    ]
}
w("colab/chatlab_export_manifest.json", json.dumps(export_manifest, indent=2))

# 3. colab/requirements-colab.txt
req_colab = """# Google Colab Dependencies for ChatLab Qwen QLoRA Fine-Tuning
# Compatible with Colab T4 / L4 / A100 GPU environments (Linux)

unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
torch>=2.1.0
transformers>=4.40.0
peft>=0.10.0
trl>=0.8.0
bitsandbytes>=0.43.0
datasets>=2.18.0
accelerate>=0.28.0
sentencepiece
protobuf
ninja
scipy
scikit-learn
"""
w("colab/requirements-colab.txt", req_colab)

# 4. colab/README.md
readme_colab = """# ChatLab — Qwen3 4B QLoRA Google Colab Fine-Tuning Guide

This directory contains the complete Google Colab fine-tuning workflow for ChatLab Layer 2.

## Files in this Directory
- ChatLab_Qwen3_QLoRA_Train.ipynb: Ready-to-run Google Colab Notebook.
- project_manifest.json: Full hyperparameter and dataset manifest.
- chatlab_export_manifest.json: Inventory of all ChatLab monolith components.
- requirements-colab.txt: PyPI dependencies for Colab Linux environment.
- local_integration.md: Step-by-step guide for importing the trained adapter back into ChatLab.
- checksums.sha256: SHA256 hashes verifying dataset and training script integrity.

## Quick Start on Google Colab

1. Open Google Colab (https://colab.research.google.com).
2. Upload ChatLab_Qwen3_QLoRA_Train.ipynb or open it from GitHub.
3. Select Runtime -> Change runtime type -> T4 GPU (or L4 / A100 if available).
4. Upload qa_train.jsonl, qa_validation.jsonl, and qa_test.jsonl when prompted in Cell 6 (or mount Google Drive).
5. Click Runtime -> Run all.
6. At the end of the run, download ChatLab_Qwen3_AIML_QLoRA_Adapter.zip.
7. Extract qwen3-4b-qlora-adapter/ into your local ChatLab/models/ folder.
8. Follow local_integration.md to set LAYER2_ENABLED = True.
"""
w("colab/README.md", readme_colab)

# 5. colab/local_integration.md
local_integration = """# ChatLab Layer 2 Adapter Integration Guide

Follow these steps once your Google Colab fine-tuning run has completed and you have downloaded ChatLab_Qwen3_AIML_QLoRA_Adapter.zip.

## Step 1: Unzip the Adapter
Extract ChatLab_Qwen3_AIML_QLoRA_Adapter.zip on your local machine.

Copy the qwen3-4b-qlora-adapter directory into your ChatLab models/ directory:

ChatLab/
├── models/
│   ├── qwen3-4b-bnb-4bit/          <-- Base Qwen 3-4B model
│   └── qwen3-4b-qlora-adapter/     <-- Copy downloaded adapter here
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer_config.json

## Step 2: Enable Layer 2 in Environment
Set the environment variable or update backend/app/config.py:

PowerShell:
$env:LAYER2_ENABLED="true"

Bash:
export LAYER2_ENABLED="true"

or edit backend/app/config.py:
LAYER2_ENABLED: bool = True

## Step 3: Load Model in ModelManager
When the application receives a Layer 2 query via the Fuzzy Router, load the model:

from backend.app.services.model_manager import model_manager
model_manager.load_qwen(adapter_path="models/qwen3-4b-qlora-adapter")

The existing Layer2Engine.generate() protocol in backend/app/services/chat_engine.py will now automatically invoke the fine-tuned Qwen QLoRA adapter!
"""
w("colab/local_integration.md", local_integration)

print("Colab files generated cleanly.")
