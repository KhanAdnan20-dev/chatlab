# ChatLab Layer 2 Adapter Integration Guide

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
