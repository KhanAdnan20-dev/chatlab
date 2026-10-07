# Layer 2: Qwen QLoRA Fine-Tuning Guide

This guide details the **QLoRA (Quantized Low-Rank Adaptation)** fine-tuning procedure configured for ChatLab's Layer 2 generative model.

---

## 1. Target Hardware & Model Architecture

- **Hardware Target**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
- **Base Model**: `unsloth/Qwen3-4B-Instruct-2507-bnb-4bit` (Local copy in `models/qwen3-4b-bnb-4bit`)
- **Quantization**: 4-bit NormalFloat (`nf4`) with double quantization (`bnb_4bit_use_double_quant=True`) and `torch.float16` compute precision.
- **VRAM Footprint**: ~3.2 GB base model VRAM usage + ~1.5 GB activation VRAM with gradient accumulation.

---

## 2. LoRA Adapter Configuration

```python
peft_config = LoraConfig(
    r=16,                         # Rank of update matrices
    lora_alpha=32,                # Scaling factor (alpha / r = 2.0)
    lora_dropout=0.05,            # Regularization dropout
    target_modules=[              # Target all linear projection layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

## 3. Training Hyperparameters

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Max Sequence Length** | `512` | Fits 6 GB VRAM comfortably without truncation |
| **Per-Device Batch Size** | `1` | Minimizes peak activation memory spikes |
| **Gradient Accumulation** | `16` | Achieves effective batch size of $1 \times 16 = 16$ |
| **Learning Rate** | `2e-4` | Standard recommended LR for QLoRA fine-tuning |
| **Optimizer** | `paged_adamw_8bit` | Offloads optimizer state memory to 8-bit precision |
| **LR Scheduler** | `cosine` | Smooth decay over epochs |
| **Epochs** | `3` | ~450 total update steps over 2,400 training samples |
| **Precision** | `fp16` | Hardware accelerated on NVIDIA Tensor Cores |

---

## 4. How to Start Training (Single Command)

When ready to execute the training run on your GPU:

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Start fine-tuning
python scripts/train_qwen_qlora.py --execute
```

---

## 5. Artifact Output & Inference

Upon completion, the trained LoRA adapter weights and tokenizer will be saved to:
```
models/qwen3-4b-qlora-adapter/
├── adapter_config.json
├── adapter_model.safetensors
├── special_tokens_map.json
└── tokenizer.json
```

---

## 6. Pre-Training Inference Sanity Check

To test base model inference prior to fine-tuning:
```powershell
python scripts/test_qwen_inference.py
```
