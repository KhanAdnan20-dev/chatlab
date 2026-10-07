"""
ChatLab - Layer 2 Qwen QLoRA Supervised Fine-Tuning Pipeline.

Target Hardware: NVIDIA RTX 4050 (6 GB VRAM)
Target Model: unsloth/Qwen3-4B-Instruct-2507-bnb-4bit (Local copy in models/qwen3-4b-bnb-4bit)
Dataset: data/qa_train.jsonl (2,400 samples), data/qa_validation.jsonl (300 samples)
Adapter Output: models/qwen3-4b-qlora-adapter

NOTE: This script is prepared and pre-configured for execution when training is started.
DO NOT run training automatically during setup phases.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Enforce UTF-8 encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import settings


# ==============================================================================
# CONFIGURATION FOR 6 GB VRAM (RTX 4050)
# ==============================================================================
@dataclass
class QLoraTrainingConfig:
    # Model and dataset paths
    model_path: str = str(settings.QWEN_MODEL_PATH)
    train_file: str = str(settings.QA_TRAIN_PATH)
    validation_file: str = str(settings.QA_VALIDATION_PATH)
    output_dir: str = str(settings.QWEN_ADAPTER_OUTPUT_DIR)

    # Sequence & Batch Memory Settings (Optimized for 6GB VRAM)
    max_seq_length: int = int(os.getenv("MAX_SEQ_LENGTH", "512"))
    per_device_train_batch_size: int = int(os.getenv("BATCH_SIZE", "1"))
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = int(os.getenv("GRAD_ACCUM", "16"))

    # Optimizer & Learning Rate Schedule
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-4"))
    num_train_epochs: int = int(os.getenv("NUM_EPOCHS", "3"))
    warmup_ratio_pct: float = float(os.getenv("WARMUP_RATIO", "0.05"))
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True

    # LoRA Adapter Configuration
    lora_r: int = int(os.getenv("LORA_R", "16"))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "32"))
    lora_dropout: float = float(os.getenv("LORA_DROPOUT", "0.05"))
    target_modules: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # Logging and Saving Checkpoints
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 2


def format_instruction(sample: dict) -> str:
    """Formats an instruction/response dictionary into the standard ChatML prompt template."""
    instruction = sample.get("instruction", "").strip()
    context = sample.get("context", "").strip()
    response = sample.get("response", "").strip()

    if context:
        user_content = f"{instruction}\n\nContext:\n{context}"
    else:
        user_content = instruction

    prompt = (
        f"<|im_start|>system\n"
        f"You are ChatLab, an intelligent educational conversational AI assistant specializing in Artificial Intelligence, Machine Learning, Deep Learning, NLP, and Soft Computing.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{response}<|im_end|>"
    )
    return prompt


def run_training(config: QLoraTrainingConfig = None, dry_run: bool = False) -> None:
    """
    Executes or verifies QLoRA fine-tuning setup on the local Qwen model using SFTTrainer.
    When dry_run is True, loads all components and initializes SFTTrainer without executing trainer.train().
    """
    if config is None:
        config = QLoraTrainingConfig()

    print("=" * 80)
    print("         ChatLab Layer 2: Qwen QLoRA Supervised Fine-Tuning Pipeline         ")
    print("=" * 80)
    print(f"Model Path:          {config.model_path}")
    print(f"Training Dataset:    {config.train_file}")
    print(f"Validation Dataset:  {config.validation_file}")
    print(f"Output Adapter Dir:  {config.output_dir}")
    print(f"Max Sequence Length: {config.max_seq_length}")
    print(f"Batch Size:          {config.per_device_train_batch_size} (Grad Accum: {config.gradient_accumulation_steps})")
    print(f"Effective Batch:     {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"Learning Rate:       {config.learning_rate}")
    print(f"Epochs:              {config.num_train_epochs}")
    print(f"Precision:           {'BF16' if config.bf16 else ('FP16' if config.fp16 else 'FP32')}")
    print(f"LoRA Config:         r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print("-" * 80)

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        print(f"Error importing required fine-tuning libraries: {e}")
        print("Ensure 'requirements-gpu.txt' packages (transformers, peft, trl, bitsandbytes, datasets) are installed.")
        return

    # 1. 4-bit Quantization Configuration (using bfloat16 compute dtype for RTX 4050)
    compute_dtype = torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else torch.float32)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load Local Tokenizer and Base Model
    print("\n[1/4] Loading local 4-bit quantized base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.config.use_cache = False

    # 3. Prepare Model for QLoRA
    print("\n[2/4] Initializing PEFT LoRA adapter...")
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Load Datasets
    print("\n[3/4] Loading and formatting QA dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": config.train_file,
            "validation": config.validation_file,
        },
    )
    train_count = len(dataset["train"])
    val_count = len(dataset["validation"])
    effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    steps_per_epoch = max(1, train_count // effective_batch)
    total_optimizer_steps = steps_per_epoch * config.num_train_epochs
    warmup_steps = max(1, int(total_optimizer_steps * config.warmup_ratio_pct))

    print(f"  • Training Samples:       {train_count}")
    print(f"  • Validation Samples:     {val_count}")
    print(f"  • Steps Per Epoch:        {steps_per_epoch}")
    print(f"  • Total Optimizer Steps:  {total_optimizer_steps}")
    print(f"  • Calculated Warmup Steps:{warmup_steps} ({config.warmup_ratio_pct*100:.1f}%)")

    # 5. Training Arguments (using version-compatible SFTConfig)
    training_args = SFTConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to="none",
        max_length=config.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=format_instruction,
        args=training_args,
    )

    if dry_run:
        print("\n" + "=" * 80)
        print("           [SUCCESS] DRY INITIALIZATION TEST COMPLETED SUCCESSFULLY           ")
        print("=" * 80)
        print("  [OK] Local 4-bit Qwen model loaded into memory")
        print("  [OK] PEFT LoRA adapter initialized successfully")
        print("  [OK] QA Train & Validation datasets loaded and validated")
        print("  [OK] SFTConfig and SFTTrainer initialized with calculated warmup steps")
        print(f"  [OK] Ready to fine-tune for {total_optimizer_steps} total optimizer steps")
        print("=" * 80)
        return

    # 6. Execute Training Run
    print("\n[4/4] Starting QLoRA Fine-Tuning Run...")
    trainer.train()

    # 7. Save Final Adapter Weights
    print(f"\nSaving final LoRA adapter to: {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("Fine-tuning completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--execute":
        run_training(dry_run=False)
    elif len(sys.argv) > 1 and sys.argv[1] in ("--dry-run", "--verify"):
        run_training(dry_run=True)
    else:
        print("=" * 80)
        print("ChatLab Layer 2: QLoRA Fine-Tuning Script is PREPARED and READY.")
        print("=" * 80)
        print("To verify initialization without training, execute:")
        print("  python scripts/train_qwen_qlora.py --dry-run")
        print("\nTo start fine-tuning when ready, execute:")
        print("  python scripts/train_qwen_qlora.py --execute")
        print("=" * 80)
