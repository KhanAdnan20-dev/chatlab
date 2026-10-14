# ChatLab — Qwen3 4B QLoRA Google Colab Fine-Tuning Guide

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
