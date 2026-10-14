#!/bin/bash
export HF_HOME="/mnt/c/Users/Adnan Khan/.cache/huggingface"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
source /home/adnan_khan/vllm_env/bin/activate

echo "Starting vLLM OpenAI-compatible server for google/gemma-4-E2B-it on port 8001 (fp8_per_tensor)..."
vllm serve "/mnt/c/Users/Adnan Khan/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7" \
    --served-model-name "google/gemma-4-E2B-it" \
    --port 8001 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.78 \
    --quantization fp8_per_tensor \
    --max-model-len 2048 \
    --trust-remote-code \
    --enforce-eager \
    --generation-config vllm \
    --skip-mm-profiling \
    --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}'



