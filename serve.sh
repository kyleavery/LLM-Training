#!/bin/sh

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --enable-lora \
    --lora-modules '{"name": "custom-model", "path": "data/rlvr_adapter", "base_model_name": "Qwen/Qwen3-8B"}' \
    --max_lora_rank 16 \
    --host 127.0.0.1 \
    --port 8000 \
    --lora-dtype bfloat16 \
    --dtype bfloat16
