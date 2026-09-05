#!/usr/bin/env bash
# 27B BF16 originale (Qwen/Qwen3.8-27B) su SGLang, SOLO per il confronto di qualita' BF16 vs NVFP4 (05/09/2026).
# Derivato da ~/vllm/run-sglang-38.sh: senza KV fp8 (il ckpt BF16 non ha le scale), mem 0.87, senza restart.
set -euo pipefail
docker rm -f sglang-27b-bf16 2>/dev/null || true
docker run -d --name sglang-27b-bf16 \
  -e SGLANG_ENABLE_SPEC_V2=True \
  -e SGLANG_SANITIZE_NAN_LOGITS=True \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --gpus all --ipc=host --shm-size 32g \
  -p 8000:8000 \
  -v /opt/hf-cache:/root/.cache/huggingface \
  lmsysorg/sglang:dev-cu13 \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-27B \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 8000 \
  --context-length 262144 \
  --mem-fraction-static 0.87 \
  --chat-template /root/.cache/huggingface/templates/chat-template-38-medium.jinja \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-metrics \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --speculative-attention-mode decode \
  --cuda-graph-max-bs 8 \
  --max-running-requests 8 \
  --mamba-ssm-dtype bfloat16 \
  --strip-thinking-cache
echo "sglang-27b-bf16 avviato su :8000"
