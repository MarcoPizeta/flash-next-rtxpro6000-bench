#!/usr/bin/env bash
# Via A — vLLM + primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8 + tabella PLE INT4 in mmap.
# Fonte: scheda HF primitive-ai/Qwen3.8-Flash-Next-PLE-quant (04/09/2026).
# Uso: vllm-flash-next.sh on|off   (MTP)
set -euo pipefail
MTP="${1:?uso: vllm-flash-next.sh on|off}"
Q=/opt/flash-next/ple-quant
SP=/usr/local/lib/python3.12/dist-packages
SPEC=()
[ "$MTP" = on ] && SPEC=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
docker rm -f flash-next-vllm 2>/dev/null || true
docker run -d --name flash-next-vllm --gpus all --ipc=host --shm-size 32g -p 8010:8000 \
  -v /opt/hf-cache:/root/.cache/huggingface \
  -v $Q/worker_image_quant.py:$SP/vllm/v1/ple_offload/worker.py:ro \
  -v $Q/ple_layer_quant.py:$SP/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
  -v $Q/connector_mrv2.py:$SP/vllm/v1/ple_offload/connector.py:ro \
  -v $Q/ples_int4:/ples_int4:ro \
  -e VLLM_PLE_QUANT_DIR=/ples_int4 -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600 \
  -e VLLM_GDN_DECODE_KERNEL=triton \
  vllm/vllm-openai:qwen38-flash-next \
  --model primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8 \
  --served-model-name flash-next \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.92 --max-num-seqs 64 --max-model-len 135168 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  "${SPEC[@]}"
echo "flash-next-vllm avviato (MTP=$MTP) su :8010 — segui con: docker logs -f flash-next-vllm"
