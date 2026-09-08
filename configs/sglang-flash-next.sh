#!/usr/bin/env bash
# Via C — SGLang + RadixArk/Qwen3.8-Flash-Next-NVFP4 su 1x RTX PRO 6000, tabella n-gram FP8 (47,7 GiB) in RAM pinnata.
# Fonte: cella verificata del cookbook SGLang (docs/src/snippets/configs/Qwen/qwen3.8-flash-next.jsx, hw=rtx6000,
#        strategy=low-latency, PR #37995 del 07/09/2026) + immagine lmsysorg/sglang:dev-qwen38-next-local (build 4ccff141db).
# Uso: sglang-flash-next.sh on|off [PORT]   (MTP=NEXTN on/off; default porta 8012)
set -euo pipefail
MTP="${1:?uso: sglang-flash-next.sh on|off [PORT]}"; PORT="${2:-8012}"
SPEC=(); MRR=16; MMC=48; MEM=0.96
if [ "$MTP" = on ]; then
  SPEC=(--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
else
  MRR=64; MMC=192; MEM=0.93     # cella high-throughput: senza speculazione, 64 richieste, 192 slot bf16
fi
# PLE_BACKEND=pinned (default cookbook: 47,7 GiB in RAM pinnata, serve >=64 GB liberi) | file (mmap sparso su NVMe,
#   pensato per GB10; qui forzato con SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1 perche' la RTX PRO 6000 ha HMM ma non
#   PageableMemoryAccessUsesHostPageTables — 08/09/2026)
PLE_BACKEND="${PLE_BACKEND:-pinned}"; PLEARGS=(); PLEENV=()
if [ "$PLE_BACKEND" = file ]; then
  sudo -n mkdir -p /opt/flash-next/ple-cache && sudo -n chown "$(id -u):$(id -g)" /opt/flash-next/ple-cache
  rm -rf /opt/flash-next/ple-cache/*   # il cookbook: cancellare il file prima di ogni boot (riscrittura a 17 MB/s)
  PLEARGS=(--ple-offload-backend file --ple-offload-dir /ple-cache)
  PLEENV=(-e SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1 -v /opt/flash-next/ple-cache:/ple-cache)
fi
docker rm -f flash-next-sglang 2>/dev/null || true
docker run -d --name flash-next-sglang --gpus all --ipc=host --shm-size 32g --ulimit memlock=-1 \
  -p ${PORT}:8000 \
  -v /opt/hf-cache:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 \
  "${PLEENV[@]}" \
  lmsysorg/sglang:dev-qwen38-next-local \
  python3 -m sglang.launch_server \
  --model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name flash-next \
  --host 0.0.0.0 --port 8000 \
  --tp 1 \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend flashinfer_cutlass \
  --moe-runner-backend flashinfer_cutlass \
  --page-size 64 \
  --mamba-track-interval 64 \
  --chunked-prefill-size 4096 \
  --context-length 262144 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-running-requests $MRR \
  --max-mamba-cache-size $MMC \
  --mamba-ssm-dtype bfloat16 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mem-fraction-static $MEM \
  --ple-offload-embedding \
  "${PLEARGS[@]}" \
  --enable-metrics \
  "${SPEC[@]}"
echo "flash-next-sglang avviato (MTP=$MTP, PLE=$PLE_BACKEND, max-running-requests=$MRR, mem=$MEM) su :$PORT — segui con: docker logs -f flash-next-sglang"
