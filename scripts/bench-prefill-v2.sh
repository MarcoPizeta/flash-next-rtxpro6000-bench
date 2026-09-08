#!/usr/bin/env bash
# Prefill isolato v2: input {8k,32k,128k}, output 1 token, c=1, 3 run -> prefill tok/s = input / TTFT.
# v2: --seed unico per run e NESSUN warmup (i kernel sono gia' caldi dopo la matrice; il warmup
# riusava il primo prompt e ne metteva il prefill in cache). I 3 TTFT sono tutti "freddi".
# Uso: bench-prefill-v2.sh ENGINE PORT MTP
set -uo pipefail
ENGINE=$1; PORT=$2; MTP=$3
B=/opt/flash-next/bench; TOK="${TOK:-/opt/flash-next/exl3-4.05bpw}"
TAG="${ENGINE}_mtp-${MTP}"
EXTRA='{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
for IN in 8192 32768 131072; do for RUN in 1 2 3; do
  SEED=$(( IN / 1024 * 100 + 90 + RUN ))
  OUT="/results/prefill_${TAG}_in${IN}_r${RUN}.json"
  echo "=== $(date '+%H:%M:%S') PREFILL $TAG in=$IN run=$RUN seed=$SEED ==="
  docker run --rm --network host \
    -v $B/results:/results -v /opt/hf-cache:/opt/hf-cache:ro \
    lmsysorg/sglang:dev-cu13 python3 -m sglang.benchmark.serving \
      --backend vllm-chat --host 127.0.0.1 --port "$PORT" --model flash-next --tokenizer "$TOK" \
      --dataset-name random --random-input-len "$IN" --random-output-len 1 --random-range-ratio 0 \
      --num-prompts 3 --max-concurrency 1 --warmup-requests 0 --seed "$SEED" \
      --extra-request-body "$EXTRA" \
      --output-file "$OUT" --output-details 2>&1 | grep -E 'Median TTFT|Mean TTFT|Successful' | sed 's/^/  /'
  sleep 5
done; done
