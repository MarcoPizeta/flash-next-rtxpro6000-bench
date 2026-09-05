#!/usr/bin/env bash
# Matrice core v2: input {1k,8k,32k} x output 512 x concorrenza {1,4} x 3 run.
# v2 (04/09/2026 sera): --seed UNICO per (input, concorrenza, run). Con il seed fisso 42 del client
# i 3 run avevano prompt identici e il prefill veniva servito dalla prefix cache (TTFT 90 ms su 131k).
# Resta il warmup (2 richieste sul PRIMO prompt del set): 1 richiesta su 6-20 ha TTFT sottostimato,
# la mediana non ne risente. Stesso seed tra motori/config => stesso set di prompt per l'A/B.
# Uso: bench-matrix-v2.sh ENGINE PORT MTP(on|off)
set -uo pipefail
ENGINE=$1; PORT=$2; MTP=$3
B=/opt/flash-next/bench
TAG="${ENGINE}_mtp-${MTP}"
EXTRA='{"temperature":0.6,"top_p":0.95,"chat_template_kwargs":{"enable_thinking":false}}'
for IN in 1024 8192 32768; do for CONC in 1 4; do for RUN in 1 2 3; do
  NP=$(( CONC * 5 )); [ $NP -lt 6 ] && NP=6
  SEED=$(( IN / 1024 * 100 + CONC * 10 + RUN ))
  OUT="/results/${TAG}_in${IN}_out512_c${CONC}_r${RUN}.json"
  echo "=== $(date '+%H:%M:%S') $TAG in=$IN c=$CONC run=$RUN seed=$SEED ==="
  docker run --rm --network host \
    -v $B/results:/results -v /opt/flash-next/exl3-4.05bpw:/tok:ro \
    lmsysorg/sglang:dev-cu13 python3 -m sglang.benchmark.serving \
      --backend vllm-chat --host 127.0.0.1 --port "$PORT" --model flash-next --tokenizer /tok \
      --dataset-name random --random-input-len "$IN" --random-output-len 512 --random-range-ratio 0 \
      --num-prompts "$NP" --max-concurrency "$CONC" --warmup-requests 2 --seed "$SEED" \
      --extra-request-body "$EXTRA" \
      --output-file "$OUT" --output-details 2>&1 | grep -E 'Output token throughput|Median TTFT|Median TPOT|Median ITL|Request throughput|Successful' | sed 's/^/  /'
  sleep 10
done; done; done
echo "=== matrice $TAG completata ==="
