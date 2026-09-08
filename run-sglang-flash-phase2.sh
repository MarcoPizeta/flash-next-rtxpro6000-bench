#!/usr/bin/env bash
# Fase 2 del round SGLang (08/09/2026): velocità. Il server flash-next-sglang è già acceso su :8012.
# Fix: il tokenizer del client era /opt/flash-next/exl3-4.05bpw (cancellato stamattina) → snapshot RadixArk Flash-Next.
set -uo pipefail
B=/opt/flash-next/bench; R=$B/results; LOG=$B/logs/run-sglang-flash-phase2-$(date +%Y%m%d-%H%M).log
PORT=8012; BASE=http://127.0.0.1:$PORT
say(){ echo "##### $(date '+%d/%m/%Y %H:%M:%S') — $*" | tee -a "$LOG"; }
notify(){ sudo /usr/local/bin/pizeta-notify.sh "🧪 srv-lm sglang-flash — $*" >/dev/null 2>&1 || true; }
pkill -f 'bash run-sglang-flash.sh' && say "orchestratore fase 1 fermato" || say "nessun orchestratore fase 1 attivo"
pkill -f 'sglang.benchmark.serving' 2>/dev/null; sleep 2
export TOK=$(ls -d /opt/hf-cache/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/*/ | head -1)
say "tokenizer: $TOK"; ls -la $TOK | grep -E 'tokenizer|vocab|merges' | awk '{print "  "$9" -> "$11}' | tee -a "$LOG"
# patch bench-matrix-v2 / bench-prefill-v2: tokenizer da $TOK (default il vecchio percorso, per compatibilità)
for S in bench-matrix-v2.sh bench-prefill-v2.sh; do
  grep -q 'TOK=' $B/scripts/$S || sed -i 's#^B=/opt/flash-next/bench$#B=/opt/flash-next/bench; TOK="${TOK:-/opt/flash-next/exl3-4.05bpw}"#' $B/scripts/$S
  sed -i 's#-v /opt/flash-next/exl3-4.05bpw:/tok:ro#-v /opt/hf-cache:/opt/hf-cache:ro#; s#-v "$TOK":/tok:ro#-v /opt/hf-cache:/opt/hf-cache:ro#; s#--tokenizer /tok#--tokenizer "$TOK"#' $B/scripts/$S
  grep -nE 'TOK' $B/scripts/$S | cut -c1-140 | tee -a "$LOG"
done
curl -sf -m 5 $BASE/v1/models | grep -o '"id":"[^"]*"' | tee -a "$LOG" || { say "SERVER NON RISPONDE — esco"; exit 1; }
say "PROVA di un punto (1k x1, 1 run) prima della matrice"
P=$(docker run --rm --network host -v $R:/results -v /opt/hf-cache:/opt/hf-cache:ro lmsysorg/sglang:dev-cu13 python3 -m sglang.benchmark.serving --backend vllm-chat --host 127.0.0.1 --port $PORT --model flash-next --tokenizer "$TOK" --dataset-name random --random-input-len 1024 --random-output-len 128 --random-range-ratio 0 --num-prompts 2 --max-concurrency 1 --warmup-requests 0 --seed 7 --extra-request-body '{"temperature":0.6,"chat_template_kwargs":{"enable_thinking":false}}' 2>&1 | grep -E 'Successful|Output token throughput|Error|error' | head -4)
echo "$P" | tee -a "$LOG"
echo "$P" | grep -q 'Successful requests' || { say "IL CLIENT NON PRODUCE RISULTATI — mi fermo senza toccare il server"; notify "❌ sglang-flash fase 2: client bench non funziona"; exit 1; }
notify "sglang-flash fase 2 (velocità) partita"
say "MATRICE v2 (1k/8k/32k x c1/c4 x 3 run)"; bash $B/scripts/bench-matrix-v2.sh sglang $PORT on 2>&1 | tee -a "$LOG"
say "PREFILL v2 (8k/32k/128k x 3)"; bash $B/scripts/bench-prefill-v2.sh sglang $PORT on 2>&1 | tee -a "$LOG"
say "CONCORRENZA 32k c=8/16"
for C in 8 16; do
  docker run --rm --network host -v $R:/results -v /opt/hf-cache:/opt/hf-cache:ro lmsysorg/sglang:dev-cu13 python3 -m sglang.benchmark.serving \
    --backend vllm-chat --host 127.0.0.1 --port $PORT --model flash-next --tokenizer "$TOK" --dataset-name random --random-input-len 32768 --random-output-len 256 --random-range-ratio 0 \
    --num-prompts $((C*2)) --max-concurrency $C --warmup-requests 1 --seed $((3200+C)) --extra-request-body '{"temperature":0.6,"top_p":0.95,"chat_template_kwargs":{"enable_thinking":false}}' \
    --output-file /results/conc_flash-next-sglang_in32768_out256_c$C.json --output-details 2>&1 | grep -E 'Successful|Output token throughput|Median TTFT|P99 TTFT|Median TPOT' | sed 's/^/  /' | tee -a "$LOG"
done
say "PREFIX-CACHE"; python3 $B/scripts/prefix-cache-test.py $BASE flash-next $R/prefix_flash-next-sglang.json 8 2>&1 | tail -4 | tee -a "$LOG"
say "STOP Flash-Next SGLang, RIAVVIO 27B"
docker logs flash-next-sglang 2>&1 | grep -acE 'Failed to advance|xgrammar.*[Ee]rror|Traceback' | sed 's/^/righe errore-traceback nel log server: /' | tee -a "$LOG"
docker logs flash-next-sglang > $B/logs/flash-next-sglang-server-$(date +%Y%m%d-%H%M).log 2>&1
docker rm -f flash-next-sglang >/dev/null 2>&1
bash /home/pizeta/vllm/run-sglang-38.sh >/dev/null 2>&1
for i in $(seq 1 60); do
  if curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    RR=$(curl -s -m 90 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Rispondi solo PONG"}],"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}' | grep -o '"content":"[^"]*"')
    say "27B READY dopo $((i*10))s — $RR"; break
  fi; sleep 10
done
notify "✅ sglang-flash: round finito, 27B riacceso"
say "FINE fase 2 — log: $LOG"
