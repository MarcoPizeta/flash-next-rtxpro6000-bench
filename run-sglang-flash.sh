#!/usr/bin/env bash
# Round "Flash-Next su SGLang" (08/09/2026, Pizeta chiusa, ok Marco):
#  attende download RadixArk + immagine dev-qwen38-next-local → stop 27B → SGLang Flash-Next MTP on (:8012)
#  → boot/RAM/KV → required (12 tool) → tool-eval low (88x8) → tool-eval thinking off (88x8)
#  → matrice v2 + prefill v2 + concorrenza 32k + prefix-cache → stop → riavvio 27B → FINE
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
B=/opt/flash-next/bench; R=$B/results; LOG=$B/logs/run-sglang-flash-$(date +%Y%m%d-%H%M).log
PORT=8012; BASE=http://127.0.0.1:$PORT
say(){ echo "##### $(date '+%d/%m/%Y %H:%M:%S') — $*" | tee -a "$LOG"; }
notify(){ sudo /usr/local/bin/pizeta-notify.sh "🧪 srv-lm sglang-flash — $*" >/dev/null 2>&1 || true; }
TE="tool-eval-bench run --seed 42 --trials 8 --hardmode --temperature 0.6 --timeout 600"
say "INIZIO run-sglang-flash — attendo download RadixArk e immagine"
for i in $(seq 1 240); do
  DL=$(cat /opt/flash-next/logs/prep-sglang-flash-*.log 2>/dev/null | grep -c 'download OK')
  IMG=$(docker images -q lmsysorg/sglang:dev-qwen38-next-local | wc -l)
  [ "$DL" -ge 1 ] && [ "$IMG" -ge 1 ] && break
  [ $((i % 20)) -eq 0 ] && say "attesa: download OK=$DL immagine=$IMG ckpt=$(du -sh /opt/hf-cache/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4 2>/dev/null | cut -f1)"
  sleep 60
done
[ "$DL" -ge 1 ] && [ "$IMG" -ge 1 ] || { say "TIMEOUT attesa download/immagine — esco senza toccare il 27B"; notify "❌ sglang-flash: download/immagine non pronti dopo 4 h"; exit 1; }
say "pronti: ckpt $(du -sh /opt/hf-cache/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4 | cut -f1), immagine $(docker images --format '{{.ID}} {{.CreatedAt}}' lmsysorg/sglang:dev-qwen38-next-local)"
notify "parte il round Flash-Next su SGLang (27B fermo ~3 h)"
say "STOP 27B"; docker stop sglang-workhorse >/dev/null; free -g | awk '/Mem/{print "RAM prima: used "$3" avail "$7" GB"}' | tee -a "$LOG"
say "svuoto la page cache (la tabella n-gram va pinnata: 47,7 GiB)"; sync; sudo -n sysctl -w vm.drop_caches=3 >/dev/null; free -g | awk '/Mem/{print "RAM dopo drop_caches: used "$3" free "$4" avail "$7" GB"}' | tee -a "$LOG"
say "PROVA: una sola allocazione pinnata da 47,7 GiB in un processo nuovo (quella che il boot 'pinned' fa)"
cat > /tmp/pin1.py <<'PY'
import torch, time, sys
gib = float(sys.argv[1]); t0 = time.time()
try:
    t = torch.empty(int(gib * 1024**3), dtype=torch.uint8, device="cpu", pin_memory=True); print(f"pinned {gib} GiB: OK in {time.time()-t0:.1f}s")
except Exception as e: print(f"pinned {gib} GiB: FALLITO -> {str(e)[:100]}")
PY
for G in 40 47.7; do docker run --rm --gpus all --ulimit memlock=-1 --entrypoint python3 -v /tmp/pin1.py:/p.py:ro lmsysorg/sglang:dev-qwen38-next-local /p.py $G 2>&1 | grep -v Warning | tee -a "$LOG"; done
free -g | awk '/Mem/{print "RAM dopo la prova: used "$3" free "$4" avail "$7" GB"}' | tee -a "$LOG"
export PLE_BACKEND="${PLE_BACKEND:-file}"
say "AVVIO SGLang Flash-Next MTP on — PLE_BACKEND=$PLE_BACKEND (file = mmap sparso su NVMe /opt/flash-next/ple-cache, device check saltato)"
T0=$(date +%s); bash $B/configs/sglang-flash-next.sh on $PORT | tee -a "$LOG"
sleep 3; docker exec flash-next-sglang sh -c 'ulimit -l' 2>/dev/null | sed 's/^/memlock nel container: /' | tee -a "$LOG"
if ! bash $B/scripts/wait-ready.sh $PORT 1500 flash-next-sglang | tee -a "$LOG"; then
  say "BOOT FALLITO — log server salvato in $B/logs/flash-next-sglang-boot-fail-$(date +%H%M).log; righe chiave:"
  docker logs flash-next-sglang > $B/logs/flash-next-sglang-boot-fail-$(date +%H%M).log 2>&1
  docker logs flash-next-sglang 2>&1 | grep -aiE 'error|Traceback|memlock|pinned|cannot|out of memory|KV' | tail -25 | cut -c1-220 | tee -a "$LOG"
  free -g | awk '/Mem/{print "RAM al fallimento: used "$3" free "$4" avail "$7" GB"}' | tee -a "$LOG"
  docker rm -f flash-next-sglang >/dev/null 2>&1; bash /home/pizeta/vllm/run-sglang-38.sh >/dev/null 2>&1
  notify "❌ sglang-flash: boot fallito, 27B in riavvio"; say "FINE (fallito)"; exit 1
fi
say "boot totale $(( $(date +%s) - T0 )) s"
docker logs flash-next-sglang 2>&1 | grep -aE 'ple|PLE|n-gram|ngram|pinned|KV Cache is allocated|max_total_num_tokens|Memory pool|mamba|Load weight|load weight|took|avail mem|Capture cuda graph end|speculative|nccl==' | grep -viE 'sample' | tail -30 | cut -c1-200 | tee -a "$LOG"
free -g | awk '/Mem/{print "RAM con Flash-Next su SGLang: used "$3" avail "$7" GB"}' | tee -a "$LOG"; grep -E 'Mlocked|Unevictable' /proc/meminfo | tee -a "$LOG"
nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader | sed 's/^/GPU: /' | tee -a "$LOG"
say "SANITY (la tabella via HMM deve dare testo sensato, non spazzatura)"
for Q in "Quanto fa 17 per 23? Rispondi solo con il numero." "Qual è la capitale della Francia? Una parola." "Elenca tre metalli comuni separati da virgola."; do
  curl -s -m 120 $BASE/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"flash-next\",\"max_tokens\":40,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"chat_template_kwargs\":{\"enable_thinking\":false}}" | grep -o '"content":"[^"]*"' | cut -c1-120 | sed "s/^/  Q: $Q → /" | tee -a "$LOG"
done
say "REQUIRED (12 tool, stream/non-stream, think on/off)"
/home/pizeta/.local/share/uv/tools/tool-eval-bench/bin/python $B/scripts/required-test.py $BASE flash-next 2>&1 | tee -a "$LOG"
say "TOOL-EVAL Flash-Next SGLang, reasoning_effort=low (88x8)"
cd $B/runs && $TE --base-url $BASE/v1 --model flash-next --label "SGLang dev-qwen38-next-local, RadixArk Flash-Next NVFP4, NEXTN 3/4, PLE pinned RAM, reasoning_effort=low, thinking on" --backend-kwargs '{"chat_template_kwargs":{"reasoning_effort":"low"}}' 2>&1 | tail -20 | tee -a "$LOG"; cd /
notify "sglang-flash: tool-eval low fatto → $(ls -t $B/runs/runs/2026/09/*_summary.md | head -1 | xargs grep -m1 'Final Score' | grep -oE '\*\*[0-9.]+ ± [0-9.]+\*\*')"
say "TOOL-EVAL Flash-Next SGLang, thinking OFF (88x8)"
cd $B/runs && $TE --base-url $BASE/v1 --model flash-next --no-think --label "SGLang dev-qwen38-next-local, RadixArk Flash-Next NVFP4, NEXTN 3/4, PLE pinned RAM, thinking OFF" 2>&1 | tail -20 | tee -a "$LOG"; cd /
say "MATRICE v2 (1k/8k/32k x c1/c4 x 3 run)"; bash $B/scripts/bench-matrix-v2.sh sglang $PORT on 2>&1 | tee -a "$LOG"
say "PREFILL v2 (8k/32k/128k x 3)"; bash $B/scripts/bench-prefill-v2.sh sglang $PORT on 2>&1 | tee -a "$LOG"
say "CONCORRENZA 32k c=8/16"
for C in 8 16; do
  docker run --rm --network host -v $R:/results -v /opt/flash-next/exl3-4.05bpw:/tok:ro lmsysorg/sglang:dev-cu13-20260827 python3 -m sglang.benchmark.serving \
    --backend vllm-chat --host 127.0.0.1 --port $PORT --model flash-next --tokenizer /tok --dataset-name random --random-input-len 32768 --random-output-len 256 --random-range-ratio 0 \
    --num-prompts $((C*2)) --max-concurrency $C --warmup-requests 1 --seed $((3200+C)) --extra-request-body '{"temperature":0.6,"top_p":0.95,"chat_template_kwargs":{"enable_thinking":false}}' \
    --output-file /results/conc_flash-next-sglang_in32768_out256_c$C.json --output-details 2>&1 | grep -E 'Successful|Output token throughput|Median TTFT|P99 TTFT|Median TPOT' | sed 's/^/  /' | tee -a "$LOG"
done
say "PREFIX-CACHE"; python3 $B/scripts/prefix-cache-test.py $BASE flash-next $R/prefix_flash-next-sglang.json 8 2>&1 | tail -4 | tee -a "$LOG"
say "STOP Flash-Next SGLang, RIAVVIO 27B"
docker logs flash-next-sglang 2>&1 | grep -acE 'Failed to advance|xgrammar.*[Ee]rror|Traceback' | sed 's/^/righe errore/traceback nel log server: /' | tee -a "$LOG"
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
say "FINE run-sglang-flash — log: $LOG"
