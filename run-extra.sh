#!/usr/bin/env bash
# Catena "extra" (05/09/2026), dopo 'FINE tool-eval 27B thinking OFF':
#  A) 27B NVFP4 (in servizio): prefix-cache test (Hermes-like), concorrenza 32k c=8/16, context-pressure sweep 16k-64k
#  B) vLLM Flash-Next MTP on (miglior effort): idem
#  C) TabbyAPI/EXL3 MTP on: prefix-cache + concorrenza
#  D) 27B BF16 originale (se scaricato): tool-eval medium 8 trial (costo della quantizzazione NVFP4)
#  E) 27B NVFP4 riacceso; energia per fase (energy.py); FINE run-extra
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
B=/opt/flash-next/bench
R=$B/results
LOG=$B/logs/run-extra-$(date +%Y%m%d-%H%M).log
exec > >(tee -a "$LOG") 2>&1
say()    { echo; echo "##### $(date '+%d/%m/%Y %H:%M:%S') — $*"; }
notify() { sudo /usr/local/bin/pizeta-notify.sh "🧪 srv-lm extra — $*" >/dev/null 2>&1 || true; }
score()  { grep -oE '\*\*[0-9.]+ ± [0-9.]+\*\*' "$1" | head -1 | tr -d '*'; }
lines()  { grep -E '\*\*Final Score\*\*|\*\*Pass@8\*\*|\*\*Pass\^8\*\*' "$1" | tr -s ' ' | cut -c1-200; }
TE="tool-eval-bench run --seed 42 --hardmode --temperature 0.6 --timeout 600"

prefix() { # NAME BASE MODEL
  say "PREFIX-CACHE $1 (system ~4,7k tok identico, 8 domande diverse, c=1)"
  python3 $B/scripts/prefix-cache-test.py "$2" "$3" $R/prefix_$1.json 8
}
conc() { # NAME PORT MODEL
  for C in 8 16; do
    say "CONCORRENZA $1: 32k input, out 256, c=$C ($((C*2)) richieste)"
    docker run --rm --network host -v $R:/results -v /opt/flash-next/exl3-4.05bpw:/tok:ro lmsysorg/sglang:dev-cu13 \
      python3 -m sglang.benchmark.serving --backend vllm-chat --host 127.0.0.1 --port "$2" --model "$3" --tokenizer /tok \
      --dataset-name random --random-input-len 32768 --random-output-len 256 --random-range-ratio 0 \
      --num-prompts $((C*2)) --max-concurrency $C --warmup-requests 1 --seed $((3200+C)) \
      --extra-request-body '{"temperature":0.6,"top_p":0.95,"chat_template_kwargs":{"enable_thinking":false}}' \
      --output-file /results/conc_$1_in32768_out256_c$C.json --output-details 2>&1 | grep -E 'Successful|Output token throughput|Median TTFT|P99 TTFT|Median E2E|P99 E2E' | sed 's/^/  /'
  done
}
sweep() { # NAME BASE MODEL LABEL [extra args]
  say "CONTEXT-PRESSURE SWEEP $1: 25/50/75/100% di 65.536 tok, 2 trial"
  cd $B/runs && $TE --base-url "$2" --model "$3" --trials 2 --context-pressure-sweep 0.25-1.0 --sweep-steps 4 --context-size 65536 --label "$4" "${@:5}" 2>&1 | tail -30; cd /
  ls -t $B/runs/runs/2026/09/*summary.md 2>/dev/null | head -1
}

say "INIZIO run-extra — attendo FINE tool-eval 27B thinking OFF"
while ! grep -aq 'FINE tool-eval 27B thinking OFF' $(ls -t $B/logs/tooleval-27b-nothink-*.log | head -1) 2>/dev/null; do sleep 60; done
BEST=$(grep -aoE 'miglior effort: [a-z]+' $(ls -t $B/logs/run-effort-*.log | head -1) | tail -1 | awk '{print $3}'); BEST=${BEST:-xhigh}
say "miglior effort Flash-Next: $BEST"
notify "parte run-extra (~4-5 h): 27B prefix/concorrenza/sweep → Flash-Next vLLM ($BEST) idem → EXL3 → 27B BF16 → energia."

# ---- A) 27B NVFP4 in servizio
for i in $(seq 1 30); do curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null && break; sleep 10; done
prefix 27b-nvfp4 http://127.0.0.1:8000 qwen3.8-27b
conc 27b-nvfp4 8000 qwen3.8-27b
sweep 27b-nvfp4 http://127.0.0.1:8000/v1 qwen3.8-27b "SGLang, RadixArk Qwen3.8-27B NVFP4, thinking medium (template), context-pressure sweep 16k-64k"
notify "✅ 27B: prefix/concorrenza/sweep fatti. Ora Flash-Next su vLLM (27B fermo ~2 h)."

# ---- B) vLLM Flash-Next MTP on
say "STOP 27B → vLLM Flash-Next MTP on (alias)"
docker stop sglang-workhorse >/dev/null; docker rm -f flash-next-vllm >/dev/null 2>&1
sed 's/--served-model-name flash-next/--served-model-name flash-next qwen3.8-27b/' $B/configs/vllm-flash-next.sh > /tmp/vllm-alias.sh
bash /tmp/vllm-alias.sh on | tail -1
if bash $B/scripts/wait-ready.sh 8010 900 flash-next-vllm; then
  prefix flash-next-vllm http://127.0.0.1:8010 flash-next
  conc flash-next-vllm 8010 flash-next
  sweep flash-next-vllm http://127.0.0.1:8010/v1 flash-next "vLLM 0.1.dev20073 MTP on, PLE INT4 offload, reasoning_effort=$BEST, context-pressure sweep 16k-64k" --backend-kwargs "{\"chat_template_kwargs\":{\"reasoning_effort\":\"$BEST\"}}"
  notify "✅ Flash-Next vLLM: prefix/concorrenza/sweep fatti."
else docker logs --tail 40 flash-next-vllm | cut -c1-200; notify "❌ vLLM non parte: salto B"; fi
docker rm -f flash-next-vllm >/dev/null 2>&1

# ---- C) TabbyAPI / EXL3 MTP on
say "TabbyAPI/EXL3 MTP on"
bash $B/configs/tabby-flash-next.sh on | tail -1
if bash $B/scripts/wait-ready.sh 8011 900 flash-next-tabby; then
  prefix flash-next-exl3 http://127.0.0.1:8011 flash-next
  conc flash-next-exl3 8011 flash-next
  echo "--- errori server tabby ---"; docker logs flash-next-tabby 2>&1 | grep -aiE 'error|exception' | grep -av server_info | tail -8 | cut -c1-200
  notify "✅ EXL3: prefix/concorrenza fatti."
else docker logs --tail 40 flash-next-tabby | cut -c1-200; notify "❌ Tabby non parte: salto C"; fi
docker rm -f flash-next-tabby >/dev/null 2>&1

# ---- D) 27B BF16 originale
say "27B BF16 (Qwen/Qwen3.8-27B): download completo?"
NST=$(ls /opt/hf-cache/hub/models--Qwen--Qwen3.8-27B/snapshots/*/*.safetensors 2>/dev/null | wc -l)
if [ "$NST" -ge 18 ] && ! pgrep -f 'hf download Qwen/Qwen3.8-27B' >/dev/null; then
  bash $B/configs/run-sglang-27b-bf16.sh
  if bash $B/scripts/wait-ready.sh 8000 1500 sglang-27b-bf16; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/VRAM: /'
    say "TOOL-EVAL 27B BF16, thinking medium, 8 trial"
    cd $B/runs && $TE --base-url http://127.0.0.1:8000/v1 --model qwen3.8-27b --trials 8 --label "SGLang dev-cu13, Qwen/Qwen3.8-27B BF16 originale, KV bf16, NEXTN 2, thinking medium (template)" 2>&1 | tail -20; cd /
    S=$(ls -t $B/runs/runs/2026/09/*summary.md | head -1); echo "summary BF16: $S"; lines "$S"
    notify "✅ 27B BF16 medium: $(score "$S") (NVFP4 medium: 91,0 ± 1,5)"
  else docker logs --tail 60 sglang-27b-bf16 | cut -c1-200; notify "❌ 27B BF16 non parte: salto D"; fi
  docker rm -f sglang-27b-bf16 >/dev/null 2>&1
else echo "safetensors presenti: $NST — download non completo, salto D"; notify "⚠️ 27B BF16 non ancora scaricato ($NST/18 shard): salto D, da rifare a parte."; fi

# ---- E) 27B NVFP4 di nuovo in servizio + energia
say "RIAVVIO 27B NVFP4 (script canonico)"
bash /home/pizeta/vllm/run-sglang-38.sh >/dev/null 2>&1
for i in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "27B READY dopo ~$((i*10)) s"; break; }; sleep 10; done
say "ENERGIA per fase"
python3 $B/scripts/energy.py $B | tail -40
say "FINE run-extra"
notify "FINE run-extra. 27B $(curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo riacceso || echo NON RISPONDE)."
