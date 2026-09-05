#!/usr/bin/env bash
# Campagna "reasoning effort" (05/09/2026): miglior settaggio di Flash-Next e conferma sui due motori.
#  1. attende la fine del tool-eval sul 27B (medium, template di prod)
#  2. vLLM MTP on: tool-eval a reasoning_effort=medium e =low (xhigh gia' fatto ieri: 85,6)
#  3. sceglie il miglior effort di Flash-Next e lo misura anche su TabbyAPI/EXL3 MTP on
#  4. test vision/tool-call/idle su vLLM, poi riaccende il 27B
# Ogni tool-eval ha --label (nel report e nel nome file) cosi' i report sono autoesplicativi per la pubblicazione.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
B=/opt/flash-next/bench
LOG=$B/logs/run-effort-$(date +%Y%m%d-%H%M).log
exec > >(tee -a "$LOG") 2>&1
say()    { echo; echo "##### $(date '+%d/%m/%Y %H:%M:%S') — $*"; }
notify() { sudo /usr/local/bin/pizeta-notify.sh "🧪 srv-lm Flash-Next effort — $*" >/dev/null 2>&1 || true; }
score()  { grep -oE '\*\*[0-9.]+ ± [0-9.]+\*\*' "$1" | head -1 | tr -d '*'; }
lines()  { grep -E '\*\*Final Score\*\*|\*\*Pass@8\*\*|\*\*Pass\^8\*\*' "$1" | tr -s ' ' | cut -c1-200; }
TE="tool-eval-bench run --seed 42 --trials 8 --hardmode --temperature 0.6 --timeout 600"

say "INIZIO run-effort — log $LOG; attendo la fine del tool-eval sul 27B"
while pgrep -f 'tool-eval-bench run' >/dev/null; do sleep 30; done
say "tool-eval 27B finito"
S27=$(ls -t $B/runs/runs/2026/09/*summary.md | head -1); echo "summary 27B: $S27"; lines "$S27"
notify "tool-eval 27B (medium) finito: $(score "$S27"). Ora Flash-Next medium/low su vLLM, poi EXL3 al migliore (~2,5 h), 27B fermo."

say "STOP 27B, START vLLM Flash-Next MTP on (alias qwen3.8-27b)"
docker stop sglang-workhorse >/dev/null; docker rm -f flash-next-vllm >/dev/null 2>&1
sed 's/--served-model-name flash-next/--served-model-name flash-next qwen3.8-27b/' $B/configs/vllm-flash-next.sh > /tmp/vllm-alias.sh
bash /tmp/vllm-alias.sh on | tail -1
if ! bash $B/scripts/wait-ready.sh 8010 900 flash-next-vllm; then docker logs --tail 40 flash-next-vllm | cut -c1-200; notify "❌ vLLM non parte, riaccendo il 27B"; bash /home/pizeta/vllm/run-sglang-38.sh; exit 1; fi

mkdir -p $B/runs && cd $B/runs
declare -A SC; SC[xhigh]=85.6
for EFF in medium low; do
  say "TOOL-EVAL Flash-Next vLLM MTP on — reasoning_effort=$EFF (thinking ON)"
  N0=$(docker logs flash-next-vllm 2>&1 | grep -ac 'Failed to advance FSM')
  $TE --base-url http://127.0.0.1:8010/v1 --model flash-next \
    --backend-kwargs "{\"chat_template_kwargs\":{\"reasoning_effort\":\"$EFF\"}}" \
    --label "vLLM 0.1.dev20073 MTP on, PLE INT4 offload, reasoning_effort=$EFF, thinking on" 2>&1 | tail -20
  N1=$(docker logs flash-next-vllm 2>&1 | grep -ac 'Failed to advance FSM')
  S=$(ls -t $B/runs/runs/2026/09/*summary.md | head -1); echo "summary $EFF: $S"; lines "$S"
  SC[$EFF]=$(score "$S" | cut -d' ' -f1); echo "errori xgrammar durante $EFF: $((N1-N0))"
  notify "✅ Flash-Next vLLM effort=$EFF: $(score "$S") (xgrammar +$((N1-N0)))"
done
cd /
BEST=xhigh; for e in medium low; do awk -v a="${SC[$e]:-0}" -v b="${SC[$BEST]}" 'BEGIN{exit !(a>b)}' && BEST=$e; done
say "punteggi Flash-Next su vLLM: xhigh=${SC[xhigh]} medium=${SC[medium]:-n/d} low=${SC[low]:-n/d} → miglior effort: $BEST"

say "conferma su EXL3: TabbyAPI MTP on, reasoning_effort=$BEST"
docker rm -f flash-next-vllm >/dev/null 2>&1
bash $B/configs/tabby-flash-next.sh on | tail -1
if bash $B/scripts/wait-ready.sh 8011 900 flash-next-tabby; then
  cd $B/runs
  $TE --base-url http://127.0.0.1:8011/v1 --model flash-next \
    --backend-kwargs "{\"chat_template_kwargs\":{\"reasoning_effort\":\"$BEST\"}}" \
    --label "TabbyAPI / ExLlamaV3 1.4.6 EXL3 4.05bpw, MTP on, ngram_ram, reasoning_effort=$BEST, thinking on" 2>&1 | tail -20
  S=$(ls -t $B/runs/runs/2026/09/*summary.md | head -1); echo "summary EXL3 $BEST: $S"; lines "$S"
  echo "--- errori server tabby ---"; docker logs flash-next-tabby 2>&1 | grep -aiE 'error|exception' | grep -av server_info | tail -5 | cut -c1-200
  notify "✅ EXL3 effort=$BEST: $(score "$S")"
  cd /
else
  docker logs --tail 40 flash-next-tabby | cut -c1-200; notify "❌ Tabby non parte: salto il run EXL3"
fi
docker rm -f flash-next-tabby >/dev/null 2>&1

say "test vision / tool-call / idle su vLLM (riaccende il 27B alla fine)"
bash /tmp/test-vision-idle.sh
say "FINE run-effort"
notify "FINE run-effort: 27B $(curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo riacceso || echo NON RISPONDE)."
