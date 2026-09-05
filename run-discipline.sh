#!/usr/bin/env bash
# Sesta catena (05/09/2026): il prompt puo' correggere la disciplina di Flash-Next sui tool?
# Dopo 'FINE run-decide': vLLM MTP on con chat template "discipline" (regole da agente nel blocco tools,
# default effort low) → tool-eval sugli 8 scenari critici (8 trial) → tool-eval completo 88 (8 trial) → 27B.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
B=/opt/flash-next/bench
LOG=$B/logs/run-discipline-$(date +%Y%m%d-%H%M).log
exec > >(tee -a "$LOG") 2>&1
say()    { echo; echo "##### $(date '+%d/%m/%Y %H:%M:%S') — $*"; }
notify() { sudo /usr/local/bin/pizeta-notify.sh "🧪 srv-lm discipline — $*" >/dev/null 2>&1 || true; }
score()  { grep -oE '\*\*[0-9.]+ ± [0-9.]+\*\*' "$1" | head -1 | tr -d '*'; }
lines()  { grep -E '\*\*Final Score\*\*|\*\*Pass@8\*\*|\*\*Pass\^8\*\*' "$1" | tr -s ' ' | cut -c1-200; }
TE="tool-eval-bench run --seed 42 --trials 8 --hardmode --temperature 0.6 --timeout 600 --base-url http://127.0.0.1:8010/v1 --model flash-next"

say "INIZIO run-discipline — attendo FINE run-decide"
while ! grep -aq 'FINE run-decide' $(ls -t $B/logs/run-decide-*.log | head -1) 2>/dev/null; do sleep 60; done
notify "parte run-discipline (~1,3 h): Flash-Next con template disciplina, 27B fermo."
say "STOP 27B → vLLM Flash-Next MTP on con chat_template_flash_discipline.jinja"
docker stop sglang-workhorse >/dev/null; docker rm -f flash-next-vllm >/dev/null 2>&1
bash $B/configs/vllm-flash-next-discipline.sh on | tail -1
if ! bash $B/scripts/wait-ready.sh 8010 900 flash-next-vllm; then docker logs --tail 60 flash-next-vllm | cut -c1-200; notify "❌ vLLM discipline non parte"; bash /home/pizeta/vllm/run-sglang-38.sh; exit 1; fi
docker logs flash-next-vllm 2>&1 | grep -aiE 'chat template|chat_template' | head -3 | cut -c1-160

say "TOOL-EVAL 8 scenari critici (TC-45 55 80 85 43 49 50 57), effort default (low), 8 trial"
cd $B/runs && $TE --scenarios TC-45 TC-55 TC-80 TC-85 TC-43 TC-49 TC-50 TC-57 --label "vLLM MTP on, chat template DISCIPLINE (agent rules in tools block, default effort low), thinking on, 8 critical scenarios" 2>&1 | tail -14; cd /
S=$(ls -t $B/runs/runs/2026/09/*summary.md | head -1); echo "summary 8 scenari: $S"
grep -E '^\| TC-(45|55|80|85|43|49|50|57) ' "$S" | tr -s ' ' | cut -c1-120
notify "8 scenari critici con disciplina: $(score "$S")"

say "TOOL-EVAL completo 88 scenari, effort default (low), 8 trial"
cd $B/runs && $TE --label "vLLM 0.1.dev20073 MTP on, PLE INT4 offload, chat template DISCIPLINE, reasoning_effort=low (default), thinking on" 2>&1 | tail -20; cd /
S=$(ls -t $B/runs/runs/2026/09/*summary.md | head -1); echo "summary 88: $S"; lines "$S"
echo "errori xgrammar: $(docker logs flash-next-vllm 2>&1 | grep -ac 'Failed to advance FSM')"
notify "✅ Flash-Next DISCIPLINE (88 scenari, low): $(score "$S") — senza disciplina low: 87,9 ± 1,9; 27B medium: 91,0 ± 1,5"
docker rm -f flash-next-vllm >/dev/null 2>&1

say "RIAVVIO 27B"
bash /home/pizeta/vllm/run-sglang-38.sh >/dev/null 2>&1
for i in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "27B READY dopo ~$((i*10)) s"; break; }; sleep 10; done
say "FINE run-discipline"
notify "FINE run-discipline. 27B $(curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo riacceso || echo NON RISPONDE)."
