#!/usr/bin/env bash
# Dopo 'FINE run-effort' (27B riacceso): tool-eval sul 27B con thinking OFF, per il confronto alla pari
# con Flash-Next thinking off (vLLM 87,0 / EXL3 85,5). Non ferma nulla: gira sul 27B in servizio.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
B=/opt/flash-next/bench
LOG=$B/logs/tooleval-27b-nothink-$(date +%Y%m%d-%H%M).log
exec > >(tee -a "$LOG") 2>&1
notify() { sudo /usr/local/bin/pizeta-notify.sh "🧪 srv-lm 27B thinking off — $*" >/dev/null 2>&1 || true; }
echo "##### $(date '+%d/%m/%Y %H:%M:%S') — attendo FINE run-effort"
while ! grep -aq 'FINE run-effort' $(ls -t $B/logs/run-effort-*.log | head -1); do sleep 60; done
for i in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 10; done
curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null || { echo "27B non risponde: esco"; notify "❌ 27B non risponde, tool-eval thinking off saltato"; exit 1; }
echo "##### $(date '+%d/%m/%Y %H:%M:%S') — TOOL-EVAL 27B thinking OFF"
notify "parte il tool-eval sul 27B con thinking OFF (~30 min, 27B in servizio)"
mkdir -p $B/runs && cd $B/runs
tool-eval-bench run --base-url http://127.0.0.1:8000/v1 --model qwen3.8-27b --seed 42 --trials 8 --hardmode --temperature 0.6 --timeout 600 --no-think \
  --label "SGLang dev-cu13, RadixArk Qwen3.8-27B NVFP4, NEXTN 2, thinking OFF (enable_thinking=false)" 2>&1 | tail -20
S=$(ls -t $B/runs/runs/2026/09/*summary.md | head -1); echo "summary: $S"
grep -E '\*\*Final Score\*\*|\*\*Pass@8\*\*|\*\*Pass\^8\*\*' "$S" | tr -s ' ' | cut -c1-200
notify "✅ 27B thinking OFF: $(grep -oE '\*\*[0-9.]+ ± [0-9.]+\*\*' "$S" | head -1 | tr -d '*') (Flash-Next off: vLLM 87,0 / EXL3 85,5)"
echo "##### $(date '+%d/%m/%Y %H:%M:%S') — FINE tool-eval 27B thinking OFF"
