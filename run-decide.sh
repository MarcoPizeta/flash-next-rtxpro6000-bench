#!/usr/bin/env bash
# Catena "decide" (05/09/2026): i test che mancano per scegliere il workhorse. Dopo 'FINE run-exl3-5bpw':
#  1) 27B NVFP4 in servizio su :8000 → AIME 2025+2026 (60 problemi, xhigh, c=4) + A/B Pizeta (RAG reale via pizeta-rag,
#     6 foto DDT reali, 3 casi NC) a effort medium
#  2) stop 27B → vLLM Flash-Next MTP on SULLA PORTA 8000 con alias qwen3.8-27b (così pizeta-rag/gateway lo usano
#     senza toccare nulla) → AIME xhigh + A/B a effort medium
#  3) 27B riacceso → pagina HTML per il giudizio alla cieca → FINE run-decide
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
B=/opt/flash-next/bench; E=/opt/flash-next/eval; PY=/opt/flash-next/venv/bin/python
LOG=$B/logs/run-decide-$(date +%Y%m%d-%H%M).log
exec > >(tee -a "$LOG") 2>&1
say()    { echo; echo "##### $(date '+%d/%m/%Y %H:%M:%S') — $*"; }
notify() { sudo /usr/local/bin/pizeta-notify.sh "🧪 srv-lm decide — $*" >/dev/null 2>&1 || true; }
mkdir -p $E/out

say "INIZIO run-decide — attendo FINE run-exl3-5bpw"
while ! grep -aq 'FINE run-exl3-5bpw' $(ls -t $B/logs/run-exl3-5bpw-*.log | head -1) 2>/dev/null; do sleep 60; done
for i in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 10; done
notify "parte run-decide (~2,5 h): AIME + A/B Pizeta sul 27B, poi su Flash-Next (27B fermo ~1,5 h)."

say "27B — AIME 2025+2026, xhigh, c=4"
$PY $E/reasoning-eval.py http://127.0.0.1:8000 qwen3.8-27b xhigh $E/out/aime_27b-nvfp4_xhigh.jsonl 4 32000 2>&1 | tail -8
say "27B — A/B Pizeta (RAG reale + DDT + NC), effort medium"
$PY $E/ab-gen.py 27b-nvfp4 medium $E/out/ab_27b-nvfp4.jsonl 2>&1 | tail -22
notify "✅ 27B: AIME $(grep -oE 'RISULTATO.*' $LOG | tail -1 | cut -c1-80). Ora Flash-Next su :8000."

say "STOP 27B → vLLM Flash-Next MTP on su :8000 (alias qwen3.8-27b)"
docker stop sglang-workhorse >/dev/null; docker rm -f flash-next-vllm >/dev/null 2>&1
sed 's/--served-model-name flash-next/--served-model-name flash-next qwen3.8-27b/; s/-p 8010:8000/-p 8000:8000/' $B/configs/vllm-flash-next.sh > /tmp/vllm-8000.sh
grep -c '8000:8000' /tmp/vllm-8000.sh | sed 's/^/  mappatura 8000: /'
bash /tmp/vllm-8000.sh on | tail -1
if bash $B/scripts/wait-ready.sh 8000 900 flash-next-vllm; then
  curl -sf http://127.0.0.1:8000/v1/models | grep -o '"id":"[^"]*"' | tr '\n' ' '; echo
  say "Flash-Next — AIME 2025+2026, xhigh, c=4"
  $PY $E/reasoning-eval.py http://127.0.0.1:8000 qwen3.8-27b xhigh $E/out/aime_flash-next-vllm_xhigh.jsonl 4 32000 2>&1 | tail -8
  say "Flash-Next — prefix-cache test a modello CALDO (dopo AIME)"
  python3 $B/scripts/prefix-cache-test.py http://127.0.0.1:8000 qwen3.8-27b $B/results/prefix_flash-next-vllm-warm.json 8
  say "Flash-Next — A/B Pizeta, effort medium"
  $PY $E/ab-gen.py flash-next-vllm medium $E/out/ab_flash-next-vllm.jsonl 2>&1 | tail -22
  notify "✅ Flash-Next: AIME $(grep -oE 'RISULTATO.*' $LOG | tail -1 | cut -c1-80)."
else docker logs --tail 40 flash-next-vllm | cut -c1-200; notify "❌ vLLM su :8000 non parte: salto Flash-Next"; fi
docker rm -f flash-next-vllm >/dev/null 2>&1

say "RIAVVIO 27B"
bash /home/pizeta/vllm/run-sglang-38.sh >/dev/null 2>&1
for i in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "27B READY dopo ~$((i*10)) s"; break; }; sleep 10; done
say "pagina di giudizio alla cieca"
$PY $E/ab-judge.py $E/out/ab_27b-nvfp4.jsonl $E/out/ab_flash-next-vllm.jsonl $E/out/ab-judge.html && cp -r $E/ddt $E/out/ 2>/dev/null
echo "--- AIME riepilogo ---"; grep -aE '^RISULTATO' $LOG
say "FINE run-decide"
notify "FINE run-decide. 27B $(curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo riacceso || echo NON RISPONDE). Pagina A/B pronta."
