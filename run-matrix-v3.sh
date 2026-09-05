#!/usr/bin/env bash
# Orchestratore v3 (04/09/2026, sera): SOLO matrici + prefill sui 4 config con gli script v2 (seed unico).
# I risultati della prima notte (seed fisso 42, prefill in cache) vanno in results-v1-seed42/.
# Il 27B viene riacceso in ogni caso alla fine (trap su EXIT).
set -uo pipefail
B=/opt/flash-next/bench
TS=$(date +%Y%m%d-%H%M)
LOG=$B/logs/run-matrix-v3-$TS.log
exec > >(tee -a "$LOG") 2>&1
say()    { echo; echo "##### $(date '+%d/%m/%Y %H:%M:%S') — $*"; }
notify() { sudo /usr/local/bin/pizeta-notify.sh "🧪 srv-lm bench Flash-Next v3 — $*" >/dev/null 2>&1 || echo "(notifica Telegram non riuscita)"; }
gpu_cooldown() { local T=99; for i in $(seq 1 30); do T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "$T" -le 50 ] && break; sleep 10; done; echo "GPU a ${T} C"; }
tok1() { # tok/s @1 mediato sui 3 run a 8k
  python3 -c "import json,glob,sys; v=[json.load(open(f)).get('output_throughput',0) for f in glob.glob('$B/results/$1_in8192_out512_c1_r*.json')]; print('%.1f tok/s @1 (8k, %d run)' % (sum(v)/len(v), len(v)) if v else 'n/d')" 2>/dev/null
}
restart_27b() {
  say "RIAVVIO 27B (script canonico)"
  docker rm -f flash-next-vllm flash-next-tabby 2>/dev/null || true
  bash /home/pizeta/vllm/run-sglang-38.sh
  for i in $(seq 1 60); do
    curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "27B READY dopo ~$((i*10))s"; notify "FINE v3. 27B riacceso e pronto ($(date '+%H:%M')). Log: $(basename $LOG)"; return 0; }
    sleep 10
  done
  notify "⚠️ FINE v3 ma il 27B NON risponde dopo 10 min: controllare sglang-workhorse"
}
trap restart_27b EXIT

say "INIZIO run-matrix-v3 — log: $LOG"
if [ ! -d $B/results-v1-seed42 ]; then
  mkdir -p $B/results-v1-seed42 && mv $B/results/*.json $B/results/mem_*.csv $B/results-v1-seed42/ 2>/dev/null
  echo "risultati v1 archiviati in results-v1-seed42/ ($(ls $B/results-v1-seed42 | wc -l) file)"
fi
notify "PARTITO alle $(date '+%H:%M'): rifaccio matrici+prefill con seed unico (4 config, ~2,5 h). 27B fermo fino alla fine."
say "STOP 27B"
docker stop sglang-workhorse 2>/dev/null; docker rm -f flash-next-vllm flash-next-tabby 2>/dev/null; sleep 5

run_engine() { # ENGINE PORT MTP
  local E=$1 P=$2 M=$3 TAG="${1}_mtp-${3}"
  say "===== $TAG ====="
  bash $B/configs/${E}-flash-next.sh "$M"
  if ! bash $B/scripts/wait-ready.sh "$P" 1800 flash-next-$E; then
    echo "!!! $TAG NON PRONTO — ultime righe del container:"; docker logs --tail 150 flash-next-$E 2>&1 | sed 's/^/    /'
    docker rm -f flash-next-$E 2>/dev/null; echo "!!! $TAG SCARTATO"; notify "❌ $TAG non è partito: scartato."; return 1
  fi
  nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader | sed 's/^/VRAM dopo il caricamento: /'
  free -h | awk '/Mem/{print "RAM: used " $3 " avail " $7}'
  notify "▶ $TAG caricato, parte la matrice."
  bash $B/scripts/mem-monitor.sh flash-next-$E $B/results/mem_${TAG}.csv &
  local MON=$!
  bash $B/scripts/bench-matrix-v2.sh "$E" "$P" "$M"
  bash $B/scripts/bench-prefill-v2.sh "$E" "$P" "$M"
  kill $MON 2>/dev/null
  echo "--- errori server durante la serie (ultimi 20) ---"
  docker logs flash-next-$E 2>&1 | grep -aiE 'error|exception|traceback' | grep -av 'server_info' | tail -20 | cut -c1-240
  docker rm -f flash-next-$E 2>/dev/null
  echo "===== $TAG completato ====="; notify "✅ $TAG completato: $(tok1 $TAG)"
  gpu_cooldown
}
run_engine vllm  8010 on
run_engine vllm  8010 off
run_engine tabby 8011 on
run_engine tabby 8011 off


say "TOOL-EVAL vLLM MTP off (thinking ON) — confronto xgrammar con/senza speculativo (gli errori compaiono solo col thinking)"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p $B/runs && cd $B/runs
bash $B/configs/vllm-flash-next.sh off
if bash $B/scripts/wait-ready.sh 8010 1800 flash-next-vllm; then
  notify "▶ tool-eval vLLM MTP off (thinking ON, ~60 min) per il confronto xgrammar."
  tool-eval-bench run --base-url "http://127.0.0.1:8010/v1" --model flash-next --seed 42 --trials 8 --hardmode --temperature 0.6 --timeout 600 2>&1 | tail -25
  echo "--- errori xgrammar SENZA MTP: $(docker logs flash-next-vllm 2>&1 | grep -ac 'Failed to advance FSM') (richieste $(docker logs flash-next-vllm 2>&1 | grep -ac 'POST /v1/chat/completions')) ---"
  notify "✅ tool-eval vLLM MTP off completato: errori xgrammar $(docker logs flash-next-vllm 2>&1 | grep -ac 'Failed to advance FSM')."
else
  echo "!!! vLLM MTP off non pronto per il tool-eval — saltato"
fi
docker rm -f flash-next-vllm 2>/dev/null; cd /

say "RIEPILOGO risultati v3"
for f in $B/results/*_c1_r*.json; do [ -f "$f" ] || continue
  python3 -c "import json; d=json.load(open('$f')); print('  %-48s %7.1f tok/s  TTFT %6.0f ms  ok %s' % ('$(basename $f .json)', d.get('output_throughput',0), d.get('median_ttft_ms',0), d.get('completed','?')))" 2>/dev/null
done
say "FINE run-matrix-v3"
