#!/usr/bin/env bash
# Orchestratore benchmark Flash-Next su srv-lm — 04/09/2026.
# Esegue TUTTA la sequenza senza supervisione, con log, punti di controllo e notifiche
# Telegram a Marco (bot di Brown, stesso canale degli allarmi NUT/templog).
# Se un motore non parte entro il timeout, lo scarta e passa oltre: la notte non si ferma.
# Il 27B viene RIACCESO in ogni caso alla fine (trap su EXIT).
set -uo pipefail
B=/opt/flash-next/bench
TS=$(date +%Y%m%d-%H%M)
LOG=$B/logs/run-all-$TS.log
exec > >(tee -a "$LOG") 2>&1
export PATH="$HOME/.local/bin:$PATH"

say()    { echo; echo "##### $(date '+%d/%m/%Y %H:%M:%S') — $*"; }
notify() { sudo /usr/local/bin/pizeta-notify.sh "🧪 srv-lm bench Flash-Next — $*" >/dev/null 2>&1 || echo "(notifica Telegram non riuscita)"; }
gpu_cooldown() { # aspetta che la GPU torni sotto 50 C (max 5 min) prima della serie successiva
  local T=99
  for i in $(seq 1 30); do T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "$T" -le 50 ] && break; sleep 10; done
  echo "GPU a ${T} C"
}
tok1() { # tok/s @1 mediato sui 3 run a 8k, se ci sono
  python3 - "$B/results" "$1" << 'PY' 2>/dev/null
import json,sys,glob
v=[json.load(open(f)).get('output_throughput',0) for f in glob.glob(f"{sys.argv[1]}/{sys.argv[2]}_in8192_out512_c1_r*.json")]
print("%.1f tok/s @1 (8k, %d run)" % (sum(v)/len(v), len(v)) if v else "n/d")
PY
}
restart_27b() {
  say "RIAVVIO 27B (script canonico, non docker start)"
  docker rm -f flash-next-vllm flash-next-tabby 2>/dev/null || true
  bash /home/pizeta/vllm/run-sglang-38.sh
  for i in $(seq 1 60); do
    curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "27B READY dopo ~$((i*10))s"; notify "FINE. 27B riacceso e pronto ($(date '+%H:%M')). Log: $(basename $LOG)"; return 0; }
    sleep 10
  done
  echo "ATTENZIONE: 27B non risponde dopo 10 min — controllare docker logs sglang-workhorse"
  notify "⚠️ FINE ma il 27B NON risponde dopo 10 min: controllare sglang-workhorse"
}
trap restart_27b EXIT

say "INIZIO run-all — log: $LOG"
notify "PARTITO alle $(date '+%H:%M'). Sequenza: vLLM (MTP on/off) → TabbyAPI/EXL3 (MTP on/off) → tool-eval-bench. Il 27B resta fermo fino alla fine (~6-8 h)."
say "snapshot hardware/software"
bash $B/scripts/snapshot-hw.sh >/dev/null
echo "salvato in $B/results/HARDWARE.md"

say "STOP 27B (docker stop: al termine riparte con run-sglang-38.sh)"
docker stop sglang-workhorse && echo "27B fermato"
sleep 10
nvidia-smi --query-gpu=memory.used --format=csv,noheader

run_engine() { # ENGINE PORT MTP
  local E=$1 P=$2 M=$3 TAG="${1}_mtp-${3}"
  say "===== $TAG ====="
  bash $B/configs/${E}-flash-next.sh "$M"
  if ! bash $B/scripts/wait-ready.sh "$P" 1800 flash-next-$E; then
    echo "!!! $TAG NON PRONTO entro 30 min — ultime righe del container:"
    docker logs --tail 150 flash-next-$E 2>&1 | sed 's/^/    /'
    docker rm -f flash-next-$E 2>/dev/null; echo "!!! $TAG SCARTATO"
    notify "❌ $TAG non è partito entro 30 min: scartato, passo oltre."
    return 1
  fi
  nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader | sed 's/^/VRAM dopo il caricamento: /'
  free -h | awk '/Mem/{print "RAM: used " $3 " avail " $7}'
  notify "▶ $TAG caricato, parte la matrice."
  bash $B/scripts/mem-monitor.sh flash-next-$E $B/results/mem_${TAG}.csv &
  local MON=$!
  bash $B/scripts/bench-matrix.sh "$E" "$P" "$M"
  bash $B/scripts/bench-prefill.sh "$E" "$P" "$M"
  kill $MON 2>/dev/null
  docker rm -f flash-next-$E 2>/dev/null
  echo "===== $TAG completato ====="
  notify "✅ $TAG completato: $(tok1 $TAG)"
  gpu_cooldown
}

# Via A (validata) prima, poi Via B. MTP on prima perche' e' la config d'uso; off per l'A/B.
run_engine vllm  8010 on
run_engine vllm  8010 off
run_engine tabby 8011 on
run_engine tabby 8011 off

say "TOOL-EVAL-BENCH (seed 42, 8 trial, hard mode) — qualita' confrontabile con MiaAI"
notify "▶ matrici finite, parte il tool-eval-bench (8 trial × 2 motori × thinking on/off)."
mkdir -p $B/runs && cd $B/runs
for E in vllm tabby; do
  P=8010; [ "$E" = tabby ] && P=8011
  say "tool-eval-bench su $E (MTP on)"
  bash $B/configs/${E}-flash-next.sh on
  if bash $B/scripts/wait-ready.sh "$P" 1800 flash-next-$E; then
    echo "--- thinking ON ---"
    tool-eval-bench run --base-url "http://127.0.0.1:$P/v1" --model flash-next --seed 42 --trials 8 --hardmode --temperature 0.6 --timeout 600 2>&1 | tail -25
    echo "--- thinking OFF ---"
    tool-eval-bench run --base-url "http://127.0.0.1:$P/v1" --model flash-next --seed 42 --trials 8 --hardmode --temperature 0.6 --timeout 600 --no-think 2>&1 | tail -25
    notify "✅ tool-eval-bench su $E completato."
  else
    echo "!!! $E non pronto per tool-eval — saltato"
    notify "❌ $E non pronto per il tool-eval-bench: saltato."
  fi
  docker rm -f flash-next-$E 2>/dev/null
  gpu_cooldown
done
cd /

say "RIEPILOGO risultati"
ls -1 $B/results/*.json 2>/dev/null | wc -l | sed 's/^/  json prodotti: /'
for f in $B/results/*_c1_r*.json; do
  [ -f "$f" ] || continue
  python3 -c "import json; d=json.load(open('$f')); print('  %-48s %7.1f tok/s  TTFT %6.0f ms' % ('$(basename $f .json)', d.get('output_throughput',0), d.get('median_ttft_ms',0)))" 2>/dev/null
done
say "FINE run-all"
