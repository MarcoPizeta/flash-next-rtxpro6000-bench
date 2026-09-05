#!/usr/bin/env bash
# Campiona VRAM, RSS del motore e RAM disponibile ogni 5 s. Uso: mem-monitor.sh CONTAINER OUT.csv
C=$1; OUT=$2
echo "ts,vram_used_mib,gpu_temp_c,gpu_power_w,engine_rss_mib,mem_available_mib,swap_used_mib" > "$OUT"
while docker ps -q -f name="$C" | grep -q .; do
  G=$(nvidia-smi --query-gpu=memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits | tr -d ' ')
  PID=$(docker inspect -f '{{.State.Pid}}' "$C" 2>/dev/null)
  RSS=$(ps -o rss= --ppid "$PID" -p "$PID" 2>/dev/null | awk '{s+=$1} END{printf "%d", s/1024}')
  M=$(awk '/MemAvailable/{a=$2} /SwapTotal/{t=$2} /SwapFree/{f=$2} END{printf "%d,%d", a/1024, (t-f)/1024}' /proc/meminfo)
  echo "$(date +%s),$G,$RSS,$M" >> "$OUT"
  sleep 5
done
