#!/usr/bin/env bash
# EXL3 5.05bpw_h6_ng6 (05/09/2026): quanto costa il bit in meno? Dopo 'FINE run-extra' e download completo:
# TabbyAPI MTP on con cache 131072 (i pesi a 5 bpw sono ~80 GiB: 96 GB stretti) → tool-eval al miglior effort
# di Flash-Next (label) → matrice v2 (c=1/4, 1k/8k/32k, tag tabby5_mtp-on) → 27B riacceso.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
B=/opt/flash-next/bench
LOG=$B/logs/run-exl3-5bpw-$(date +%Y%m%d-%H%M).log
exec > >(tee -a "$LOG") 2>&1
say()    { echo; echo "##### $(date '+%d/%m/%Y %H:%M:%S') — $*"; }
notify() { sudo /usr/local/bin/pizeta-notify.sh "🧪 srv-lm EXL3 5bpw — $*" >/dev/null 2>&1 || true; }
score()  { grep -oE '\*\*[0-9.]+ ± [0-9.]+\*\*' "$1" | head -1 | tr -d '*'; }
lines()  { grep -E '\*\*Final Score\*\*|\*\*Pass@8\*\*|\*\*Pass\^8\*\*' "$1" | tr -s ' ' | cut -c1-200; }
D=/opt/flash-next/exl3-5.05bpw

say "INIZIO run-exl3-5bpw — attendo FINE run-extra e download completo"
while ! grep -aq 'FINE run-extra' $(ls -t $B/logs/run-extra-*.log | head -1) 2>/dev/null; do sleep 60; done
while ! grep -aq 'DOWNLOAD COMPLETO' /opt/flash-next/logs/download-exl3-5.05.log 2>/dev/null; do
  grep -aq 'FALLITO dopo' /opt/flash-next/logs/download-exl3-5.05.log 2>/dev/null && { notify "❌ download 5.05bpw fallito: salto"; exit 1; }
  sleep 60
done
say "download completo: $(du -sh $D | cut -f1), $(ls $D/*.safetensors | wc -l) safetensors"
BEST=$(grep -aoE 'miglior effort: [a-z]+' $(ls -t $B/logs/run-effort-*.log | head -1) | tail -1 | awk '{print $3}'); BEST=${BEST:-xhigh}
notify "parte EXL3 5.05bpw (effort $BEST): tool-eval + matrice, ~1,5 h, 27B fermo."

say "config Tabby 5.05bpw (cache 131072) e stop 27B"
sed 's/exl3-4.05bpw/exl3-5.05bpw/; s/max_seq_len: 262144/max_seq_len: 131072/; s/cache_size: 262144/cache_size: 131072/; s/__DRAFT_MODE__/mtp/' $B/configs/tabby-config.yml > $B/configs/tabby-config-5bpw.generated.yml
grep -nE 'model_name|cache_size|max_seq_len|draft_mode' $B/configs/tabby-config-5bpw.generated.yml
docker stop sglang-workhorse >/dev/null; docker rm -f flash-next-tabby flash-next-vllm >/dev/null 2>&1
docker run -d --name flash-next-tabby --gpus all --shm-size 8g --ulimit memlock=-1 -p 8011:5000 \
  -v $D:/app/models/exl3-5.05bpw:ro -v $B/configs/tabby-config-5bpw.generated.yml:/app/config.yml:ro tabbyapi:cu13-pizeta >/dev/null
if ! bash $B/scripts/wait-ready.sh 8011 1200 flash-next-tabby; then
  docker logs --tail 60 flash-next-tabby | cut -c1-200; notify "❌ Tabby 5.05bpw non parte (VRAM?): riaccendo il 27B"
  docker rm -f flash-next-tabby >/dev/null 2>&1; bash /home/pizeta/vllm/run-sglang-38.sh >/dev/null 2>&1; exit 1
fi
nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader | sed 's/^/VRAM dopo il caricamento: /'
free -h | awk '/Mem/{print "RAM: used " $3 " avail " $7}'

say "TOOL-EVAL EXL3 5.05bpw MTP on, reasoning_effort=$BEST, 8 trial"
cd $B/runs && tool-eval-bench run --base-url http://127.0.0.1:8011/v1 --model flash-next --seed 42 --trials 8 --hardmode --temperature 0.6 --timeout 600 \
  --backend-kwargs "{\"chat_template_kwargs\":{\"reasoning_effort\":\"$BEST\"}}" \
  --label "TabbyAPI / ExLlamaV3 1.4.6 EXL3 5.05bpw_h6_ng6, MTP on, ngram_ram, cache 131072, reasoning_effort=$BEST, thinking on" 2>&1 | tail -20; cd /
S=$(ls -t $B/runs/runs/2026/09/*summary.md | head -1); echo "summary 5.05bpw: $S"; lines "$S"
notify "✅ EXL3 5.05bpw effort=$BEST: $(score "$S") (4.05bpw xhigh: 86,4)"

say "MATRICE v2 su 5.05bpw (tag tabby5_mtp-on)"
bash $B/scripts/mem-monitor.sh flash-next-tabby $B/results/mem_tabby5_mtp-on.csv & MON=$!
sed 's/TAG="${ENGINE}_mtp-${MTP}"/TAG="tabby5_mtp-${MTP}"/' $B/scripts/bench-matrix-v2.sh > /tmp/bench-matrix-v2-5bpw.sh
bash /tmp/bench-matrix-v2-5bpw.sh tabby 8011 on
sed 's/TAG="${ENGINE}_mtp-${MTP}"/TAG="tabby5_mtp-${MTP}"/' $B/scripts/bench-prefill-v2.sh > /tmp/bench-prefill-v2-5bpw.sh
bash /tmp/bench-prefill-v2-5bpw.sh tabby 8011 on
kill $MON 2>/dev/null
echo "--- errori server ---"; docker logs flash-next-tabby 2>&1 | grep -aiE 'error|exception' | grep -av server_info | tail -6 | cut -c1-200
docker rm -f flash-next-tabby >/dev/null 2>&1

say "RIAVVIO 27B"
bash /home/pizeta/vllm/run-sglang-38.sh >/dev/null 2>&1
for i in $(seq 1 60); do curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "27B READY dopo ~$((i*10)) s"; break; }; sleep 10; done
say "FINE run-exl3-5bpw"
notify "FINE EXL3 5bpw. 27B $(curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null && echo riacceso || echo NON RISPONDE)."
