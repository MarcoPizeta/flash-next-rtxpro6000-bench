#!/usr/bin/env bash
# Via B — TabbyAPI cu13 (ExLlamaV3 1.4.6) + turboderp exl3 4.05bpw, tabella n-gram in RAM.
# Uso: tabby-flash-next.sh on|off   (MTP)
set -euo pipefail
MTP="${1:?uso: tabby-flash-next.sh on|off}"
MODE=disabled; [ "$MTP" = on ] && MODE=mtp
CFG=/opt/flash-next/bench/configs/tabby-config.generated.yml
sed "s/__DRAFT_MODE__/$MODE/" /opt/flash-next/bench/configs/tabby-config.yml > "$CFG"
docker rm -f flash-next-tabby 2>/dev/null || true
docker run -d --name flash-next-tabby --gpus all --shm-size 8g --ulimit memlock=-1 -p 8011:5000 \
  -v /opt/flash-next/exl3-4.05bpw:/app/models/exl3-4.05bpw:ro \
  -v "$CFG":/app/config.yml:ro \
  tabbyapi:cu13-pizeta
echo "flash-next-tabby avviato (MTP=$MTP, draft_mode=$MODE) su :8011 — segui con: docker logs -f flash-next-tabby"
