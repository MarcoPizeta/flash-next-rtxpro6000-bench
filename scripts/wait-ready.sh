#!/usr/bin/env bash
# Attende che l'endpoint OpenAI risponda e fa una gen di prova. Uso: wait-ready.sh PORT [max_s] [container]
# Se il container e' indicato e muore, esce subito con 1 (non aspetta il timeout).
PORT=$1; MAX=${2:-1800}; CT=${3:-}; T0=$(date +%s)
BODY='{"model":"flash-next","messages":[{"role":"user","content":"Rispondi solo PONG"}],"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}'
while :; do
  if curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    R=$(curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY" | grep -o '"content":"[^"]*"' | head -1)
    echo "READY dopo $(( $(date +%s) - T0 ))s — gen di prova: ${R:-<vuota>}"; exit 0
  fi
  if [ -n "$CT" ] && [ "$(docker inspect -f '{{.State.Running}}' "$CT" 2>/dev/null)" != "true" ]; then
    echo "CONTAINER $CT MORTO dopo $(( $(date +%s) - T0 ))s (stato: $(docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}}' "$CT" 2>/dev/null || echo assente))"; exit 1
  fi
  [ $(( $(date +%s) - T0 )) -ge "$MAX" ] && { echo "TIMEOUT dopo ${MAX}s"; exit 1; }
  sleep 10
done
