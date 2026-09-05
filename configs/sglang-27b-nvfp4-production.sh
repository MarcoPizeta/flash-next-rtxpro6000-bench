#!/usr/bin/env bash
# PRODUZIONE dal 25/08/2026: RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead (decisione Marco 25/08).
#
# PERCHE questo checkpoint (trittico misurato 24-25/08, dettagli Obsidian):
#  - NVFP4 su MLP, FP8 su attention/DeltaNet, ma lm_head/vision/MTP/embeddings in BF16
#    -> protegge la scelta del token finale (cifre DDT, verdetti deroghe).
#  - kv_cache_scheme CALIBRATO nel ckpt -> KV FP8 funziona: pool 1.393.071 token
#    (~5 sessioni full-256K) vs 344k del BF16.
#  - Misure: decode 97,7 tok/s (BF16=60,5; RadixArk base=113,2 ma lm_head a 4 bit),
#    check38 ok, visione 3/3, deroghe 8/8.
#
# VINCOLI SU SM120 (RTX PRO 6000) - non cambiare a caso:
#  - --fp4-gemm-backend flashinfer_cudnn OBBLIGATORIO: il default cutlass ha una
#    race condition con corruzione silenziosa della memoria.
#  - --kv-cache-dtype fp8_e4m3 funziona SOLO con ckpt che hanno le scale KV (questo).
#    Col ckpt BF16 = output corrotto. MAI fp8 senza scale.
#  - MTP/NEXTN richiede SGLANG_ENABLE_SPEC_V2=True (senza: seconda copia modello -> OOM).
#  - --attention-backend trtllm_mha = crash-loop silenzioso su questa build. NON usarlo.
#  - spec steps: RI-TARATI sul checkpoint NVFP4 il 27/08 -> ottimo = **2** (98,4 tok/s),
#    contro 97,4 con 3 (default) e 90,5 con 4. Sul BF16 l'ottimo era 3: la taratura
#    dipende dal checkpoint, rifarla a ogni cambio modello.
#  - DFLASH block-16 (draft z-lab/Qwen3.8-27B-DFlash2): MISURATO 27/08 = **107,4 tok/s
#    (+9%)**, check38 6/6, visione OK, MA richiede --mem-fraction-static 0.87 (a 0.94 va
#    in OOM: il draft pesa 3,6GB) e il pool KV scende a 988.598 token (-29%); acceptance
#    rate bassa (0,08-0,13 = disallineamento del draft sull'italiano). NON adottato in
#    attesa di un soak test: config pronta in ~/vllm/dflash-retry.sh.
#  - cuda-graph-max-bs 8 + max-running-requests 8: senza, lo spec forza 48 e i grafi
#    mangiano ~4GB di pool.
#
# THINKING = MEDIUM di default (adottato 25/08, ok Marco, in linea con eval Kaitchup):
#   reasoning_effort e' ignorato da SGLang (kwarg scartato) -> il livello si decide nel
#   CHAT TEMPLATE. Qui usiamo ~/vllm/chat-template-38-medium.jinja = originale del ckpt
#   con UNA riga cambiata: default('xhigh') -> default('medium') (medium = nessuna
#   istruzione iniettata). xhigh/low al bisogno = iniettare la frase ufficiale come
#   system prompt (frasi nel template originale chat-template-38-orig.jinja).
#   ⚠️ Con --chat-template esterno i parser vanno ESPLICITI (auto non risolve): qwen3 +
#   qwen3_coder. ⚠️ A ogni update del checkpoint: rigenerare la patch con
#   make-medium-template.sh e ricopiare in /opt/hf-cache/templates/.
#   Rollback a xhigh default: togliere la riga --chat-template e rimettere parser auto.
#   enable_thinking per-richiesta funziona. OFF solo per estrazione, mai per decisioni.
#   Misure medium vs xhigh (compare-effort.py 25/08): deroghe 2/2, reasoning -33% su 8D.
#   ⚡ SCOPERTA 25/08: col template ESTERNO SGLang PASSA i chat_template_kwargs al
#   template (col template del ckpt li SCARTAVA) -> `reasoning_effort` per-richiesta
#   ORA FUNZIONA davvero (xhigh/medium/low verificati). Il workaround "frase iniettata
#   nel system prompt" e' OBSOLETO. ⚠️ Corollario: un valore NON valido (es. "off")
#   faceva raise nel template -> HTTP 400 al client; la patch 2 lo neutralizza
#   (fallback a medium). Per spegnere il thinking: enable_thinking:false.
#
# ✅ 29/08 - TEMPLATE MEDIUM RIATTIVATO (richiesta di Marco: la prod deve stare a medium).
#   STORICO dell'incidente: il 27/08 16:43
#   un'altra sessione ha riscritto questo script SENZA --chat-template e ha cancellato
#   /opt/hf-cache/templates/ + tutti gli script in ~/vllm/. Rimettere la riga puntando a
#   un file inesistente = CRASH-LOOP (successo il 29/08 10:14).
#   Conseguenza: il thinking gira di nuovo a XHIGH di default (non medium come deciso il
#   25/08). Per tornare a medium: rigenerare i template (extract-template.sh +
#   make-medium-template.sh, copie nella cartella scratchpad della sessione Claude) e
#   solo DOPO rimettere --chat-template + parser espliciti qwen3/qwen3_coder.
#   ⛔ PRIMA DI LANCIARE verificare SEMPRE: ls -l /opt/hf-cache/templates/
#   Se i file mancano: bash extract-template.sh && bash make-medium-template.sh
#   NON cancellare i template: la prod resta a xhigh in silenzio (o va in crash-loop).
#   Runbook completo: Obsidian 'Workstation LM Locale' § RUNBOOK Thinking medium.
#
# ⛔ INCIDENTE 29/08 - mem-fraction 0.94 = 500 CUDA OOM IN ESERCIZIO. Con 0.94 il pool
#   si prende 93,95 GiB e restano ~1 GB: le allocazioni transitorie (path multimodale,
#   prompt lunghi, pool privati dei CUDA graph) non entrano -> "Tried to allocate 84.00 MiB"
#   -> HTTP 500 al gateway -> il worker NC accumulava NC in errore. 7x 500 il 29/08 tra
#   06:07 e 06:12, server su da 21h (NON un problema di boot ne' del testo 8D).
#   FIX: mem-fraction 0.94 -> **0.90** (headroom ~4-5 GB, pool KV ancora ~1M token = 4x
#   il BF16) + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (anti-frammentazione,
#   suggerito dall'errore stesso). ⚠️ Non rialzare 0.90 per guadagnare pool: il pool
#   abbondava gia', il margine no.
#
# METRICHE: --enable-metrics OBBLIGATORIO (SGLang non espone /metrics di default,
#   a differenza di vLLM). Prometheus ha il job `sglang` -> <srv-lm>:8000/metrics
#   (~/monitoring/prometheus/prometheus.yml): senza il flag il target e' DOWN e i
#   segnali LLM del cockpit vanno a n/d.
#
# ROLLBACK - ⚠️ 31/08/2026: i file di backup citati fin qui NON ESISTONO PIU'
#   (run-sglang-38.sh.bf16-tuned-bak e rollback-36.sh sono spariti con la morte del
#   P310 del 27/08; il riferimento era una trappola). Il rollback ora si fa da GIT:
#     git -C ~/pizeta-srv-lm log --oneline -- vllm/run-sglang-38.sh
#     git -C ~/pizeta-srv-lm checkout <commit> -- vllm/run-sglang-38.sh && bash run-sglang-38.sh
#   Marco 31/08: il rollback al BF16 non serve, il checkpoint attuale funziona bene.
#   NB: il ckpt Qwen/Qwen3.8-27B BF16 non e' piu' in cache (hf-cache ricostruita).
# ⚠️ Anche dflash-retry.sh (citato sopra per il draft DFLASH) e' ASSENTE: la config
#   e' descritta nei commenti, va riscritta a mano se si vuole riprovare.
set -euo pipefail
docker rm -f sglang-workhorse 2>/dev/null || true
docker run -d --name sglang-workhorse \
  --restart unless-stopped \
  -e SGLANG_ENABLE_SPEC_V2=True \
  -e SGLANG_SANITIZE_NAN_LOGITS=True \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --gpus all \
  --ipc=host \
  --shm-size 32g \
  -p 8000:8000 \
  -v /opt/hf-cache:/root/.cache/huggingface \
  lmsysorg/sglang:dev-cu13 \
  python3 -m sglang.launch_server \
  --model-path RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 8000 \
  --context-length 262144 \
  --mem-fraction-static 0.90 \
  --kv-cache-dtype fp8_e4m3 \
  --fp4-gemm-backend flashinfer_cudnn \
  --chat-template /root/.cache/huggingface/templates/chat-template-38-medium.jinja \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-metrics \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --speculative-attention-mode decode \
  --cuda-graph-max-bs 8 \
  --max-running-requests 8 \
  --mamba-ssm-dtype bfloat16 \
  --strip-thinking-cache \
  --sleep-on-idle
