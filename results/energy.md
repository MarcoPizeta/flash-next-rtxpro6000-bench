# Energia GPU (power-log 5 s) — 05/09/2026

## Per fase (tra marker consecutivi dei log)

| inizio | fase | durata min | W medi | Wh | campioni |
|---|---|---|---|---|---|
| 08:18 | attendo FINE run-effort | 8 | 407 | 33.4 | 59 |
| 08:26 | INIZIO run-extra — attendo FINE tool-eval 27B thinking OFF | 27 | 405 | 180.6 | 321 |
| 08:53 | TOOL-EVAL Flash-Next vLLM MTP on — reasoning_effort=low (thinking ON) | 38 | 414 | 261.9 | 455 |
| 09:31 | punteggi Flash-Next su vLLM: xhigh=85.6 medium=86.5 low=87.9 → miglior | 41 | 439 | 301.8 | 495 |
| 10:13 | test vision / tool-call / idle su vLLM (riaccende il 27B alla fine) | 10 | 96 | 16.0 | 120 |
| 10:23 | FINE run-effort | 0 | 84 | 0.1 | 1 |
| 10:23 | TOOL-EVAL 27B thinking OFF | 32 | 485 | 255.8 | 380 |
| 10:55 | FINE tool-eval 27B thinking OFF | 0 | 75 | 0.6 | 6 |
| 10:55 | CONCORRENZA 27b-nvfp4: 32k input, out 256, c=8 (16 richieste) | 1 | 370 | 5.7 | 11 |
| 10:56 | CONCORRENZA 27b-nvfp4: 32k input, out 256, c=16 (32 richieste) | 1 | 515 | 12.2 | 17 |
| 10:58 | CONTEXT-PRESSURE SWEEP 27b-nvfp4: 25/50/75/100% di 65.536 tok, 2 trial | 51 | 528 | 441.9 | 603 |
| 11:48 | STOP 27B → vLLM Flash-Next MTP on (alias) | 6 | 117 | 11.4 | 70 |
| 11:54 | PREFIX-CACHE flash-next-vllm (system ~4,7k tok identico, 8 domande div | 0 | 132 | 0.4 | 2 |
| 11:54 | CONCORRENZA flash-next-vllm: 32k input, out 256, c=8 (16 richieste) | 1 | 254 | 4.9 | 14 |
| 11:56 | CONCORRENZA flash-next-vllm: 32k input, out 256, c=16 (32 richieste) | 1 | 337 | 7.5 | 16 |
| 11:57 | CONTEXT-PRESSURE SWEEP flash-next-vllm: 25/50/75/100% di 65.536 tok, 2 | 31 | 483 | 251.0 | 374 |
| 12:28 | TabbyAPI/EXL3 MTP on | 1 | 153 | 3.8 | 18 |
| 12:30 | PREFIX-CACHE flash-next-exl3 (system ~4,7k tok identico, 8 domande div | 0 | 429 | 1.2 | 2 |
| 12:30 | CONCORRENZA flash-next-exl3: 32k input, out 256, c=8 (16 richieste) | 1 | 377 | 8.4 | 16 |
| 12:31 | CONCORRENZA flash-next-exl3: 32k input, out 256, c=16 (32 richieste) | 2 | 438 | 12.2 | 20 |
| 12:33 | 27B BF16 (Qwen/Qwen3.8-27B): download completo? | 2 | 191 | 7.2 | 27 |
| 12:35 | TOOL-EVAL 27B BF16, thinking medium, 8 trial | 104 | 534 | 919.7 | 1241 |
| 14:19 | RIAVVIO 27B NVFP4 (script canonico) | 2 | 148 | 4.5 | 22 |

## Matrice v3 — Wh per 1000 token di output (mem_*.csv 10 s, intero run della config incl. caricamento)

| config | durata min | W medi | Wh totali | token output | Wh / 1k tok out | Wh / 1k tok in+out |
|---|---|---|---|---|---|---|
| vllm_mtp-on | 43 | 188 | 134 | 62605 | 2.14 | 0.050 |
| vllm_mtp-off | 46 | 190 | 144 | 62605 | 2.30 | 0.054 |
| tabby_mtp-on | 54 | 260 | 235 | 47747 | 4.92 | 0.170 |
| tabby_mtp-off | 56 | 264 | 246 | 48927 | 5.03 | 0.173 |

Note: le fasi comprendono caricamento, warm-up e attese; i W medi dei tool-eval includono i tempi morti tra richieste. Riferimento: 27B a riposo con `--sleep-on-idle` = 12 W GPU.
