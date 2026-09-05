# Qwen3.8-Flash-Next on a single RTX PRO 6000 Blackwell (96 GB) with 64 GB host RAM: vLLM (NVFP4/FP8 + INT4 PLE offload) vs ExLlamaV3 (EXL3 4.05 bpw)

Independent, reproducible benchmark of the two ways to serve **Qwen3.8-Flash-Next** (125B-A6B + 51B n-gram/PLE table) on one workstation GPU with only **64 GB of host RAM**, run on the night of 04–05 September 2026. Everything needed to reproduce is in this repo: raw client JSON, tool-eval reports, engine configs, scripts, hardware snapshot, and the full orchestrator logs.

**TL;DR** — On this hardware, **vLLM + `primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8` + INT4 PLE offload + MTP** is the production choice: faster from 8k context up and at any concurrency > 1, **2.7× faster prefill**, stable at 32k×4, ~10–12 GB host RAM. **ExLlamaV3/TabbyAPI + `turboderp/Qwen3.8-Flash-Next-exl3` 4.05bpw** wins only short single-user chat (169 tok/s at 1k input), loads in ~1–1.5 min instead of 5–6, but needs ~45 GB host RAM (`ngram_ram: true`) and fails about two thirds of the requests at 32k input × 4 concurrent (21/60 completed). **Tool-calling quality is indistinguishable** between the two (tool-eval-bench 85.5–87.0, all within 1σ, matching MiaAI's published numbers).

---

## Hardware & software

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB (97,887 MiB), 600 W, driver 595.84, PCIe 5.0 x16 |
| CPU | AMD Ryzen Threadripper 9960X (24C/48T), **max clock capped to 4.5 GHz for the whole `v3` run** (see *Caveats*) |
| RAM | 64 GB DDR5-4800 ECC RDIMM (2 × 32 GB, dual channel) — 60.9 GiB usable |
| Storage | Samsung 990 PRO 1 TB NVMe (models + HF cache) |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-30, Docker 29.7.2 |
| vLLM | `vllm/vllm-openai:qwen38-flash-next` @ `sha256:fc120ece…` (built 26/08/2026), vLLM `0.1.dev20073+g8e685d198`, torch 2.13.0+cu130 |
| ExLlamaV3 | `ghcr.io/theroyallab/tabbyapi:cu13` @ `sha256:90a01932…` (built 03/09/2026), ExLlamaV3 **1.4.6**, torch 2.11 cu130 — rebuilt as `tabbyapi:cu13-pizeta` with `python3-dev` added (see *Fixes*) |
| Client | `python3 -m sglang.benchmark.serving --backend vllm-chat` from `lmsysorg/sglang:dev-cu13` (same client, same `/v1/chat/completions` endpoint for both engines) |
| Quality | `tool-eval-bench 2.6.1.dev42+g6a98f0324` (`run --seed 42 --trials 8 --hardmode --temperature 0.6 --timeout 600`) |
| Checkpoints | `primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8` rev `adf14c6b` (172 GiB on disk incl. INT4 PLE table) · `turboderp/Qwen3.8-Flash-Next-exl3` branch `4.05bpw_h6_ng6` (101 GiB) |

Full snapshot: [`HARDWARE.md`](HARDWARE.md). Background load during the runs: Prometheus/Grafana/cAdvisor/node-exporter, two idle OCR containers, Open WebUI (idle).

## Engine configurations

**vLLM** ([`configs/vllm-flash-next.sh`](configs/vllm-flash-next.sh)): primitive-ai's 3-file PLE-quant overlay (`worker.py`, `ple_layer.py`, `connector.py`) mounted over `/usr/local/lib/python3.12/dist-packages/vllm`, `VLLM_PLE_QUANT_DIR=/ples_int4`, `VLLM_PLE_CPU_OFFLOAD=1`, `VLLM_GDN_DECODE_KERNEL=triton`, `--distributed-executor-backend mp --gpu-memory-utilization 0.92 --max-num-seqs 64 --max-model-len 135168 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`. MTP: `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`. Resulting KV cache: **503k tokens without MTP, 217k with MTP** (max concurrency 1.6× at 135k).

**ExLlamaV3 / TabbyAPI** ([`configs/tabby-flash-next.sh`](configs/tabby-flash-next.sh), [`configs/tabby-config.yml`](configs/tabby-config.yml)): `backend: exllamav3`, `max_seq_len: 262144`, `cache_size: 262144`, `cache_mode: FP16`, `chunk_size: 4096`, `gpu_split_auto: true`, `vision: true`, **`ngram_ram: true`** (the 36 GiB 6-bit n-gram table lives in host RAM, never in VRAM), `tool_format: qwen3_coder`, `reasoning: true`. MTP: `draft_model: draft_mode: mtp` vs `disabled`.

Both: temperature 0.6, top_p 0.95, `enable_thinking: false` for the speed matrix.

## Method

* **Speed matrix** ([`scripts/bench-matrix-v2.sh`](scripts/bench-matrix-v2.sh)): random dataset, input {1k, 8k, 32k} × output 512 × concurrency {1, 4} × 3 runs, 6 prompts at c=1 / 20 at c=4, 2 warm-up requests. **A distinct `--seed` per (input, concurrency, run)** so no run can hit the prefix cache of a previous one (same seeds for both engines → identical prompt sets).
* **Prefill** ([`scripts/bench-prefill-v2.sh`](scripts/bench-prefill-v2.sh)): input {8k, 32k, 128k} × output 1 × c=1 × 3 runs, **no warm-up**, distinct seed per run → every TTFT is a cold prefill. `prefill tok/s = input / median TTFT`.
* **Quality**: tool-eval-bench, 88 scenarios × 8 trials, hard mode, thinking on and off, same protocol as MiaAI's published Flash-Next numbers (NVFP4 85.4 ± 2.0; EXL3 85/88).
* **Host memory**: sampled every 10 s ([`scripts/mem-monitor.sh`](scripts/mem-monitor.sh) → `results/mem_*.csv`).
* Orchestrated unattended by [`run-all.sh`](run-all.sh) (quality) and [`run-matrix-v3.sh`](run-matrix-v3.sh) (speed); logs in [`logs/`](logs/).

## Results — decode throughput

Output tok/s, mean of 3 runs (individual runs in `results/`). TTFT = median ms. `ok` = completed/total requests when not 100 %.

| input × conc | vLLM MTP on | vLLM MTP off | EXL3 MTP on | EXL3 MTP off |
|---|---|---|---|---|
| 1k × 1 | 101.6 (TTFT 259) | 78.1 (295) | **168.9** (283) | 105.3 (247) |
| 1k × 4 | **296.4** (179) | 225.1 (136) | 221.4 (638) | 198.3 (369) |
| 8k × 1 | **140.5** (193) | 95.5 (175) | 129.1 (527) | 79.7 (483) |
| 8k × 4 | **282.1** (368) | 212.7 (311) | 125.2 (1466) · ok 56/60 | 139.5 (1205) · ok 59/60 |
| 32k × 1 | **98.9** (956) | 76.7 (912) | 63.1 (2643) | 50.6 (2622) |
| 32k × 4 | **150.7** (1412) | 135.4 (1294) | 16.1 (5270) · **ok 21/60** | 15.8 (4954) · **ok 21/60** |

Median TPOT at c=1 with MTP: EXL3 **5.5 ms** vs vLLM **6.1 ms** — ExLlamaV3's pure decode step is faster; it loses on prefill and on scaling.

Takeaways: MTP is worth **+30 %** on vLLM and **+60 %** on EXL3 at c=1 (vLLM MTP acceptance length ≈3.0–3.5 of 4). EXL3 leads only at 1k×1; from 8k up vLLM is ahead, and at concurrency 4 it is not close. At 32k×4 TabbyAPI completed only 21 of 60 requests in both modes (server logs `Request disconnected`, the client receives a non-JSON SSE chunk) — a TabbyAPI/ExLlamaV3 limit on concurrent long-prompt ingestion, not an MTP issue.

## Results — prefill

Cold prefill, output 1, c=1, 3 runs (individual TTFTs in the last column).

| config | 8k | 32k | 128k (131,072) |
|---|---|---|---|
| vLLM MTP on | 290 ms → **28.2k tok/s** (213/340/290) | 1181 ms → **27.7k** (861/1423/1181) | 4662 ms → **28.1k** (6592/4662/4411) |
| vLLM MTP off | 277 ms → **29.6k** (188/304/277) | 1134 ms → **28.9k** (825/1372/1134) | 4426 ms → **29.6k** (6351/4426/4234) |
| EXL3 MTP on | 794 ms → **10.3k** (651/1010/794) | 3353 ms → **9.8k** (2563/4026/3353) | 11666 ms → **11.2k** (11666/12052/11351) · completed 1/3, 2/3, 2/3 |
| EXL3 MTP off | 772 ms → **10.6k** (633/963/772) | 3214 ms → **10.2k** (2461/3892/3214) | 11316 ms → **11.6k** (11316/11677/11002) · completed 1/3, 2/3, 2/3 |

vLLM prefills **~2.7× faster** and flat from 8k to 128k. EXL3's ~10k tok/s is far above the 1.3–1.5k reported on 4×3090, but at 128k only 5 of the 9 requests per config completed (per run: 1/3, 2/3, 2/3 — `completed` field in `results/prefill_tabby_*_in131072_r*.json`); the failures are the same non-JSON-chunk error seen at 32k×4. The TTFT medians at 128k are therefore computed on the completed requests only.

## Results — tool-calling quality (tool-eval-bench)

88 scenarios × 8 trials, hard mode, seed 42, temperature 0.6. Error rate 0.0 in every run. Reports in [`runs/`](runs/).

| engine | thinking | Final Score | Pass@8 | Pass^8 | notes |
|---|---|---|---|---|---|
| vLLM MTP on | on | 85.6 ± 2.3 | 93.2 % | 63.6 % | 33 `xgrammar Failed to advance FSM` server-log errors ² |
| vLLM MTP on | off | **87.0 ± 2.1** | 92.0 % | 56.8 % | no new xgrammar errors ² |
| vLLM MTP **off** | on | 86.5 ± 1.8 | 92.0 % | 63.6 % | **0 xgrammar errors / 2093 requests** (counted in `logs/run-matrix-v3-*.log`) |
| EXL3 MTP on | on | 86.4 ± 1.6 | 88.6 % | **67.0 %** | |
| EXL3 MTP on | off | 85.5 ± 2.2 | 89.8 % | 60.2 % | |

All five runs are within one standard deviation of each other and of MiaAI's references. EXL3 with thinking is the most repeatable (Pass^8 67 %), vLLM has the higher ceiling (Pass@8 93 %). The xgrammar errors appear **only with MTP + thinking + constrained tool-call decoding** on this vLLM build (draft tokens `\n\n` and ` ``` ` proposed at the reasoning→tool-call boundary and rejected by the grammar); they did not affect the score.

² The MTP-on counts were read live from the vLLM container log during the run (`docker logs … | grep -c 'Failed to advance FSM'`) and are recorded with timestamps in [`logs/xgrammar-observations.md`](logs/xgrammar-observations.md); that container log was not preserved, so **these two figures are operator observations, not reproducible from this repo**. The MTP-off count (0 / 2093) is in the v3 orchestrator log.

## Host memory & load times

| | vLLM MTP on | vLLM MTP off | EXL3 MTP on | EXL3 MTP off |
|---|---|---|---|---|
| Time to ready (weights from NVMe, warm page cache) | 351 s | 281 s | **88 s** | **65 s** |
| VRAM after load / max | 91.4 / 96.5 GB | 89.7 / 94.4 GB | 77.1 / 78.4 GB | 73.1 / 74.7 GB |
| Host RAM `used` (`free -h`, right after load — see `logs/run-matrix-v3-*.log`) | 10 GiB | 12 GiB | **43 GiB** | **46 GiB** |
| Engine process RSS max (`engine_rss_mib` in `results/mem_*.csv`) | 3.3 GiB ¹ | 4.0 GiB ¹ | **44.8 GiB** | **44.7 GiB** |
| Min host RAM available during run (`mem_available_mib`) | 48.9 GiB | 48.0 GiB | **7.4 GiB** | **8.3 GiB** |

¹ For vLLM the RSS column tracks the API-server/engine process only; the INT4 PLE table lives in the separate `PleOffloadWorker` process and is not included — use the `free` row for the real host footprint. The `free` figures include ~1–2 GiB of OS and background containers.
| GPU power max | 588 W | 578 W | 586 W | 593 W |

## Fixes needed to get here (all in this repo)

1. **`vm.overcommit_memory=1`** on the host. The PLE-quant overlay builds the model with the full BF16 n-gram table first (`torch.empty` of 102 GB, virtual) and only afterwards swaps in the INT4 table; with the default heuristic overcommit the kernel refuses any allocation > RAM+swap (67 GB) → `PleOffloadWorker … DefaultCPUAllocator: can't allocate memory`.
2. **`--max-num-seqs 64`** for vLLM: the default 1024 exceeds the 596 available Mamba cache blocks and CUDA graph capture aborts (`Engine core initialization failed`).
3. **`--max-model-len 135168`** for vLLM with MTP: at 262,144 the KV cache needed (7.57 GiB) exceeds what is left after the draft head (6.98 GiB).
4. **TabbyAPI image lacks `Python.h`**: Triton cannot JIT-compile `cuda_utils` for ExLlamaV3's GDN kernel (`fatal error: Python.h: No such file or directory`). [`configs/Dockerfile.tabby`](configs/Dockerfile.tabby) adds `python3-dev`.
5. **Prefix cache contamination**: `sglang.benchmark.serving` uses a fixed `--seed 42` and warms up with the first prompt, so repeated runs hit the prefix cache in both engines (TTFT of 90 ms for 131k tokens). The `v1` results in [`results-v1-seed42/`](results-v1-seed42/) are kept for transparency; **only `results/` (v3, distinct seeds) should be quoted**.

## Caveats (read before quoting)

* **CPU capped at 4.5 GHz** (`scaling_max_freq`) for the whole speed run: with boost to 5.4 GHz, the three busy cores of vLLM's PLE offload pushed Tctl to 95 °C on an air cooler. Cost ≈16 % single-thread clock; makes results more repeatable (no thermal throttling) but may slightly penalise vLLM's Python-side path.
* **Different max context**: vLLM 135,168 vs TabbyAPI 262,144. Does not affect the measured points (≤131k) but does affect capacity.
* The speed matrix keeps 2 warm-up requests on the first prompt: 1 of 6–20 requests per run has an understated TTFT; medians are unaffected.
* tool-eval-bench mislabels the backends in its reports: TabbyAPI shows as "llamacpp / llama.cpp" (unrecognised server), vLLM's quantization as "FP8" (the checkpoint is mixed NVFP4-FP8).
* Single machine, single night, 3 runs per point. Run-to-run spread is visible in the per-run numbers; treat differences under ~10 % as noise.
* Not tested: SGLang (its Flash-Next checkpoints dequantize the n-gram table to ~100 GB BF16 in host RAM — does not fit in 64 GB), llama.cpp (MTP still WIP at the time).

## Layout

```
HARDWARE.md                 hardware/software snapshot taken at run start
results/                    v3 raw client JSON (108) + host memory CSV — the numbers above
                            (the `generated_texts` field — the model's output to random-token prompts — was
                            replaced by a placeholder before publication; all timings/lengths are intact)
results-v1-seed42/          v1 raw JSON (fixed seed, prefix-cache contaminated) — for transparency only
runs/                       tool-eval-bench summaries + per-trial reports (5 runs)
configs/                    engine launch scripts, TabbyAPI config, Dockerfile for the python3-dev fix
scripts/                    matrix / prefill / readiness / memory-monitor scripts (v1 and v2)
run-all.sh, run-matrix-v3.sh  unattended orchestrators (Telegram notify calls are site-specific, harmless if absent)
logs/                       full orchestrator logs of the night
PLE-QUANT-README-primitive-ai.md  the overlay's README as downloaded (credit: primitive-ai)
```

## Credits

Checkpoints and overlay by **primitive-ai** and **turboderp**; quality protocol and reference numbers by **MiaAI** (tool-eval-bench). Benchmark run and written up by Marco Zerbato with Claude. Issues and corrections welcome.
