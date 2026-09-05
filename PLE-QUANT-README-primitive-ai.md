---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- qwen3.8-flash-next
- ple
- quantization
---

# Qwen3.8-Flash-Next quantized PLE tables

The 51.2B-parameter n-gram (PLE) table is the reason this model wants ~100 GB of free host RAM:
the CPU-offload worker holds it in BF16 (95.4 GB). This repo ships the same table quantized —
**FP8 per-row (49 GB)**, **INT4 group-16 (32 GB)**, and **NVFP4-style group-16 e2m1
(28.8 GB)** — plus a two-file overlay for the `vllm/vllm-openai:qwen38-flash-next` image that
serves them memory-mapped straight from disk. Host RAM cost becomes page cache only,
reclaimable under pressure.

Built from the original BF16 tables, so it works with any checkpoint of this model that keeps
them: [our mixed NVFP4/FP8 build](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8),
[our plain NVFP4](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-NVFP4), or the original
model. It does not apply to checkpoints that re-quantized the tables themselves.

## Serve

```bash
hf download primitive-ai/Qwen3.8-Flash-Next-PLE-quant \
  worker_image_quant.py ple_layer_quant.py --local-dir .
hf download primitive-ai/Qwen3.8-Flash-Next-PLE-quant --include "ples_int4/*" --local-dir .
# (or ples_fp8/* for the FP8 table)

docker run --gpus all --ipc=host -p 8000:8000 \
  -v $PWD/worker_image_quant.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/worker.py:ro \
  -v $PWD/ple_layer_quant.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
  -v $PWD/ples_int4:/ples_int4 -e VLLM_PLE_QUANT_DIR=/ples_int4 \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600 \
  -e VLLM_GDN_DECODE_KERNEL=triton \
  vllm/vllm-openai:qwen38-flash-next \
  --model primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

Drop `-e VLLM_GDN_DECODE_KERNEL=triton` when serving the plain NVFP4 build. No container
memory cap needed: unlike the BF16 disk path, the quantized tables fit the page cache next to
checkpoint streaming.

## Measured

One RTX PRO 6000 Blackwell (96 GB), 176 GB host, local NVMe, mixed NVFP4/FP8 checkpoint.
Throughput: 8K in / 512 out, prefix-cache-free, two seeds (shown a / b). Accuracy: the same
pinned 1,170-item knowledge + 200-item tool-calling protocol as the model cards, thinking on.

| table | size | host RSS | boot | tok/s @ 1 | tok/s @ 32 | TTFT @ 1 | knowledge | tool-calling (n=3) |
|---|---|---|---|---|---|---|---|---|
| BF16, in RAM (baseline) | 95.4 GB | ~95 GB | 302 s | 84.5 / 84.4 | 516.8 / 523.6 | 569 / 573 ms | 92.2 | 79.2 |
| FP8 per-row, mmapped | 49 GB | 52.6 GB° | 364 s | 80.3 / 80.1 | 489.7 / 500.7 | 759 / 768 ms | 92.2 | 77.7 |
| INT4 group-16, mmapped | 32 GB | 32.9 GB° | 333 s | 80.2 / 80.1 | 483.6 / 487.9 | 663 / 671 ms | 92.9 | 78.2 |
| NVFP4 group-16 e2m1, mmapped | 28.8 GB | 29.8 GB° | 417 s | 80.3 / 80.1 | 476.8 / 479.5 | 648 / 656 ms | 92.2 | 78.7 |

° mapped file pages, reclaimable under memory pressure — not anonymous RAM. Tool-calling is
the pooled 200-item suite, mean of three runs per format (suite repeat spread ±1.5; all four
formats sit in one band). Generation-sanity gates passed on every configuration.

Validated end to end inside a **48 GB container** (INT4 table): sanity PASS, tool-calling
80.5, 79.4 tok/s @ 1 and 486 @ 32 — within noise of uncapped. A 64 GB-RAM host serves this
180B model.

## Format

Sidecars are 128 shard files (`shard_N.safetensors`, 2,500,012 rows each, concatenated in
shard order) plus `META.json`. Row width 160.

| variant | tensors per shard | dequant |
|---|---|---|
| `ples_fp8` | `weight_fp8` [rows, 160] e4m3fn; `weight_scale` [rows] fp32 | `row = fp8 * scale[row]` |
| `ples_int4` | `weight_i4` [rows, 80] uint8, two nibbles, low first; `weight_scale` [rows, 10] fp16 | `row[c] = (nibble - 8) * scale[row, c // 16]` |
| `ples_nvfp4` | `weight_e2m1` [rows, 80] uint8 (code = mag index \| sign<<3, mags 0,.5,1,1.5,2,3,4,6); `weight_scale` [rows, 10] e4m3fn; `weight_scale_2` [] fp32 | `row[c] = lut[code] * scale[row, c // 16] * scale_2` |

The overlay maps every shard with safetensors' native mmap and dequantizes only the gathered
rows (~100–200 KB per decoded token), so cold-start cost and steady-state RAM both scale with
the working set, not the table.

## Notes

- The overlay targets this exact image; the gather hook lives in a vendored model file
  (`vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py`), which is why this ships as an overlay
  rather than a vLLM PR. The BF16 disk path, which needs no model-file hook, is PR
  [vllm-project/vllm#54070](https://github.com/vllm-project/vllm/pull/54070).
- MTP speculative decoding (`num_speculative_tokens: 3`) composes well with the quantized
  tables: 129.6 tok/s single-stream on the real-prompt eval with the INT4 table and 128.8 with
  NVFP4, vs 142.6 with the BF16 table in RAM and 77.5–82.3 with the BF16 table on NVMe.
  Speculation multiplies gather traffic; the quantized working sets still fit the page cache
  where the BF16 one does not, so quantization is what makes MTP + low-RAM hosts viable
  together.
- The image's stock worker cannot load quantized tables at all (it rejects
  `ngram_embedding.weight_scale`), which on this image also rules out CPU offload for checkpoints
  that ship FP8 tables with a global scale. Upstream has moved since: the offload PR branch
  ([#53899](https://github.com/vllm-project/vllm/pull/53899)) loads FP8 and NVFP4 global-scale tables since 2026-08-30, and
  [#54129](https://github.com/vllm-project/vllm/pull/54129) memory-maps the FP8 table out of the checkpoint shards. Both work at one
  scale per table; the sidecars here are per-row (FP8) or per-16-column group (INT4, NVFP4) and
  attach to any BF16-table checkpoint.
- Two other community packagings of the table exist: [edougawa](https://huggingface.co/edougawa/Qwen3.8-Flash-Next-W4A16-PLE4-MTP-Spark) carries it as per-row
  INT4 with FP16 scales inside the checkpoint (needs their vLLM patch), and [HamboneLabs](https://huggingface.co/HamboneLabs-AI/Qwen3.8-Flash-Next-uint3-g64)
  maps a raw BF16 file the way the disk overlay here does (their patch set). Neither was measured
  here.
- The current image (unchanged as of 2026-09-03) predates a fix for a startup race in the PLE
  offload path (vLLM hangs after CUDA graph capture, looping `No available shared memory broadcast
  block`; [vllm-project/vllm#53960](https://github.com/vllm-project/vllm/issues/53960), fixed on
  2026-08-29 in the offload PR branch, #53899, still unmerged). This repo ships the fixed connector as [`connector_mrv2.py`](./connector_mrv2.py) —
  add a third mount:
  `-v $PWD/connector_mrv2.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/connector.py:ro`.
  Verified with the overlays here: normal boot, sanity and accuracy unchanged.

---

<p align="center">
  <br>
  <img src="https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8/resolve/main/assets/primitive-logo.png" alt="Primitive" width="34"><br>
  <sub>
    <a href="https://primitive.com"><b>primitive</b></a> ·
    <a href="https://huggingface.co/primitive-ai">more models</a> ·
    inference economics for production LLM systems
  </sub>
</p>
