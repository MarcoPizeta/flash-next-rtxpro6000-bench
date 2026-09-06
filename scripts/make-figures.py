#!/usr/bin/env python3
"""Due figure PNG per il thread X / schede HF, dai numeri del README v2 (results/*.json e runs/*_summary.md)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_VLLM, C_EXL3, C_27B = "#2a78d6", "#eb6834", "#1baf7a"   # slot 1-3, validated
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 11,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})

def style(ax, ylabel=None):
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    if ylabel: ax.set_ylabel(ylabel, color=INK2)

def rounded_bars(ax, xs, vals, color, width, labels=None, label_color=INK):
    ax.bar(xs, vals, width=width, color=color, linewidth=0)
    if labels:
        for x, v, t in zip(xs, vals, labels):
            if t: ax.text(x, v + ax.get_ylim()[1] * 0.012, t, ha="center", va="bottom", fontsize=9, color=label_color)

# ------------------------------------------------------------------ Figure 1: speed
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6.75), dpi=200, gridspec_kw={"width_ratios": [2.2, 1]})
fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.17, wspace=0.28)
pts = ["1k × 1", "1k × 4", "8k × 1", "8k × 4", "32k × 1", "32k × 4"]
vllm = [101.6, 296.4, 140.5, 282.1, 98.9, 150.7]
exl3 = [168.9, 221.4, 129.1, 125.2, 63.1, 16.1]
x = np.arange(len(pts)); w = 0.36; gap = 0.03
a1.set_xlim(-0.6, len(pts) - 0.4); a1.set_ylim(0, 330)
rounded_bars(a1, x - w / 2 - gap / 2, vllm, C_VLLM, w, [f"{v:.0f}" for v in vllm])
rounded_bars(a1, x + w / 2 + gap / 2, exl3, C_EXL3, w, [f"{v:.0f}" for v in exl3])
a1.text(x[3] + w / 2 + gap / 2, 125.2 + 22, "56/60 ok", ha="center", fontsize=8, color=INK2)
a1.text(x[5] + w / 2 + gap / 2, 16.1 + 22, "21/60 ok", ha="center", fontsize=8, color=INK2)
a1.set_xticks(x); a1.set_xticklabels(pts, color=INK2)
a1.set_xlabel("input tokens × concurrent requests  (output 512, MTP on, mean of 3 runs)", color=INK2)
style(a1, "output tok/s")
a1.set_title("Decode throughput", loc="left", fontsize=13, color=INK, pad=10)

pre = ["8k", "32k", "128k"]; pv = [28.2, 27.7, 28.1]; pe = [10.3, 9.8, 11.2]
x2 = np.arange(3); a2.set_xlim(-0.6, 2.6); a2.set_ylim(0, 33)
rounded_bars(a2, x2 - w / 2 - gap / 2, pv, C_VLLM, w, [f"{v:.1f}k" for v in pv])
rounded_bars(a2, x2 + w / 2 + gap / 2, pe, C_EXL3, w, [f"{v:.1f}k" for v in pe])
a2.text(x2[2] + w / 2 + gap / 2, 11.2 + 2.6, "5/9 ok", ha="center", fontsize=8, color=INK2)
a2.set_xticks(x2); a2.set_xticklabels(pre, color=INK2); a2.set_xlabel("input tokens (cold, c=1)", color=INK2)
style(a2, "prefill tok/s (×1000)")
a2.set_title("Prefill", loc="left", fontsize=13, color=INK, pad=10)

fig.text(0.07, 0.94, "Qwen3.8-Flash-Next on one RTX PRO 6000 Blackwell (96 GB) + 64 GB host RAM", fontsize=16, color=INK, weight="bold")
fig.text(0.07, 0.895, "vLLM (primitive-ai NVFP4/FP8 + INT4 PLE offload)  vs  ExLlamaV3 / TabbyAPI (turboderp EXL3 4.05 bpw) — same client, distinct seed per run", fontsize=10.5, color=INK2)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_VLLM), plt.Rectangle((0, 0), 1, 1, color=C_EXL3)],
           labels=["vLLM + MTP", "ExLlamaV3 + MTP"], loc="upper right", bbox_to_anchor=(0.98, 0.875), ncol=2, frameon=False,
           fontsize=10.5, labelcolor=INK2, handlelength=1.2, handleheight=0.9)
fig.text(0.07, 0.06, "Host RAM ~10 GB (vLLM) vs ~45 GB (EXL3, ngram_ram) · load 5–6 min vs 65–88 s · EXL3 5.05 bpw runs at the same speed as 4.05 · CPU capped at 4.5 GHz", fontsize=9, color=MUTED)
fig.text(0.07, 0.03, "Raw data, configs, scripts: github.com/MarcoPizeta/flash-next-rtxpro6000-bench", fontsize=9, color=MUTED)
fig.savefig(f"{OUT}/speed.png"); plt.close(fig)

# ------------------------------------------------------------------ Figure 2: quality
rows = [  # label, score, sigma, color
    ("Qwen3.8-27B NVFP4 · SGLang · medium (production)", 91.0, 1.5, C_27B),
    ("Qwen3.8-27B BF16 · SGLang · medium", 89.0, 1.7, C_27B),
    ("Flash-Next · vLLM · low", 87.9, 1.9, C_VLLM),
    ("Flash-Next · EXL3 4.05 · low", 87.4, 1.5, C_EXL3),
    ("Flash-Next · vLLM · low + discipline template", 87.2, 1.6, C_VLLM),
    ("Flash-Next · vLLM · thinking off", 87.0, 2.1, C_VLLM),
    ("Flash-Next · EXL3 5.05 · low", 86.9, 1.6, C_EXL3),
    ("Qwen3.8-27B NVFP4 · thinking off", 86.5, 1.8, C_27B),
    ("Flash-Next · vLLM · medium", 86.5, 3.3, C_VLLM),
    ("Flash-Next · vLLM MTP off · xhigh", 86.5, 1.8, C_VLLM),
    ("Flash-Next · EXL3 4.05 · xhigh", 86.4, 1.6, C_EXL3),
    ("Flash-Next · vLLM · xhigh (default)", 85.6, 2.3, C_VLLM),
    ("Flash-Next · EXL3 4.05 · thinking off", 85.5, 2.2, C_EXL3),
]
fig, (b1, b2) = plt.subplots(1, 2, figsize=(12, 6.75), dpi=200, gridspec_kw={"width_ratios": [2.4, 1]})
fig.subplots_adjust(left=0.36, right=0.97, top=0.76, bottom=0.13, wspace=0.35)
y = np.arange(len(rows))[::-1]; h = 0.62
for yi, (lab, s, sd, c) in zip(y, rows):
    b1.barh(yi, s - 80, left=80, height=h, color=c, linewidth=0)
    b1.errorbar(s, yi, xerr=sd, fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3)
    b1.text(s + sd + 0.25, yi, f"{s:.1f}", va="center", fontsize=9.5, color=INK)
b1.set_yticks(y); b1.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK2)
b1.set_xlim(80, 95); b1.set_ylim(-0.7, len(rows) - 0.3)
b1.grid(axis="x", color=GRID, linewidth=0.8); b1.set_axisbelow(True); b1.tick_params(length=0)
b1.set_xlabel("tool-eval-bench Final Score (88 scenarios × 8 trials, hard mode, ± 1σ)", color=INK2)
b1.set_title("Tool-calling: the 27B leads with thinking on, ties with thinking off", loc="left", fontsize=12.5, color=INK, pad=10)

# right: where they tie — AIME and thinking-off, plus required
b2.set_xlim(-0.6, 1.6); b2.set_ylim(0, 100)
xa = np.array([0, 1]); wa = 0.5
vals = [80.0, 83.3]; cols = [C_27B, C_VLLM]
for xi, v, c in zip(xa, vals, cols):
    b2.bar(xi, v, width=wa, color=c, linewidth=0)
    b2.text(xi, v + 1.5, f"{v:.1f} %", ha="center", fontsize=10, color=INK)
b2.set_xticks(xa); b2.set_xticklabels(["27B NVFP4\n48/60", "Flash-Next\n50/60"], color=INK2, fontsize=9.5)
style(b2, "accuracy %")
b2.set_title("AIME 2025+2026: a tie", loc="left", fontsize=12.5, color=INK, pad=10)
b2.text(0.5, -24, "xhigh, 32k max tokens; 18 (27B) and 12 (Flash-Next)\nanswers hit the cap and count as wrong", ha="center", va="top", fontsize=8.5, color=MUTED, clip_on=False)

fig.text(0.04, 0.94, "Flash-Next vs Qwen3.8-27B: quality on the same GPU", fontsize=16, color=INK, weight="bold")
fig.text(0.04, 0.895, "Best reasoning effort for each engine. The 3-point gap sits in 18 of 88 scenarios (Flash-Next leads in 8); "
         "one of them is tool_choice=\"required\", not enforced by this vLLM build.", fontsize=10.5, color=INK2)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_27B, C_VLLM, C_EXL3)],
           labels=["Qwen3.8-27B (SGLang)", "Flash-Next on vLLM", "Flash-Next on ExLlamaV3"], loc="upper right",
           bbox_to_anchor=(0.97, 0.875), ncol=3, frameon=False, fontsize=10.5, labelcolor=INK2, handlelength=1.2, handleheight=0.9)
fig.text(0.04, 0.03, "Quantization is not the cause: 27B BF16 89.0 ≤ NVFP4 91.0; EXL3 5.05 bpw 86.9 ≈ 4.05 bpw 87.4.  Same seed (42), temperature 0.6, 14 runs.  "
         "github.com/MarcoPizeta/flash-next-rtxpro6000-bench", fontsize=9, color=MUTED)
fig.savefig(f"{OUT}/quality.png"); plt.close(fig)
print("ok", os.listdir(OUT))
