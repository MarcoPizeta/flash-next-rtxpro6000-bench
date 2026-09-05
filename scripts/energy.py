#!/usr/bin/env python3
"""Energia GPU per fase (dai marker '#####' dei log di oggi + power-log.csv a 5 s) e Wh per 1000 token di output
della matrice v3 (mem_*.csv a 10 s + JSON). Scrive results/energy.md. Uso: energy.py BENCH_DIR"""
import sys, glob, re, json, os, datetime as dt
B = sys.argv[1]
pl = []
for l in open(f"{B}/results/power-log.csv"):
    p = l.strip().split(",")
    if len(p) >= 2:
        try: pl.append((int(p[0]), float(p[1])))
        except: pass
pl.sort()
def energy(t0, t1):
    s = [w for t, w in pl if t0 <= t < t1]
    if not s: return None, None, 0
    return sum(s) / len(s), sum(s) * 5 / 3600, len(s)   # W medi, Wh (campione ogni 5 s)
marks = []
for f in glob.glob(f"{B}/logs/run-effort-*.log") + glob.glob(f"{B}/logs/tooleval-27b-nothink-*.log") + glob.glob(f"{B}/logs/run-extra-*.log"):
    for l in open(f, errors="ignore"):
        m = re.match(r"##### (\d\d)/(\d\d)/(\d{4}) (\d\d):(\d\d):(\d\d) — (.*)", l)
        if m:
            d, mo, y, H, M, S, txt = m.groups()
            marks.append((int(dt.datetime(int(y), int(mo), int(d), int(H), int(M), int(S)).timestamp()), txt.strip(), os.path.basename(f)))
marks.sort()
out = ["# Energia GPU (power-log 5 s) — 05/09/2026", "", "## Per fase (tra marker consecutivi dei log)", "", "| inizio | fase | durata min | W medi | Wh | campioni |", "|---|---|---|---|---|---|"]
for i, (t, txt, f) in enumerate(marks):
    t1 = marks[i + 1][0] if i + 1 < len(marks) else (pl[-1][0] if pl else t)
    W, Wh, n = energy(t, t1)
    if W is None: continue
    out.append(f"| {dt.datetime.fromtimestamp(t):%H:%M} | {txt[:70]} | {(t1 - t) / 60:.0f} | {W:.0f} | {Wh:.1f} | {n} |")
out += ["", "## Matrice v3 — Wh per 1000 token di output (mem_*.csv 10 s, intero run della config incl. caricamento)", "", "| config | durata min | W medi | Wh totali | token output | Wh / 1k tok out | Wh / 1k tok in+out |", "|---|---|---|---|---|---|---|"]
for cfg in ["vllm_mtp-on", "vllm_mtp-off", "tabby_mtp-on", "tabby_mtp-off"]:
    f = f"{B}/results/mem_{cfg}.csv"
    if not os.path.exists(f): continue
    rows = [l.split(",") for l in open(f).read().splitlines()[1:] if l.strip()]
    try: P = [float(r[3]) for r in rows]
    except: continue
    dur = len(P) * 10; Wavg = sum(P) / len(P); Wh = sum(P) * 10 / 3600
    tok_out = tok_in = 0
    for j in glob.glob(f"{B}/results/{cfg}_in*.json") + glob.glob(f"{B}/results/prefill_{cfg}_in*.json"):
        d = json.load(open(j)); tok_out += d.get("total_output_tokens", 0); tok_in += d.get("total_input_tokens", 0)
    out.append(f"| {cfg} | {dur / 60:.0f} | {Wavg:.0f} | {Wh:.0f} | {tok_out} | {1000 * Wh / tok_out if tok_out else 0:.2f} | {1000 * Wh / (tok_in + tok_out) if tok_in + tok_out else 0:.3f} |")
out += ["", "Note: le fasi comprendono caricamento, warm-up e attese; i W medi dei tool-eval includono i tempi morti tra richieste. Riferimento: 27B a riposo con `--sleep-on-idle` = 12 W GPU."]
open(f"{B}/results/energy.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
