#!/usr/bin/env python3
"""Eval di ragionamento (AIME 2025 + 2026, risposta intera 0-999) contro un endpoint OpenAI-compatibile.
Sampling consigliato da Qwen per il thinking: T=1.0, top_p 0.95, top_k 20. Estrae \\boxed{...} o l'ultimo intero.
Uso: reasoning-eval.py BASE_URL MODEL EFFORT OUT.jsonl [CONC=4] [MAX_TOKENS=32000] [LIMIT=0]"""
import json, sys, re, time, concurrent.futures as cf, urllib.request, glob

base, model, effort, out = sys.argv[1:5]
conc = int(sys.argv[5]) if len(sys.argv) > 5 else 4
max_tokens = int(sys.argv[6]) if len(sys.argv) > 6 else 32000
limit = int(sys.argv[7]) if len(sys.argv) > 7 else 0

probs = []
for f in sorted(glob.glob("/opt/flash-next/eval/data/math-ai_aime2*/**/*.jsonl", recursive=True)):
    if "/.cache/" in f: continue
    year = "2026" if "aime26" in f else "2025"
    for l in open(f):
        l = l.strip()
        if not l: continue
        d = json.loads(l); probs.append({"src": f"aime{year}", "id": f"aime{year}-{d.get('id')}", "problem": d["problem"], "answer": int(d["answer"])})
if limit: probs = probs[:limit]
print(f"{len(probs)} problemi, modello {model} effort {effort} conc {conc}", flush=True)

SYS = "Solve the problem. Reason carefully, then give the final answer as an integer between 0 and 999 inside \\boxed{}."
def extract(text):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text or "")
    cand = m[-1] if m else None
    if cand is None:
        nums = re.findall(r"-?\d+", text or ""); cand = nums[-1] if nums else None
    try: return int(re.sub(r"[^\d-]", "", cand))
    except: return None

def one(p):
    body = {"model": model, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": p["problem"]}],
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": effort}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=3600)); m = d["choices"][0]["message"]
        content = m.get("content") or ""; reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
        pred = extract(content) if content.strip() else extract(reasoning)
        return {**p, "pred": pred, "ok": pred == p["answer"], "finish": d["choices"][0].get("finish_reason"), "usage": d.get("usage"), "secs": round(time.time() - t0, 1), "content_tail": content[-300:]}
    except Exception as e:
        return {**p, "pred": None, "ok": False, "error": str(e)[:200], "secs": round(time.time() - t0, 1)}

res = []
with cf.ThreadPoolExecutor(conc) as ex:
    for r in ex.map(one, probs):
        res.append(r); tok = (r.get("usage") or {}).get("completion_tokens")
        print(f"  {r['id']:<14} {'OK ' if r['ok'] else 'no '} pred={r['pred']} ans={r['answer']} tok={tok} {r['secs']}s {r.get('finish','')} {r.get('error','')}", flush=True)
        with open(out, "w") as f:
            for x in res: f.write(json.dumps(x) + "\n")
n = len(res); k = sum(r["ok"] for r in res)
for src in ("aime2025", "aime2026"):
    s = [r for r in res if r["src"] == src]; print(f"  {src}: {sum(r['ok'] for r in s)}/{len(s)}")
toks = [(r.get("usage") or {}).get("completion_tokens", 0) for r in res]
trunc = sum(1 for r in res if r.get("finish") == "length")
print(f"RISULTATO {model} effort={effort}: {k}/{n} = {100*k/n:.1f}%  token medi {sum(toks)/max(1,len(toks)):.0f}  troncati {trunc}  tempo tot {sum(r['secs'] for r in res)/conc/60:.0f} min")
