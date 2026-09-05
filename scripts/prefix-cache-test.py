#!/usr/bin/env python3
"""Prefix-cache test (caso Hermes): system prompt identico da ~4.700 token (52 definizioni di tool finte),
domanda utente diversa a ogni richiesta, concorrenza 1. Misura il TTFT della prima richiesta (prefisso freddo)
e delle successive (prefisso in cache). Uso: prefix-cache-test.py BASE_URL MODEL OUT.json [N=8]
Stesso client per tutti i motori: /v1/chat/completions in streaming, TTFT = primo chunk con contenuto."""
import json, sys, time, random, urllib.request

base, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 8
random.seed(7)

# ~4.700 token di "definizioni di tool" deterministiche (stile Hermes/openclaw)
tools = []
for i in range(52):
    tools.append({"type": "function", "function": {
        "name": f"tool_{i:02d}_{random.choice(['get','set','list','create','update','delete'])}_{random.choice(['order','customer','invoice','shipment','ticket','machine','lot','report'])}",
        "description": f"Tool number {i}. " + " ".join(random.choice(["Returns", "Updates", "Validates", "Filters", "Aggregates", "Exports"]) + " " + random.choice(["records", "items", "rows", "entries", "documents"]) + " by " + random.choice(["id", "date range", "customer code", "status", "machine", "operator"]) + "." for _ in range(6)),
        "parameters": {"type": "object", "properties": {f"param_{k}": {"type": random.choice(["string", "integer", "boolean"]), "description": f"Parameter {k} for tool {i}: " + " ".join(random.choice(["optional", "required", "ISO date", "customer code", "numeric id", "free text"]) for _ in range(5))} for k in range(6)}, "required": [f"param_{k}" for k in range(3)]}}})
system = "You are the operations assistant. Follow the tool contracts exactly. " + " ".join(f"Rule {i}: " + " ".join(random.choice(["always", "never", "when possible", "unless told otherwise"]) + " " + random.choice(["confirm", "validate", "log", "escalate", "summarize"]) + " " + random.choice(["the request", "the result", "the customer", "the lot", "the shipment"]) + "." for _ in range(4)) for i in range(40))
questions = [f"Question {i}: which tool would you use to {random.choice(['list', 'update', 'export', 'validate'])} the {random.choice(['orders', 'shipments', 'tickets', 'lots'])} of customer C{random.randint(1000, 9999)} for {random.choice(['last week', 'March 2026', 'today', 'Q2'])}? Answer in one short sentence, do not call any tool." for i in range(N)]

def one(q):
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": q}],
            "tools": tools, "tool_choice": "none", "max_tokens": 48, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if ttft is None and d.get("choices") and (d["choices"][0].get("delta") or {}).get("content"): ttft = time.perf_counter() - t0
            if d.get("usage"): usage = d["usage"]
    return ttft, time.perf_counter() - t0, usage

res = []
for i, q in enumerate(questions):
    ttft, tot, usage = one(q)
    res.append({"i": i, "ttft_ms": round((ttft or tot) * 1000), "total_ms": round(tot * 1000), "usage": usage})
    print(f"  req {i}: TTFT {res[-1]['ttft_ms']} ms  total {res[-1]['total_ms']} ms  {'(prefisso freddo)' if i == 0 else ''}", flush=True)
warm = [r["ttft_ms"] for r in res[1:]]
summary = {"base_url": base, "model": model, "n": N, "cold_ttft_ms": res[0]["ttft_ms"], "warm_ttft_ms_median": sorted(warm)[len(warm) // 2] if warm else None,
           "warm_ttft_ms_min": min(warm) if warm else None, "prompt_tokens": (res[0]["usage"] or {}).get("prompt_tokens"), "requests": res}
json.dump(summary, open(out, "w"), indent=1)
print(f"  prompt {summary['prompt_tokens']} tok — TTFT freddo {summary['cold_ttft_ms']} ms, caldo mediano {summary['warm_ttft_ms_median']} ms")
