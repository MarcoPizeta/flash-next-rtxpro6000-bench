#!/usr/bin/env python3
"""tool_choice=required: replica TC-45 con i 12 UNIVERSAL_TOOLS di tool-eval-bench (o 2 tool di riserva).
Uso: required-test.py BASE_URL [MODEL]  — stream/non-stream x thinking on/off x 3 run."""
import json, sys, urllib.request
BASE = sys.argv[1].rstrip("/"); MODEL = sys.argv[2] if len(sys.argv) > 2 else "flash-next"
try:
    from tool_eval_bench.domain.tools import UNIVERSAL_TOOLS as tools
    src = f"UNIVERSAL_TOOLS ({len(tools)} tool)"
except Exception as e:
    tools = [{"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
             {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]
    src = f"2 tool di riserva ({str(e)[:40]})"
print("tools:", src)
def run(stream, think):
    body = {"model": MODEL, "max_tokens": 800, "temperature": 0.6, "tools": tools, "tool_choice": "required", "parallel_tool_calls": True,
            "messages": [{"role": "user", "content": "What is 7 times 8?"}], "chat_template_kwargs": {"enable_thinking": think}}
    if stream: body["stream"] = True
    r = urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=300)
    if not stream:
        d = json.load(r); m = d["choices"][0]["message"]
        return f"finish={d['choices'][0]['finish_reason']} tool_calls={[t['function']['name'] for t in (m.get('tool_calls') or [])]} content={repr((m.get('content') or '')[:60])}"
    names, content, finish = [], "", None
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]": continue
        ch = (json.loads(line[5:]).get("choices") or [{}])[0]
        finish = ch.get("finish_reason") or finish; delta = ch.get("delta") or {}
        names += [t["function"]["name"] for t in delta.get("tool_calls") or [] if (t.get("function") or {}).get("name")]
        content += delta.get("content") or ""
    return f"finish={finish} tool_calls={names} content={repr(content[:60])}"
ok = tot = 0
for stream in (True, False):
    for think in (False, True):
        for i in range(3):
            try:
                res = run(stream, think); tot += 1; ok += ("tool_calls=[]" not in res)
                print(f"stream={stream!s:5} think={think!s:5} run {i} → {res}", flush=True)
            except Exception as e:
                tot += 1; print(f"stream={stream!s:5} think={think!s:5} run {i} → ERRORE {str(e)[:120]}", flush=True)
print(f"RIEPILOGO required: {ok}/{tot} risposte con almeno una tool call")
