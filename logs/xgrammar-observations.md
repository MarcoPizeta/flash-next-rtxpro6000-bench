# xgrammar `Failed to advance FSM` — operator observations (vLLM MTP on, tool-eval-bench run of 04/09/2026)

These counts were taken **live** during the `run-all.sh` tool-eval phase with
`docker logs flash-next-vllm 2>&1 | grep -ac 'Failed to advance FSM'` (cumulative since container start at 21:40 local time).
The container was removed by the orchestrator at the end of the phase and its log was not preserved,
so these numbers **cannot be re-derived from this repository**. They are reported for transparency only.

| local time | tool-eval phase | cumulative errors | cumulative `POST /v1/chat/completions` (all phases) |
|---|---|---|---|
| 22:05 | thinking ON | 10 (7 distinct requests) | 651 |
| 22:17 | thinking ON | 14 | 1071 |
| 22:27 | thinking ON | 21 | 1429 |
| 22:37 | thinking ON | 28 | 1753 |
| 22:47 | thinking ON finished 22:44, thinking OFF started | 33 | 2169 |
| 22:57 | thinking OFF | 33 | 2920 |
| 23:07 | thinking OFF | 33 | 3667 |

* First error at 21:51 local time. All 33 occurred during the thinking-ON phase; none during thinking-OFF (≈2500 further requests).
* Tokens rejected by the grammar (decoded with the model tokenizer): id 271 = `\n\n` (majority), id 71093 = ` ``` `, plus single occurrences of ids 760 and 248058.
* Every affected request still completed with HTTP 200; the tool-eval error rate was 0.0.
* Control run without MTP (thinking ON, `run-matrix-v3.sh` final phase, 05/09/2026 03:02–04:32): **0 errors in 2093 requests** — counted by the orchestrator and written to `run-matrix-v3-20260905-0107.log`.

Example log line:

```
(EngineCore pid=739) ERROR 09-04 20:00:14 [backend_xgrammar.py:167] Failed to advance FSM for request chatcmpl-af3c2f07f55c9768-af43f69a for tokens 271. Please file an issue.
```
