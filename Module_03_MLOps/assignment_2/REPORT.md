# AI Performance Engineering Assignment Report

## Environment

Deployed on an HPC cluster interactive node (`lgn15`, 1×H100 80GB) via LSF `bsub` instead of the Nebius cloud assumed by the assignment. Prometheus and Grafana ran on the local PC via Docker; connectivity used a multi-hop SSH tunnel chain (local PC → `hpcoodv01` → `lgn15`) with a reverse tunnel to expose Langfuse to the cluster.

---

## Phase 1: vLLM Serving

Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`, 1×H100.

| Flag | Rationale |
|---|---|
| `--max-model-len 8192` | Caps KV cache allocation; agent prompts are 1.5–3K tokens |
| `--gpu-memory-utilization 0.68` | Fits weights safely while reserving KV cache headroom |
| `--enable-prefix-caching` | Reuses KV cache for static schema/system-prompt prefix across loop iterations |
| `--enable-chunked-prefill` | Prevents long-prompt prefill from blocking decoding of concurrent requests |
| `--max-num-seqs 32` | Limits scheduler queue to avoid OOM under burst load |
| `--disable-log-requests` | Reduces I/O overhead under high RPS |

---

## Phase 2: Observability

Grafana dashboard (`infra/grafana/provisioning/dashboards/serving.json`) with panels: E2E Request Latency (P50/P95/P99), TTFT, TPOT, Generated Tokens/sec, Requests Running, KV Cache Usage.

**Gotcha**: Prometheus scrape target must point to `host.docker.internal:9000` (the SSH tunnel port), not `8000` (unbound on the local PC). UI edits to provisioned dashboards revert on container restart — always write changes back to the JSON file.

---

## Phase 3: LangGraph Agent

Graph: `START → attach_schema → generate_sql → execute → verify → (ok → END) / (not ok → revise → execute → verify)`, capped at `MAX_ITERATIONS=3`. The `llm()` helper targets vLLM via `ChatOpenAI` at `temperature=0.0` (revise node uses `0.4`).

---

## Phase 4: Langfuse Tracing

Keys wired via `.env` (`LANGFUSE_HOST=http://localhost:3001`). Traces tagged with `metadata={"phase": "4"}` per the README's "tag your traces" requirement. Langfuse reached from `lgn15` via a two-hop tunnel: `lgn15:3001 → hpcoodv01:3001 → local PC:3001`.

**Gotcha**: Reverse tunnel originally used `-R 3001:localhost:3000`, which silently routed to Grafana instead of Langfuse, producing structurally identical 401 errors. Fixed to `-R 3001:localhost:3001`.

---

## Phase 5: Evaluation

### Baseline (no schema descriptions)

| Metric | Value |
|---|---|
| Overall pass rate | 0.333 |
| Per-iteration pass rate | [0.333, 0.333, 0.333] |
| Iteration distribution | 1: 19, 2: 3, 3: 8 |
| Errors | 0 |

The verify→revise loop fired on 11/30 questions but changed zero outcomes — flat pass rate across all iterations.

### Root cause

`render_schema()` emitted only raw DDL. BIRD databases use cryptic column names (`A14`, `A15`, …) with no semantics. The model could not identify the correct column, and verify had no basis to diagnose it — producing only "result is empty" feedback regardless of the actual logic error.

### Fix: inlining column descriptions

Extended `render_schema()` to read BIRD's `database_description/*.csv` files and append descriptions as SQL comments (e.g., `"A15" INTEGER -- no. of committed crimes 1995`). This immediately corrected column selection on the test question.

### With descriptions

| Metric | Value |
|---|---|
| Overall pass rate | 0.300 |
| Per-iteration pass rate | [0.300, 0.267, 0.300] |
| Iteration distribution | 1: 20, 2: 2, 3: 8 |
| Errors | 0 |

Slight regression (0.333 → 0.300): richer context made verify more critical of plausible-but-imperfect queries that previously passed on exact-match. Qualitatively, verify now produces semantically meaningful feedback rather than just detecting empty results.

---

## Phase 6: SLO Tuning

Goal: hit P95 latency under 5 seconds at 5 RPS.

### Iteration log

- Baseline run showed P95 23.46s with 191 HTTP 500s and 11 client errors; this established that the system was failing both the latency and reliability targets.
- Disabled Langfuse tracing in `agent/server.py` after observing export timeouts and agent 500s; this reduced failure noise and improved P95 to 14.72s.
- Captured failing response bodies and identified `AttributeError: 'NoneType' object has no attribute 'replace'`; fixed the bug in `agent/schema.py`, which restored near-complete request success.
- Increased `--max-num-seqs` from 32 to 48 after the service became stable; this improved serving behavior, but tail latency still remained above the SLO.
- Reduced `MAX_ITERATIONS` in `agent/graph.py` from 3 to 2; this produced the first run that met the latency target at the required load.

### Final result

SLO achieved. In the final 300-second run at 5 RPS, the system completed 1499/1500 requests successfully, achieved 4.96 RPS, and recorded P50 1.18s, P95 3.94s, P99 7.60s, with a maximum latency of 16.98s.

---

## Phase 7: What I'd Do With More Time

- Include `value_description` fields (units, ranges) from BIRD CSVs for finer column disambiguation.
- Add a self-consistency check: run generated SQL and include row-count context in the verify prompt.
- Tune verify to distinguish "logically wrong" from "stylistically different but correct" to reduce false rejection of valid queries.
- Deploy on a dedicated cloud node to eliminate SSH tunnel fragility and enable reproducible load tests.
