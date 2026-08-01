# Model Optimization Matrix — Multi-Dimensional Experiment Log

**Date:** 2026-07-13  
**Status:** canonical for stacked research / compound serve experiments  
**SoT files:** `experiments.jsonl` (append-only) · wiki log pages · model-kb `records.jsonl` (rebuildable)  
**Never:** overwrite a measured cell; always append a new experiment row.

---

## 1. Why multi-dimensional (not a 2D table)

A serve result is a **point in a product space**, not a single “best config” column.

| Axis | Code | Examples | Notes |
|------|------|----------|--------|
| **A Artifact** | `artifact` | Qwen3.6-35B-A3B-4bit, MiniMax-M2.7-mxfp4 | Weights on disk / HF family |
| **Q Quant** | `quant` | 4bit, 8bit, mxfp4, nvfp4, 3.6bit, mixed-4-6 | Orthogonal to accel |
| **E Engine** | `engine` | mlx_lm, mlx_vlm, mtplx, dflash-mlx, ds4 | Who runs the weights |
| **B Backend** | `backend` | single, jaccl | Topology / TP |
| **X Accelerators** | `accelerators[]` | dflash, turboquant, mtp, ddtree, prompt_cache, continuous_batching, triattention, msa | **Composable set** — order + combo matter |
| **H Hardware** | `hardware` | node-a 128GB, node-b 512GB, jaccl 2–4 node | Peak RAM, RDMA |
| **W Workload** | `workload` | context_len, concurrency, max_tokens, task (chat/code/agent) | Same recipe ≠ same score |
| **M Metrics** | `metrics` | agg_tps, per_req_tps, ttft_ms, acceptance, peak_ram_gb, quality | What we optimize |

**Compound experiments** = fix some axes, vary the **X** power set (and/or Q/E/B) and stack results over time.

---

## 2. Experiment row (append-only)

One row = one measured (or planned) point. Never edit `metrics` after `status=done`.

```json
{
  "experiment_id": "exp:2026-07-13-qwen36-35b-dflash-tq-55k",
  "ts": "2026-07-13T12:00:00Z",
  "status": "done",
  "hypothesis": "TQ+DFlash wins survivability at 55K even if tok/s drops",
  "parent_experiment_id": null,
  "campaign": "qwen36-35b-compound-2026-07",
  "axes": {
    "artifact": "Qwen3.6-35B-A3B-4bit",
    "hf_id": "Qwen/Qwen3.6-35B-A3B",
    "quant": "4bit",
    "engine": "mlx_lm",
    "backend": "single",
    "accelerators": ["dflash", "turboquant"],
    "hardware": {
      "nodes": ["hub"],
      "chip": "M3 Ultra",
      "ram_gb": 512
    },
    "workload": {
      "context_len": 55000,
      "concurrency": 1,
      "max_tokens": 256,
      "task": "long_context_gen"
    }
  },
  "metrics": {
    "agg_tps": 22.9,
    "baseline_tps": 29.1,
    "ttft_ms": null,
    "acceptance_pct": 69.0,
    "peak_ram_gb": null,
    "prefill_s": 109.6,
    "quality": null,
    "notes": "TQ -18% vs AR at 55K; value is non-OOM at higher ctx"
  },
  "baseline_ref": "exp:…-ar-only-55k",
  "recipe_id": "qwen36-35b-a3b-tq-dflash-1node",
  "source": {
    "type": "wiki|bench|manual",
    "path": "~/wiki/atlas/concepts/research/inference/dflash-turboquant-mlx.md",
    "span": "Long Context A/B table"
  },
  "confidence": "measured",
  "author": "ma"
}
```

### Status lifecycle

`planned` → `running` → `done` | `failed` | `aborted`

Failed rows keep error text; they still occupy the matrix cell as “tried”.

---

## 3. Compound matrix (what we actually search)

### 3.1 Singleton axes (usually fixed per campaign)

Pick **one artifact** (and quant) per campaign so X/E/B stay comparable.

### 3.2 Composable accelerator catalog (X)

From live serve-configs + research (extend as discovered):

| Accel | Compounds with | Known anti-compounds / caveats |
|-------|----------------|--------------------------------|
| dflash | turboquant, ddtree, jaccl, prompt_cache | Long-ctx falls back toward AR; prefill cost |
| turboquant | dflash, long-ctx, jaccl | Overhead can lose short-ctx; wins survivability |
| ddtree | dflash | +10–15% on code; weak on prose |
| mtp / mtplx | single-node, some MoE | Engine-specific |
| jaccl | large MoE, multi-node | RDMA required; not a free “speed” toggle |
| prompt_cache | multi-turn chat | Workload-dependent |
| continuous_batching | concurrency>1 | Different metric: agg vs per-req |
| triattention | mtplx / research engines | Narrow support |

### 3.3 Campaign = partial factorial

Do **not** full-factorial every axis (combinatorial explosion).

```
Campaign: <artifact>@<quant> on <hardware>
  fixed: artifact, quant, hardware, workload suite W={short, mid, long}
  vary:  power set of interesting X (size ≤ 8 combos per wave)
  optional vary: engine, backend
```

Waves:

1. **Baselines** — AR / plain engine only  
2. **Singles** — each accel alone  
3. **Pairs** — top singles × each other (dflash+TQ, dflash+ddtree, …)  
4. **Triples+** — only if pair Pareto-improves baseline  
5. **Topology** — promote winners to jaccl / multi-node  

---

## 4. Stacked wiki log (human-readable SoT for narrative)

Path convention:

```
~/wiki/atlas/concepts/research/inference/campaigns/
  <campaign-id>/
    README.md                 # hypothesis, axes fixed, wave plan
    matrix.md                 # pivot views (see §5)
    log.md                    # chronological append-only narrative
    experiments/              # optional per-run notes
      <experiment_id>.md
```

### log.md entry template

```markdown
## YYYY-MM-DD — <experiment_id>
- **status:** done
- **axes:** engine=mlx_lm backend=single X=[dflash,turboquant] W=55k@c1
- **metrics:** 22.9 tok/s (baseline 29.1) acceptance 69%
- **delta vs baseline:** -18% tok/s; +survivability
- **decision:** keep for long-ctx wave; drop for short-ctx default
- **next:** pair dflash+ddtree at short ctx
```

Machine SoT remains `~/infra/model-kb/experiments.jsonl`. Wiki is the narrative stack.

---

## 5. Matrix views (derived, not hand-maintained)

Generate from `experiments.jsonl` (script: `matrix_report.py`):

1. **Accel compound grid** — rows/cols = accel sets, cell = best agg_tps for fixed A/Q/H/W  
2. **Workload heat** — same X, columns = short/mid/long context  
3. **Pareto** — tok/s vs peak_ram vs quality (when quality present)  
4. **Coverage** — which (A,Q,X,W) cells are empty / planned / failed  

Empty cells = next experiment candidates.

---

## 6. Join into model-kb recipes

On `model_kb ingest` / `pipeline.py`:

1. Load `experiments.jsonl` where `status=done` and `confidence=measured`  
2. If `recipe_id` set → append to that recipe’s `benchmarks[]` (never delete old rows)  
3. Else match artifact+quant+engine+backend+accel set → best recipe  
4. Emit `kind=experiment` rows into BM25 so “55k turboquant dflash qwen” retrieves evidence  

---

## 7. CLI surface

```bash
# append a planned or measured point
python3 experiment_log.py add --campaign ... --artifact ... --accels dflash,turboquant ...

# list coverage holes for a campaign
python3 matrix_report.py --campaign qwen36-35b-compound-2026-07 --view coverage

# full rebuild including experiments
python3 pipeline.py
```

---

## 8. Hard rules

1. **No invented numbers** — metrics only from bench/wiki tables/runs.  
2. **Append-only metrics** — corrections = new row with `supersedes`.  
3. **Compound ≠ sum of singles** — always remeasure pairs.  
4. **Workload is an axis** — never compare 512-ctx to 55k-ctx as “better recipe”.  
5. **asmi load** still only via ServeRequest from a recipe — experiments inform recipes; they don’t bypass them.
