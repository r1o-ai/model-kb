# Model KB Schema — BM25 + RRF over quantizations & serve enrichments

**Date:** 2026-07-11  
**Status:** canonical for model-kb + asmi serve path  
**Depends on:** asmi control plane, `~/.r1o/serve-configs.json` v3, qwen-bm25-hybrid, model-inventory, note-reader/wiki research  
**ARDD plan (sources + multimodal + updates):** `~/.claude/docs/plans/2026-07-11-model-kb-ardd-plan.md`

## Problem

Unstructured / half-structured blobs today:

| Source | Shape | Problem |
|--------|--------|---------|
| asmi `/models` | path, size, config.json arch | no quant vocabulary, no t/s, no recipe |
| `serve-configs.json` | rich profiles + `benchmark_results` + `accelerators[]` | not searchable; 31 profiles, no hybrid index |
| model-inventory Supabase | 85 cols + benchmarks + fit_scores | registry, not retrieval |
| ad-hoc notes / chat | “use dflash on 35b, ~X tok/s” | not machine-loadable |

**Requirement:** BM25 + RRF fusion over (1) quantizations, (2) serve-config enrichments (t/s × optimizers: dflash, turboquant, jaccl, mtp, …), (3) one standardized object when *serving*.

Semantic VL embed is optional later. **BM25 + RRF is mandatory now.**

---

## Dual identity (never collapse these)

| Identity | What it is | Key |
|----------|------------|-----|
| **Model artifact** | Weights on disk / HF | `artifact_id` = stable hash of `(node, path)` or HF id |
| **Serve recipe** | How you run it | `recipe_id` = kebab name from serve-configs (or generated) |

One artifact → many recipes  
(e.g. `Qwen3.6-35B-A3B-4bit` × {baseline, dflash, turboquant+dflash, jaccl-2node}).

Retrieval returns **recipes** (or artifact+best-recipe).  
**asmi load** always materializes a **ServeRequest** from a recipe — never free-form JSON from chat.

---

## L0 record (canonical, objective)

One row per **recipe** (preferred) or artifact-only stub if no recipe yet.

```json
{
  "id": "recipe:qwen36-35b-a3b-dflash-1node",
  "kind": "serve_recipe",
  "artifact": {
    "name": "Qwen3.6-35B-A3B-4bit",
    "path": "/path/to/models/Qwen3.6-35B-A3B-4bit",
    "hf_id": "Qwen/Qwen3.6-35B-A3B",
    "nodes": ["node-N", "node-N"],
    "size_gb": 19.0,
    "params_total": "35B",
    "params_active": "3B",
    "architecture": "MoE",
    "arch_class": "Qwen3_5MoeForConditionalGeneration",
    "is_vlm": true,
    "is_moe": true,
    "required_engine": null
  },
  "quant": {
    "label": "4bit",
    "scheme": "affine",
    "bits": 4,
    "group_size": 64,
    "bpw": null,
    "aliases": ["4-bit", "q4", "mlx-4bit"]
  },
  "serve": {
    "engine": "mlx_lm",
    "backend": "single",
    "port": 19080,
    "decode_concurrency": 1,
    "prefill_step_size": null,
    "prompt_cache_size": null
  },
  "accelerators": [
    {
      "type": "dflash",
      "enabled": true,
      "draft_model": "/path/to/models/Qwen3.6-35B-A3B-DFlash",
      "params": { "block_size": 16, "verify_mode": "lossless" }
    }
  ],
  "hardware": {
    "min_nodes": 1,
    "chip": "M5 Max",
    "ram_gb": 128,
    "rdma_required": false,
    "peak_memory_gb": null,
    "max_context": null
  },
  "benchmarks": [
    {
      "id": "1x@256",
      "concurrency": 1,
      "max_tokens": 256,
      "context_len": null,
      "agg_tps": 45.2,
      "per_req_tps": 45.2,
      "ttft_ms": null,
      "p50_ms": null,
      "p99_ms": null,
      "wall_s": 5.6,
      "node": "node-N",
      "date": "2026-07-01",
      "source": "serve-configs"
    }
  ],
  "roles_fit": ["task", "default_local", "active"],
  "source": {
    "serve_config_name": "qwen35-35b-a3b-dflash-1node",
    "asmi_path": "/path/to/models/Qwen3.6-35B-A3B-4bit",
    "inventory_model_id": null
  },
  "known_issues": [],
  "verified_on": []
}
```

### Quant vocabulary (normalize on ingest)

| Input noise | Canonical `quant.label` |
|-------------|-------------------------|
| 4bit, 4-bit, q4, mlx-4bit | `4bit` |
| 8bit, q8 | `8bit` |
| 6bit | `6bit` |
| mxfp4, MXFP4 | `mxfp4` |
| mixed-4-6, mpq-… | `mixed-4-6` / keep full label in `quant.label`, parse `bits`/`bpw` when present |
| 2.7bpw-dynamic | `2.7bpw-dynamic` |
| bf16 | `bf16` |
| default / empty | `unknown` |

Always keep **raw** string in `quant.raw` for BM25.

### Accelerator vocabulary

`type` ∈  
`dflash | turboquant | jaccl | mtp | speculative | prompt_cache | continuous_batching | triattention | msa | mtp_lx | other`

Enabled accelerators flatten into BM25 tokens:  
`accel:dflash accel:turboquant draft:Qwen3.6-35B-A3B-DFlash`.

---

## L1 search fields (hybrid contract)

From qwen-bm25-hybrid: **separate** keyword vs semantic targets.

### `search_text` → BM25 only

Keyword concatenation (order stable, lowercase):

```
{artifact.name}
{hf_id}
{quant.label} {quant.raw} {quant.aliases}
{engine} {backend}
{accel types and draft model basenames}
{roles_fit}
{node names}
{params_total} {params_active} moe|dense vlm|text
t/s tokens: {rounded agg_tps values} tok/s
recipe:{recipe_id}
```

Example:

```
qwen3.6-35b-a3b-4bit qwen/qwen3.6-35b-a3b 4bit mlx-4bit
mlx_lm single dflash draft:qwen3.6-35b-a3b-dflash
task active node-N 35b 3b moe vlm
45 tok/s 38 tok/s recipe:qwen36-35b-a3b-dflash-1node
```

### `search_narrative` → embedding (optional until VL server up)

Prose for intent:

> Qwen3.6 35B-A3B MoE 4-bit on a single M-series node via mlx_lm with DFlash speculative decoding. Best for local agent task subagents needing tool-calling with higher decode rate than baseline 4-bit. Measured ~45 tok/s single-stream @256 gen tokens on node-N. Not for multi-node JACCL; draft model required.

### RRF fusion (mandatory)

```
BM25(search_text)  ── top_k=50 ──┐
                                  ├── RRF(k=60) → ranked recipes
(optional) Embed(search_narrative) top_k=50 ──┘
```

**Quant-heavy queries** (“mxfp4 minimax jaccl batch8”) ride BM25.  
**Goal queries** (“fast local coding with dflash”) need narrative/embed when available; until embed is up, narrative is still stored and can be grepped / used for display.

### Benchmark-aware ranking (post-RRF boost, not a third index)

After RRF, optional deterministic boost:

```
score' = rrf_score
       + w_tps * log1p(best_agg_tps)
       + w_role * role_match
       - w_issue * known_issues_count
       - w_rdma * (rdma_required && !mesh_ready)
```

Do **not** put raw floats only in BM25 and hope — also store structured `benchmarks[]` for filters:

- `min_tps=40`
- `accelerator=dflash`
- `quant=4bit`
- `backend=single`
- `node=node-N`
- `role=smol`

---

## Serve-time standard object: `ServeRequest`

Every load path (omp-go, r1o web `guardedServeLoad`, Hermes, model-kb MCP) resolves to this and **only then** calls asmi.

```typescript
interface ServeRequest {
  recipe_id: string;              // required if from KB
  model_path: string;             // absolute path or HF id asmi accepts
  engine: "mlx_lm" | "mlx_vlm" | "vllm_mlx" | "dflash" | "dflash-mlx" | "mtplx" | "ds4" | "glmx";
  backend: "single" | "jaccl";    // never "auto" for local aux
  port: number;                   // 19080 mlx_lm, 19084 mlx_vlm, …
  accelerators: Accelerator[];    // from recipe; asmi maps what it understands
  asmi_body: {                    // exact POST /serve/load body
    model_path: string;
    backend: "single" | "jaccl";
    engine?: string;
    draft_model?: string;
    num_draft_tokens?: number;
    decode_concurrency?: number;
    prefill_step_size?: number;
    prompt_cache_size?: number;
    cache_type?: string;          // turboquant / kv quant if supported
    hostfile?: string;            // jaccl only, explicit path
    use_mtp?: boolean;
    max_tokens?: number;
  };
  sampling?: Record<string, number>;
  expected_metrics?: {            // from recipe benchmarks — for post-load verify
    min_agg_tps?: number;
    max_ttft_ms?: number;
  };
  source: "model-kb" | "serve-configs" | "manual";
}
```

### Mapping recipe → asmi (rules)

| Recipe field | asmi `/serve/load` |
|--------------|-------------------|
| `serve.engine` | `engine` (guard via required_engine) |
| `serve.backend` | `backend` — **force `single` unless RDMA recipe + mesh ready** |
| `serve.port` | query `?port=` |
| `accelerators[dflash].draft_model` | `draft_model` + engine dflash family |
| `accelerators[turboquant].params` | `cache_type` / engine-specific params |
| `accelerators[jaccl]` | `backend: jaccl` + hostfile from launch template |
| `launch.curl` | fallback only if asmi body incomplete |

**Hard rules**

1. No `backend: auto` for inventory-driven serves.  
2. `required_engine` from arch wins over recipe if mismatch (deploy-engine-preflight).  
3. Post-load: `GET /serve/status` + optional completion probe; compare to `expected_metrics` when present.  
4. Live serving state is **never** the KB — always join asmi at query time.

---

## Ingest pipeline (L0 build)

```
┌─────────────────────┐
│ asmi /models (nodes)│── artifact rows
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│ serve-configs.json  │── recipe rows (primary enrichment)
│   accelerators[]    │
│   benchmark_results │
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│ model_benchmarks    │── extra t/s points (inventory)
│ model_fit_scores    │── node fit
└─────────┬───────────┘
          │
          ▼
   normalize quant + accel
   build search_text / search_narrative
   write records.jsonl → BM25 index (+ vectors later)
```

**Join key:** fuzzy  
`serve-config.model.name|hf_id|path basename` ↔ asmi `name|path`.  
Unjoined recipes still index (path may be on another node).  
Unjoined asmi artifacts get **stub recipes** (`engine` from arch, `quant` from config.json, no benchmarks).

---

## Query patterns (what RRF must answer)

| Query | BM25 hits |
|-------|-----------|
| `4bit dflash qwen 35b` | quant + accel + name |
| `mxfp4 minimax jaccl 2node` | quant + model + backend + nodes |
| `fastest local single stream` | needs narrative + t/s boost |
| `smol omp titles` | `roles_fit:smol` + small size_gb |
| `turboquant long context` | accel + hardware.max_context |
| `draft for qwen3.6-35b` | draft_model field tokens |

Filters (structured, not BM25):

```
quant=4bit accel=dflash backend=single min_tps=30 role=task node=node-N
```

---

## Storage layout (`~/infra/model-kb`)

```
~/infra/model-kb/
  records.jsonl          # L0+L1 recipes
  bm25_index/            # or in-process rebuild
  vectors.npy            # optional
  model-kb.db            # FTS5 mirror for MCP (cluster-kb pattern)
  model-kb-mcp.py
  ingest.py              # asmi + serve-configs + inventory → records
  schema.md → symlink to ~/.r1o/model-kb-schema.md
```

Graph name if docs ever go FalkorDB: **`MODEL_SERVING_DOCS`** only — never `WIKI_ATLAS`.

---

## Acceptance tests

1. Ingest 31 serve-configs + node-N asmi models → N recipe records.  
2. BM25 `dflash 4bit` returns dflash recipes before baseline.  
3. BM25 `mxfp4 jaccl` returns minimax multi-node profiles.  
4. Filter `min_tps=35` drops weak benchmarks.  
5. `recipe_to_serve_request(id)` produces asmi-valid body; `backend` never `auto`.  
6. Live `model_serving()` ignores stale KB, hits asmi.  
7. Load via recipe for currently free port succeeds; engine guard rejects ds4 under mlx_lm.

---

## Phased delivery

| Phase | Deliverable |
|-------|-------------|
| **P0** | This schema + ingest from `serve-configs.json` + asmi `/models` → `records.jsonl` + BM25 |
| **P1** | MCP: `model_search`, `model_get_recipe`, `model_serving` (asmi), `model_load(recipe_id)` |
| **P2** | Wire omp `smol`/`task` selection: search → recipe → asmi load → `r1o/active` |
| **P3** | VL embed + rerank when `:19091/:19092` up |
| **P4** | Continuous ingest from new benchmarks / serve-config saves |

---

## Multimodal (summary — full algorithm in ARDD plan §3)

- Multi-label `modalities` object, **not** a single `is_vlm` bool.  
- Evidence stack: `config.json` vision/audio keys (high) → asmi `is_vlm` (corroborate) → arch class → name hints (low) → wiki.  
- Derive `requires_vlm_engine` vs `text_only_ok_on_mlx_lm` for serve routing.  
- DFlash drafts: text-only, `roles_fit` never includes task/default/smol.

## Research enrichments (summary — ARDD plan §2 S3)

serve-configs are **not** the full recipe universe. Also ingest:

- `~/wiki/atlas/concepts/research/inference/*`  
- `~/wiki/atlas/services/*-serve-reference.md`  
- `~/wiki/atlas/techniques/mlx-vlm-*`  
- graph-indexer `optimal_params`  
- optional plans under `~/.claude/docs/plans/*dflash*|*serve*|*tq*`

Research → `kind=research_note` enrichments joined via `applies_to` globs; unjoined notes still BM25-searchable.

## Durable updates (summary — ARDD plan §4)

- Sources append/versioned; index rebuildable (`ingest` → `build-index` → `verify`).  
- `content_hash` skip unchanged rows.  
- `corrections.jsonl` for human overrides.  
- Benchmarks append-only; live serving always from asmi.  
- `model-kb status` reports per-source lag.

## Non-goals

- Replacing asmi  
- Storing live serve state only in SQLite  
- Full wiki kb-builder fiber graph as model registry  
- Semantic-only search without BM25 for quant/accel tokens  
- Treating asmi `is_vlm` as sole multimodal truth


## HF family tree + Unsloth/card measurements (2026-07-13)

Ingest:
- `ingest_hf_family.py` — Hub `GET /api/models?filter=base_model:{id}` → `kind=hf_family` levels
- `ingest_hf_cards.py` — Unsloth/HF README tables + quant inventory → `kind=hf_card`

Joined onto recipes as:
- `hf_family`: `{root_base, member_count, levels, recommend_serve}`
- `community_quality`: metric→score map from card tables (MMLU-Pro, SWE-bench, …)

Local tok/s remain in `benchmarks[]` (`source: serve-configs`). Community quality ranks *which* quant/family; local benches rank *how* we run it.

Pipeline: `pipeline.py` runs research → hf_cards → hf_family → merge → join → BM25.


## Optimization transfer (2026-07-13)

Not whole-family recipe cloning. When an accelerator is **proven** on recipe A, codify whether it applies to B:

| Relation | Example |
|----------|---------|
| same_artifact | exact same HF id |
| same_base_quant | base 4bit → same base 8bit/mxfp4 |
| same_base_finetune | base → abliterated/SFT (often **degraded** acceptance) |
| same_family_sibling | HF `base_model:` tree sibling |
| same_arch_class | weak; family token required (no bare MoE↔MoE) |
| cross_family | deny for draft/MTP heads |

Rules: `optimization_transfer_rules.json`  
Engine: `optimization_transfer.py`  
Batch: `enrich_optimization_transfer.py` → `recipe.optimization_transfer.transfers[]`  
MCP: `model_optimization_transfer`

Example: Qwen MTP on base → same-base quant **likely ×0.88**; finetune **degraded ×0.55** (re-bench). DFlash draft **deny** across arch_class.
