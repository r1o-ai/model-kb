# model-kb kit — a knowledge base for *how to actually serve a model*

Most model metadata tells you what a model **is** (params, license, architecture).
This tells you what it **does on real hardware**: which quant, which engine, how many nodes,
what it measured, and what broke. It is the corpus behind a serving advisor — searchable by
plain questions like *"MoE that fits on 2 nodes with mxfp4"*.

```bash
./setup.sh                                        # deps + seed corpus + index
cd engine && python3 model_kb.py search "MoE that fits 2 nodes" --top-k 5
```

Ships with **456 anonymized reference records** so it is useful the moment it lands.

---

## What a record actually holds

Five kinds, one JSONL corpus. The value is that they are *joined*: a recipe knows its
family, its research notes, and its measurements.

| Kind | Seed | Answers |
|---|---:|---|
| `cloud_context` | 251 | context window, max output, options, quota per hosted model |
| `research_note` | 132 | findings from your own notes — why X beat Y, what broke |
| `serve_recipe` | 46 | **the payload**: quant + engine + node count + measured throughput |
| `hf_family` | 22 | base-model lineage — which quants derive from which root |
| `hf_card` | 5 | upstream model-card claims |

A `serve_recipe` is the interesting one:

```json
{ "id": "recipe:minimax-2node-batch8", "kind": "serve_recipe",
  "artifact": { "hf_id": "MiniMaxAI/MiniMax-M2.7", "size_gb": 113,
                "params_total": "456B", "params_active": "46B", "is_moe": true },
  "quant":    { "label": "mxfp4" },
  "serve":    { "engine": "mlx_lm", "topology": "2node" },
  "benchmarks": [...], "known_issues": [...], "verified_on": "..." }
```

`known_issues` and `verified_on` are what make it a knowledge base rather than a spec dump —
they carry the things you only learn by running the thing.

---

## ⛔ The ordering trap — read this before any ingest

**`ingest_serve_configs.py` opens `records.jsonl` with mode `"w"` and writes only
`serve_recipe` rows.** Every `cloud_context`, `research_note`, `hf_family`, and `hf_card`
record is destroyed. There is no warning, no diff, and **no non-zero exit** — the run prints
success while the corpus silently loses most of its content.

On the seed corpus that is **410 of 456 records gone**.

Every *other* ingest merges (read → merge → write). So the trap fires exactly once: on the
one script listed first, which is the one people run first.

`scripts/kb-guard.py` makes it visible **before** it happens:

```bash
python3 scripts/kb-guard.py backup engine/records.jsonl        # snapshot
python3 scripts/kb-guard.py plan   engine/records.jsonl recipes # exits 4 if destructive
```

```
⛔ DESTRUCTIVE — these records would be DELETED:
     cloud_context    251
     research_note    132
     TOTAL LOST       410  of 456
```

**Safe rebuild order** — recipes *first* (it clobbers), then the merging phases re-add:

```
1. recipes   (CLOBBERS — must be first)
2. cloud     (merge)
3. research  (merge)
4. vendor    (merge)
5. join + index
```

`pipeline.py` already encodes this order. Run it rather than calling ingests by hand.

Afterwards, always gate:
```bash
python3 scripts/kb-guard.py verify engine/records.jsonl --min 400
```
`verify` also detects the trap *after the fact* from corpus shape alone: a corpus that is
100% recipe rows almost certainly means it fired.

| Exit | Meaning |
|---|---|
| 0 | safe / healthy |
| 2 | usage error |
| 3 | corpus missing or unreadable |
| 4 | **destructive** — the planned run would delete non-recipe records |
| 5 | **regressed** — below the floor, only-recipes, or duplicate ids |

### The rule that prevents most damage
**Never hand-write into `records.jsonl`.** It is a *derived* store — every record traces to an
upstream source of truth. Fix the upstream and re-ingest. A hand-edited record is silently
destroyed by the next `recipes` run, and you will not find out until a search comes up empty.

---

## What's in the box

```
setup.sh                     installer — will not overwrite an existing corpus
data/records.seed.jsonl      456 anonymized records (hostnames → node-N)
scripts/kb-guard.py          the destructive-ingest guard  ← read this one
engine/model_kb.py           CLI: search / get / serving / load
engine/model-kb-mcp.py       MCP server — 6 tools
engine/pipeline.py           full rebuild in the SAFE order
engine/build_bm25.py         index builder
engine/ingest_*.py           six ingests, one per upstream source
engine/join_enrichments.py   fold enrichments into records
engine/optimization_transfer.py  carry tuning from one model to a similar one
engine/verify_goldens.py     retrieval regression suite (golden-queries.yaml)
skill/SKILL.md               the Claude Code skill
```

### Upstream sources — swap these for yours
The seed came from one deployment. To make it *yours*, repoint each ingest:

| Ingest | Reads |
|---|---|
| `ingest_serve_configs.py` | your serve-config JSON (recipes you have actually run) |
| `ingest_cloud_context.py` | provider APIs / a proxy's `/v1/models` |
| `ingest_research.py` | your notes directory (markdown) |
| `ingest_hf_family.py` | HF Hub `base_model` tree |
| `ingest_hf_cards.py` | HF model cards |
| `ingest_vendor_research.py` | `vendor-research/*.jsonl` (hand-curated vendor notes) |

---

## Search

```bash
cd engine
python3 model_kb.py search "MoE model that fits on 2 nodes with mxfp4" --top-k 5
python3 model_kb.py get recipe:minimax-2node-batch8
python3 model_kb.py search "..." --json          # machine-readable
```

BM25 over a `search_text` + `search_narrative` field per record. `rrf_fusion.py` is present
to fuse in a dense retriever if you add one; **out of the box this is sparse-only** — no
embedding server required, which is why it runs anywhere.

Because recipes carry `known_issues` and measurements, queries phrased as *problems*
("gets OOM at long context", "slow prefill on 1 node") often work better than model names.

### MCP server

```json
{ "mcpServers": { "model-kb": {
    "command": "python3",
    "args": ["/abs/path/engine/model-kb-mcp.py"]
}}}
```

Tools: `model_search`, `model_get`, `model_family`, `model_serving`, `model_load`,
`model_optimization_transfer`.

⚠️ `model_serving` and `model_load` talk to a **cluster control API** (asmi) that is specific
to the originating deployment. Without it they fail; search/get/family/transfer work standalone.

---

## About the seed corpus

Anonymized from a live deployment: hostnames → `node-N`, local paths → `/path/to/models/`,
LAN IPs → `10.0.0.N`. Verified zero private references. What survives is the part that
transfers — HF ids, quant labels, engines, topologies, measurements, known issues.

It is **reference knowledge, not your inventory**. `serve_recipe` rows describe hardware you
may not have. Treat them as prior art; re-run the ingests against your own sources to make the
corpus describe your machines.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Search returns nothing | index stale or corpus empty | `python3 build_bm25.py`; `kb-guard.py check` |
| Corpus lost most records | the ordering trap fired | restore a `.bak-*`, rebuild via `pipeline.py` |
| `model_load` / `model_serving` fail | need the cluster control API | expected standalone |
| Ingest wrote nothing | upstream source path is wrong | check each ingest's input path |
| Duplicate ids in verify | two ingests claiming one id | fix the id scheme upstream, re-ingest |

## Provenance
Extracted from a working private deployment. The engine is a reference implementation —
read `pipeline.py` and `kb-guard.py` before pointing it at data you cannot regenerate.
