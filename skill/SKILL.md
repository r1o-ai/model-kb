---
name: model-kb-ingest
description: "Ingest and refresh the model knowledge base at ~/infra/model-kb — serve recipes, HF cards, hf_family trees, wiki research notes, cloud model contexts. Use when: 'ingest into model-kb', 'refresh the model DB', 'add a recipe to the KB', 'rebuild the BM25 index', 'model-kb is stale', 'the advisor can't find my model', 'benchmarks missing from the KB'."
argument-hint: recipes|research|hf|family|cloud|vendor|index|all|verify [--dry-run]
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# model-kb-ingest — Model Knowledge Base Ingest

<overview>
`~/infra/model-kb/records.jsonl` is the corpus behind the `model-kb` MCP server AND the
r1o Benchmarks optimization advisor (`/api/benchmarks/chat` → `src/lib/model-kb/`).
This skill refreshes it safely.

It is a DERIVED store. Every record traces back to an upstream source of truth:

| Record kind | Count* | Upstream SoT | Ingest script |
|---|---|---|---|
| `serve_recipe` | 43 | `~/.r1o/serve-configs.json` | `ingest_serve_configs.py` |
| `cloud_context` | 245 | CLIProxy / provider APIs | `ingest_cloud_context.py` |
| `research_note` | 73 | `~/wiki/atlas/**` | `ingest_research.py` |
| `hf_family` | 19 | HF Hub `base_model` tree | `ingest_hf_family.py` |
| `hf_card` | 5 | HF model cards | `ingest_hf_cards.py` |

\* as of 2026-07-26 — re-count with `verify`, never assume.

NEVER hand-write records into `records.jsonl`. Fix the upstream SoT and re-ingest.
A hand-edited record is silently destroyed by the next `recipes` run.
</overview>

## ⛔ THE ORDERING TRAP — read before running anything

`ingest_serve_configs.py` **CLOBBERS** `records.jsonl` (`open("w")`, recipes only).
`ingest_cloud_context.py` and `ingest_vendor_research.py` **MERGE** (read → merge → write).

So running `recipes` on a populated corpus destroys all 245 `cloud_context` +
73 `research_note` + `hf_family` records — 318 of 385 — with no warning.

**Mandatory order when doing a full rebuild:**

```
1. recipes   (clobbers → recipes only)
2. cloud     (merges cloud_context back in)
3. research  (writes enrichments.jsonl, then joins)
4. family    (enrichments_hf_family.jsonl → join)
5. hf        (enrichments_hf.jsonl → join)
6. index     (rebuild BM25)
```

**ALWAYS back up first** (step 0 below). If you only need one kind refreshed,
prefer the merge-safe scripts and avoid `recipes` entirely.

## Step 0 — ALWAYS: back up + baseline count

```bash
cd ~/infra/model-kb
cp records.jsonl "records.jsonl.bak-$(date +%Y%m%d-%H%M%S)"
python3 -c "
import json,collections
c=collections.Counter()
for l in open('records.jsonl'):
    c[json.loads(l).get('kind')]+=1
print('BASELINE:', dict(c), 'total', sum(c.values()))
"
```

Record the baseline numbers. After ingest, **no kind may drop to zero** unless
that was the explicit goal. A drop is the ordering trap firing.

## Modes

### `recipes` — serve recipes from serve-configs.json
The advisor's measured data (benchmarks, accelerators, known_issues) comes from here.

```bash
cd ~/infra/model-kb
python3 ingest_serve_configs.py            # --config / --out to override
```
⛔ CLOBBERS. Must be followed by `cloud` (and any other kinds you need) + `index`.

Fix data problems in `~/.r1o/serve-configs.json` (use the `/serve-config` skill), not here.

### `cloud` — cloud model contexts (merge-safe)
```bash
python3 ingest_cloud_context.py --dry-run          # inspect first
python3 ingest_cloud_context.py --rebuild-index
```
Merge semantics: drops existing `cloud_context` rows, re-adds fresh ones, keeps
everything else. Safe to run alone.

### `research` — wiki notes → enrichments → join
```bash
python3 ingest_research.py --json                  # → enrichments.jsonl
python3 join_enrichments.py                        # → folds into records.jsonl
```

⚠️ **`join_enrichments.py` DROPS standalone `hf_family` + `hf_card` rows** (by design —
`join_enrichments.py:268` preserves every kind EXCEPT `serve_recipe`, `research_note`,
`hf_family`, `hf_card`, folding the latter two into recipe fields). So a bare
research join takes the corpus 385 → 377 and empties `model_family()`'s hf_id-only
fallback path. **Always merge all three enrichment files before joining:**

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from pathlib import Path
from ingest_hf_cards import merge_enrichment_files
R=Path('.')
print(merge_enrichment_files(
  [R/'enrichments.jsonl', R/'enrichments_hf_family.jsonl', R/'enrichments_hf.jsonl'],
  R/'enrichments.merged.jsonl'))
"
mv enrichments.merged.jsonl enrichments.jsonl
python3 join_enrichments.py
```
Observed 2026-07-26: research-only join → `attach_notes=57 recipes_enriched=21/43`;
after merging all three → `attach_notes=84 recipes_enriched=41/43`, 404 records.

Other gotchas:
- **`research-corpus.glob` gates everything.** A note not matched by a pattern is
  never ingested. Verify before assuming a note landed:
  ```bash
  python3 -c "
  import glob as g; from pathlib import Path
  H=Path.home(); hits=[]
  for l in open('research-corpus.glob'):
      l=l.strip()
      if not l or l.startswith('#') or l.endswith('.py'): continue
      hits += [Path(m) for m in g.glob(str(H/l)) if Path(m).suffix=='.md']
  print(len(set(hits)),'files'); print(sorted({p.stem for p in set(hits)})[:5])"
  ```
- **If your notes directory exists on more than one machine, the copies are NOT
  automatically shared.** model-kb ingest reads the LOCAL copy, while a graph/search
  layer may index a different host's. Sync them before ingesting, or newly written
  notes stay invisible to search — with no error to tell you.
- **`measured` vs `pointer` is decided by table shape.** `extract_table_claims()`
  emits `confidence: measured` only when a GFM table header matches
  `tok/s`/`tps`/`throughput`/`speed` (excluding `prefill`/`accept` cols) AND the cell
  parses numerically; col 1 becomes the config label, `NNk` in a heading ≤8 lines
  above becomes `context_len`, an `accept` col becomes `acceptance_pct`, max 12 claims.
  Categorical tables ingest as `pointer` with empty `claims[]` — correct when the note
  has no measurements of its own. Write numeric tables when it does.
- `_infer_applies_to()` derives `accel_types`/`engines`/`model_globs` by substring
  over filename + first 8000 chars. Name the accelerator and model family in the body
  or the advisor's `research_notes` tool won't match it.

### `family` — HF base_model family tree
```bash
python3 ingest_hf_family.py --limit 0 --family-limit 60
```
Populates `hf_family.levels` (base / mlx / gguf_unsloth / draft_dflash / …) and
`recommend_serve`. Network-bound; use `--limit` while testing.

### `hf` — HF model cards + community quality
```bash
python3 ingest_hf_cards.py --limit 5 --sleep 0.35        # test
python3 ingest_hf_cards.py --join                        # merge + join + BM25
```
`--join` merges into `enrichments.jsonl`, runs `join_enrichments.py`, and rebuilds
the index in one shot. Rate-limited — keep `--sleep`.

### `vendor` — vendor research packs (merge-safe)
```bash
python3 ingest_vendor_research.py --vendor <slug> --dry-run
python3 ingest_vendor_research.py --vendor <slug> --rebuild-index
```

### `index` — rebuild BM25
```bash
python3 build_bm25.py --rebuild
python3 build_bm25.py --query "dflash speculative decoding" --top-k 5   # smoke test
```
Required after ANY records change. The index is a Python pickle
(`bm25_index/index.pkl`) — the r1o web advisor cannot read it and builds its own
BM25 over each record's `search_text`, so a stale pickle only breaks the MCP
server, not the web panel. Rebuild anyway to keep the two consistent.

### `all` — full rebuild, correct order
```bash
cd ~/infra/model-kb
cp records.jsonl "records.jsonl.bak-$(date +%Y%m%d-%H%M%S)"
python3 ingest_serve_configs.py \
  && python3 ingest_cloud_context.py \
  && python3 ingest_research.py \
  && python3 ingest_hf_family.py \
  && python3 ingest_hf_cards.py --join \
  && python3 build_bm25.py --rebuild
```
`ingest_hf_cards.py --join` does the merge-all-three + join + BM25 in one step, which is
why it comes last. If you skip `hf` (rate-limited), do the manual merge from the
`research` section instead — never run a bare `join_enrichments.py` as the final step.

Then run `verify`. If any kind is missing, restore the backup and re-run in order.

### `verify` — post-ingest health check (ALWAYS run this)
```bash
cd ~/infra/model-kb && python3 -c "
import json, collections
recs=[json.loads(l) for l in open('records.jsonl')]
c=collections.Counter(r.get('kind') for r in recs)
print('kinds:', dict(c), 'total', len(recs))
rc=[r for r in recs if r.get('kind')=='serve_recipe']
print(f'serve_recipes: {len(rc)}')
print(f'  with benchmarks[]:  {len([r for r in rc if r.get(\"benchmarks\")])}')
print(f'  with known_issues:  {len([r for r in rc if r.get(\"known_issues\")])}')
print(f'  with hf_family:     {len([r for r in rc if r.get(\"hf_family\")])}')
print(f'  explicit model path:{len([r for r in rc if (r.get(\"artifact\") or {}).get(\"path\")])}')
# engines must exist in the r1o engine registry (src/lib/engine-config.ts)
KNOWN={'mlx_lm','mlx_vlm','vllm_mlx','dflash','ds4','glmx','mtplx','mlx_turbo','llama_cpp','omlx','janq'}
bad=[(r['recipe_id'],(r.get('serve') or {}).get('engine')) for r in rc
     if (r.get('serve') or {}).get('engine') not in KNOWN]
print('UNKNOWN ENGINES (advisor cannot build a command):', bad or 'none')
missing=[r['recipe_id'] for r in rc if not (r.get('serve') or {}).get('engine')]
print('ENGINE-LESS recipes:', missing or 'none')
"
```

Then confirm the web advisor still retrieves:
```bash
cd ~/Projects/r1o/web && npx vitest run src/lib/model-kb/
```

## Health rules

<evaluation-criteria>
1. **Backed up?** `records.jsonl.bak-*` created before any write mode. (REQUIRED)
2. **Order respected?** `recipes` never run last or alone on a populated corpus. (REQUIRED)
3. **No kind zeroed?** Post-ingest counts compared against the baseline. (REQUIRED)
4. **Index rebuilt?** `build_bm25.py --rebuild` after any records change. (REQUIRED)
5. **Engines valid?** Every recipe engine exists in `src/lib/engine-config.ts` `ENGINES`. (REQUIRED)
6. **Upstream fixed, not the corpus?** No hand-edited records.jsonl rows. (REQUIRED)
</evaluation-criteria>

## Known data problems (2026-07-26)

Found by the r1o advisor's `serve_command` materializer across all 43 recipes:

| Problem | Recipes | Fix |
|---|---|---|
| `engine: "dflash-mlx"` not in the engine registry (registry has `dflash`) | `kimi-k26-dflash-1node`, `kimi-k26-dflash-4node` | Rename the engine in `~/.r1o/serve-configs.json`, re-ingest `recipes` |
| No explicit `artifact.path` — falls back to `~/Models/<name>` | 31 of 43 | Add `model.path` to the serve-config where the real path differs |
| `benchmarks[]` empty | 30 of 43 | Run the benchmark on the Benchmarks page, save the profile, re-ingest |
| `optimization_transfer` never precomputed | 43 of 43 | Expected — the advisor derives transfer live |

An unknown engine means the advisor refuses to build a serve command (by design —
guessing `mlx_lm` for a GGUF model dies on a bare mmap EINVAL). Fix upstream.

## Consumers — what breaks if you get this wrong

| Consumer | Path | Breakage |
|---|---|---|
| model-kb MCP | `~/infra/model-kb/model-kb-mcp.py` | Stale pickle → wrong search hits |
| r1o advisor | `web/src/lib/model-kb/records.ts` | Reads JSONL directly; fails CLOSED with `ModelKbUnavailableError` if absent |
| Benchmarks panel | `web/src/components/benchmarks/OptimizationAdvisorPanel.tsx` | Reports "knowledge base unavailable" rather than guessing |

The web side re-reads on mtime change — no restart needed after ingest.

<final-rules>
- NEVER run `ingest_serve_configs.py` on a populated corpus without following it with `cloud` (+ the other kinds) and comparing counts to the baseline.
- NEVER hand-edit `records.jsonl` — fix the upstream SoT and re-ingest.
- ALWAYS back up before a write mode, and ALWAYS run `verify` after.
- ALWAYS rebuild the BM25 index after a records change.
- If a recipe's engine is not in the r1o engine registry, report it — do not invent a mapping.
</final-rules>
