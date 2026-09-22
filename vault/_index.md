---
tags: [model-kb, index]
entity_type: index
status: active
---

# model-kb research vault

Obsidian notebook for **how to actually serve a model**. The JSONL corpus and BM25 engine live next door (`engine/`, `schema/`). This vault is the authored research surface.

| Folder | What goes here | Ingest |
|---|---|---|
| [[research/_index\|research/]] | Findings, traps, "why X beat Y" | `ingest_research.py` |
| [[engines/_index\|engines/]] | Engine notes (mlx_lm, ds4, dflash, …) | research glob |
| `recipes/` | Generated serve-recipe notes | do not hand-edit |
| `cloud/` | Generated cloud-context notes | do not hand-edit |
| `hubs/` | Generated dimension hubs | do not hand-edit |
| [[templates/research-note\|templates/]] | New-note templates | — |

## Dual identity (never collapse)

- **Artifact** = weights on disk / HF id
- **Serve recipe** = how you run it (quant + engine + topology + measured t/s)

Retrieval returns recipes. A `research_note` is not a config.

## How to continue research

1. New finding → `templates/research-note` into `research/`.
2. Cite numbers (`source:`) or mark `status: disputed`.
3. Wikilink engines and recipe ids: `[[mlx_lm]]`, `[[minimax-2node-batch8]]`.
4. Re-ingest: `python3 engine/ingest_research.py` (live corpus) then join.

Generated recipe notes:

```bash
python3 engine/export_obsidian.py \
  --records data/records.seed.jsonl \
  --dest vault \
  --kinds serve_recipe
```

That command replaces `recipes/`, `cloud/`, `hubs/`, and this folder's generated `_index` siblings — not `research/` or `templates/`.

## UI contract

TypeScript types: `schema/types.ts`  
Prose schema: `engine/schema.md`

A UI reads `records.jsonl` (or the MCP tools). It does not parse these markdown files.
