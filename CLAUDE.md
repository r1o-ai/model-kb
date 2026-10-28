# model-kb

Knowledge base for **how to actually serve a model**. UI implements `schema/types.ts`. Research continues in `vault/` (Obsidian).

## Layout

| Path | Role |
|---|---|
| `engine/` | ingest, BM25, MCP, schema.md |
| `schema/types.ts` | UI contract |
| `data/records.seed.jsonl` | anonymized reference corpus |
| `vault/` | Obsidian notebook — **write research here** |
| `skill/SKILL.md` | Claude Code ingest skill |

Live fleet corpus (not this repo): `~/.r1o/model-kb/records.jsonl`.

## Rules

- Never hand-write `records.jsonl`. Fix upstream, re-ingest. Recipes ingest clobbers.
- Hand-made recipe data (research / benchmark / user override, keyed model × engine) goes ONLY through
  `engine/recipe_store.py put` or MCP `model_recipe_put`, on the node whose crontab runs `pipeline.py`.
  It lives in `~/.r1o/model-kb/recipes.jsonl`; the pipeline folds it into every generation.
- `~/.r1o/model-kb/recipes.snapshot.json` is GENERATED (0444) — the chat path reads it; never edit it.
- Engines come from asmi `/serve/engines` across online nodes (`lib/engines.py`); a recipe whose engine no
  node declares is `servable: false` with a reason, never a pipeline failure.
- Run `pipeline.py` rather than calling ingests out of order.
- Do not pin model ids, hosts, or node lists. Empty live source → fail closed.
- `vault/research/` is authored. `vault/recipes/` is generated (`export_obsidian.py`).
- Cite numbers or mark `status: disputed`.

## Search

```bash
cd engine && python3 model_kb.py search "MoE that fits 2 nodes" --top-k 5
```
