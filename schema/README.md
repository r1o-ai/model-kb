# UI contract

A client (r1o Benchmarks, any other UI) implements **these files**, not the markdown vault.

| File | What |
|---|---|
| `types.ts` | TypeScript shapes for every `kind` in `records.jsonl` |
| `../engine/schema.md` | Canonical prose: dual identity, quant vocab, L0 record |
| `../data/records.seed.jsonl` | Anonymized reference corpus (one JSON object per line) |
| `../engine/model-kb-mcp.py` | MCP tools: `model_search`, `model_get`, `model_family`, … |

## Rules for a UI

1. Treat every field as optional. The ingest is loosely typed; missing ≠ default.
2. Search `serve_recipe` rows for "how do I run this". `research_note` is citation, not a config.
3. Do not pin a model-id or hostname list. Fetch the live corpus / MCP at decision time. Empty corpus → fail closed.
4. `optimization_transfer` on a record is often empty — infer at query time from `optimization_transfer_rules.json` plus sibling recipes.
5. Never write into `records.jsonl` from the UI. Fix the upstream (serve-config, research note, HF card) and re-ingest.
