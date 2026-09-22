---
tags: [model-kb, howto]
entity_type: docs
status: active
---

# HOWTO — research in this vault

## Write a note

1. New file under `research/` (Obsidian "new file" already lands there).
2. Insert the `research-note` template.
3. One finding per note. Split if it is both a recipe and a technique.
4. Wikilinks are `[[filename]]` only — no path prefixes.

## What not to put here

- Cluster hostnames, LAN IPs, `/Users/…` paths, node inventories
- Operator-only serve-configs (those ingest from the live `serve-configs.json`)
- Hand-written rows of `records.jsonl` — that file is derived and the next recipes ingest will clobber it

## Frontmatter

```yaml
---
tags: [model-kb, research]
entity_type: research_note
status: active          # or disputed
confidence: measured    # measured | inferred | pointer
applies_engines: [mlx_lm]
applies_accels: [jaccl, dflash]
source: "https://… or bench-id"
---
```

`status: disputed` stays out of embeddings.

## Ingest path

Authored markdown here → `ingest_research.py` → `enrichments.jsonl` → `join_enrichments.py` → `records.jsonl` (`kind: research_note`) → BM25.

See [[research/_index]].
