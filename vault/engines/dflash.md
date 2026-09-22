---
tags: [model-kb, engine, dflash]
entity_type: research_note
status: active
confidence: pointer
applies_engines: [dflash, dflash-mlx]
applies_accels: [dflash]
---

# dflash

Speculative-decode stack (draft model + verify). Appears both as `serve.engine: dflash` / `dflash-mlx` and as an `accelerators[].type: dflash` on an mlx_lm recipe.

A recipe without a draft artifact is not a dflash recipe.

See [[engines/_index]].
