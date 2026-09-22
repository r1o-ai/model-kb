---
tags: [model-kb, engine, mlx_lm]
entity_type: research_note
status: active
confidence: pointer
applies_engines: [mlx_lm]
---

# mlx_lm

MLX language-model server. Default engine for most Apple Silicon recipes in this corpus.

Recipe rows set `serve.engine: mlx_lm`. Topology is `serve.backend` (`single` vs `jaccl`), not a second engine name.

See [[engines/_index]].
