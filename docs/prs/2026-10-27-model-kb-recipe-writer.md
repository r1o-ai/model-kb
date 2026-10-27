## W1.0: one recipe writer for model-kb (profile-cascade, contract E-Q2/E-Q4, R-Q1)

**Stacked on `pr/2026-10-25-embed-discovery-fail-closed`.** Merge that branch first.

### What changed
- `b7ca5c9`: this repo had fallen behind the hub install (`~/.r1o/model-kb` on the node whose crontab runs `pipeline.py`). I compared hashes read-only over ssh and synced 9 files that differed: export_obsidian, ingest_cloud_context, ingest_hf_cards, ingest_hf_family, ingest_research, ingest_serve_configs, join_enrichments, model-kb-mcp, model_kb. I also added 3 files hub uses that were missing here: export_seed, fill_recipe_paths, recipe_gap.
- `engine/recipe_store.py` (new) is the only writer of hand-made recipe data:
  - It keeps one record per (model, engine, source), with `source` ∈ research | benchmark | user and a `history[]` of patches.
  - Each write is temp + `os.replace` under the shared `.ingest.lock`. If the lock is held, the CLI exits 75 ("retry") and the MCP tool returns `retry: true`.
  - Only the node whose crontab runs this install's `pipeline.py` may write. The check reads the crontab, so no hostname is hardcoded. On any other node the write is refused with exit 77.
- `recipes.snapshot.json` (new, GENERATED, 0444) is built only from the store and is byte-stable for the same input. It is re-exported on every put and every fold. The chat path reads this file, so it keeps working when model-kb is down.
- The pipeline now folds the store into every generation, after the last step that rewrites rows. The store comes back as `kind=authored_recipe` rows, so it survives the recipes clobber.
- The fold computes `servable`, `servable_on: [nodes]` and `servable_reason` for every recipe row from `lib/engines.py` `live_engine_ids()`. That is the union of `/serve/engines` over the online nodes asmi lists, reached by tailnet address, with no host literals. If the union is empty, the fold aborts with exit 4 and the previous generation is restored.
- `verify.py` no longer reads r1o `web/src/lib/engine-config.ts` (store F). It fails only on recipes with no engine or no fold. It lists recipes it cannot serve but does not fail on them.
- `lib/ingest_lock.py` is now the single lock implementation. `model-kb-mcp.py` gains the `model_recipe_put` tool.

### Hand-made data: where it lives
- Store: `~/.r1o/model-kb/recipes.jsonl` on the writer node (hub today, from its crontab). It is not tracked in git (.gitignore). It is the only hand-made data in model-kb. `records.jsonl` and `enrichments*` are regenerated and stay untracked.
- Backup: `~/.r1o/model-kb/backups/recipes/recipes-<sha16>.jsonl`. Before each rewrite, the version being replaced is copied there. The copies are content-addressed, so re-runs do not grow the directory. Each record's `history[]` holds per-record undo. **There is no off-node copy yet:** a follow-up should send it to the NAS archive with the W3 archive rule.

### Verification
`python3 -I -m unittest discover -s engine/tests`: 18 tests, OK. The tests run a real `pipeline.py` against a temp install, with a stub asmi (two online nodes, one offline), a stub embed server and a fake tailnet:
- A user record survives two full generations, and records rows stay equal to vectors rows.
- In `--research-only`, `verify.py` passes. An engine on one node gets `servable_on` = [that node]. An engine on no node gets `servable: false` with a reason.
- With asmi unreachable, the fold aborts and the previous generation stays byte-identical.
- A put while the lock is held exits 75. A put on a non-writer node exits 77.
- The snapshot is 0444, carries GENERATED and is byte-stable.

### Deploy notes (not done here; no deploys in this PR)
- Deploying to hub means copying `engine/` over `~/.r1o/model-kb` code files. The next pipeline run adds the servable fields.
- The 3 `mtplx` recipes become `servable: false` by themselves, because no asmi declares mtplx. Hub's serve-configs.json needs no edit.
- mambk's `~/.r1o/model-kb` (stale copy, older code) and its local `model-kb` gateway registration are traces for W3/W5. Non-writer nodes keep only the snapshot.

### serve-configs.json recipes on model-named or tuning-carrying engine ids (E-R1), for the conversion PR
Hub `~/.r1o/serve-configs.json` (57 configs, read 2026-10-27):
- `qwen_vlm_structured`: none. `glm53_flash_mlx`: none. `tensorfold_py`: none.
- `omlx`: `glm53-flash-4bit-mtp-omlx-mtp-1node`, `glm53-flash-4bit-mtp-omlx-dflash2-1node` (TensorFold/GLM-5.3-Flash-MLX-4bit-MTP).
- `mtplx` (known to no asmi): `qwen36-27b-mtplx-1node`, `qwen36-35b-a3b-mtplx-1node`, `qwen36-35b-a3b-mtplx-triattn-1node`.

### Follow-ups found (not in this PR)
- `ingest_derivative_audit.py:53` also reads r1o `engine-config.ts` (`--engine-ts`). It should move to `lib/engines.py`.
- `ingest_cloud_context.py:42-44` reads r1o `cloud-models.ts` and the cloud specs files. The live SoT should be CLIProxy `/v1/models`.
- `model-kb-mcp.py` `model_load` POSTs `/api/hermes/serve-configs/{name}/apply` (store G).
- Seeding `recipes.jsonl` from the 57 serve_recipe rows (source=research) and deleting `ingest_serve_configs.py`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
