/**
 * model-kb record shapes (`records.jsonl`).
 *
 * The corpus is written by loosely-typed Python ingest scripts, so every field
 * here is optional-and-nullable on purpose: the types describe what MAY be
 * present, never what is guaranteed. A UI should render missing fields as
 * absent, not invent defaults.
 *
 * Canonical prose: `engine/schema.md`.
 */

export type ModelKbKind =
  | 'serve_recipe'
  | 'cloud_context'
  | 'research_note'
  | 'hf_family'
  | 'hf_card'
  | 'hf_derivative';

export interface ModelKbArtifact {
  name?: string | null;
  hf_id?: string | null;
  path?: string | null;
  nodes?: string[] | null;
  size_gb?: number | null;
  params_total?: string | null;
  params_active?: string | null;
  architecture?: string | null;
  is_vlm?: boolean | null;
  is_moe?: boolean | null;
}

export interface ModelKbQuant {
  label?: string | null;
  raw?: string | null;
  aliases?: string[] | null;
  scheme?: string | null;
  bits?: number | null;
  bpw?: number | null;
}

/**
 * A recipe's `serve` section: the serve config's own section, carried whole.
 *
 * The named fields are asmi `POST /serve/load` LoadRequest fields under asmi's
 * names (apple-smi `types.rs`, mirrored in `engine/lib/asmi_load.py`); each
 * one reaches the load unchanged. `port` goes to asmi as `?port=N`. Which flag
 * a field becomes is the engine record's (`GET /serve/engines` →
 * `fields[].request_field`). Any other key is config metadata (hardware, os,
 * nodes, …): kept for search, never sent.
 */
export interface ModelKbServe {
  /** asmi engine id. null = the config names none: the recipe is not loadable. */
  engine?: string | null;
  /** single | jaccl (model-kb never sends auto). */
  backend?: string | null;
  port?: number | null;
  hostfile?: string | null;
  draft_model?: string | null;
  num_draft_tokens?: number | null;
  decode_concurrency?: number | null;
  prompt_concurrency?: number | null;
  prefill_step_size?: number | null;
  pipeline?: boolean | null;
  prompt_cache_size?: number | null;
  prompt_cache_bytes?: number | null;
  use_mtp?: boolean | null;
  cache_type?: string | null;
  max_tokens?: number | null;
  kv_bits?: number | null;
  kv_quant_scheme?: string | null;
  vision_cache_size?: number | null;
  ctx_size?: number | null;
  max_num_seqs?: number | null;
  draft_kind?: string | null;
  draft_block_size?: number | null;
  drafter?: string | null;
  /**
   * Engine flags asmi does not model: pre-split argv appended last to the
   * engine command. asmi refuses --port, --host, --model, --metal, any -m…
   * token and any token containing `..`. Ingest drops a value that is not a
   * list of strings and names it in the record's `problems`.
   */
  extra_args?: string[] | null;
  /** `{param.KEY}` values for an engine record's tmpl rules. */
  params?: Record<string, string> | null;
  /** Process environment (asmi allowlists keys by prefix). */
  env?: Record<string, string> | null;
  hardware?: string | null;
  os?: string | null;
  [configKey: string]: unknown;
}

export interface ModelKbAccelerator {
  type?: string | null;
  enabled?: boolean | null;
  params?: Record<string, unknown> | null;
  draft_model?: string | null;
  repo?: string | null;
  note?: string | null;
  requires_custom_launcher?: boolean | null;
}

export interface ModelKbHardware {
  min_nodes?: number | null;
  chip?: string | null;
  ram_gb?: number | null;
  rdma_required?: boolean | null;
  peak_memory_gb?: number | null;
  max_context?: number | null;
  notes?: string | null;
}

/** One measured serve benchmark at a given concurrency. */
export interface ModelKbBenchmark {
  id?: string | null;
  concurrency?: number | null;
  max_tokens?: number | null;
  context_len?: number | null;
  /** Aggregate throughput across all in-flight requests (tok/s). */
  agg_tps?: number | null;
  /** Per-request throughput (tok/s) — what a single caller feels. */
  per_req_tps?: number | null;
  ttft_ms?: number | null;
  wall_s?: number | null;
  source?: string | null;
}

export interface ModelKbHfFamily {
  root_base?: string | null;
  member_count?: number | null;
  levels?: Record<string, number> | null;
  recommend_serve?: Array<Record<string, unknown>> | null;
}

export interface ModelKbVerifiedOn {
  cluster?: string | null;
  nodes?: string[] | null;
  date?: string | null;
  mlx_versions?: Record<string, string> | null;
  hostfile?: string | null;
}

interface ModelKbBase {
  id: string;
  kind: ModelKbKind;
  search_text?: string | null;
  search_narrative?: string | null;
  content_hash?: string | null;
  created?: string | null;
  sources_used?: string[] | null;
}

export interface ServeRecipeRecord extends ModelKbBase {
  kind: 'serve_recipe';
  recipe_id?: string | null;
  artifact?: ModelKbArtifact | null;
  quant?: ModelKbQuant | null;
  serve?: ModelKbServe | null;
  accelerators?: ModelKbAccelerator[] | null;
  hardware?: ModelKbHardware | null;
  benchmarks?: ModelKbBenchmark[] | null;
  roles_fit?: string[] | null;
  known_issues?: string[] | null;
  verified_on?: ModelKbVerifiedOn[] | null;
  hf_family?: ModelKbHfFamily | null;
  /** Joined summary only. Per-child rows live on the hf_derivative record. */
  hf_derivative?: RecipeDerivativeSummary | null;
  community_quality?: Record<string, unknown> | null;
  enrichment_links?: unknown;
  /** Often empty in the corpus — infer live instead of treating as SoT. */
  optimization_transfer?: Record<string, unknown> | null;
  sampling?: Record<string, unknown> | null;
  description?: string | null;
  source?: {
    serve_config_name?: string | null;
    asmi_path?: string | null;
    inventory_model_id?: string | null;
  } | null;
  /** Why the recipe cannot load as recorded, each named. Empty = loadable. */
  problems?: string[] | null;
  /** Verify-first notes (e.g. a backend recorded as single instead of auto). */
  warnings?: string[] | null;
  /** problems is empty. */
  loadable?: boolean | null;
  /** Some online node's asmi declares the engine (pipeline fold, live). */
  servable?: boolean | null;
  servable_on?: string[] | null;
  servable_reason?: string | null;
}

export interface ResearchNoteRecord extends ModelKbBase {
  kind: 'research_note';
  title?: string | null;
  path?: string | null;
  enrichment_id?: string | null;
  applies_to?: {
    model_globs?: string[] | null;
    accel_types?: string[] | null;
    engines?: string[] | null;
  } | null;
  claims?: unknown[] | null;
  confidence?: string | null;
  source?: Record<string, unknown> | null;
}

export interface CloudContextRecord extends ModelKbBase {
  kind: 'cloud_context';
  recipe_id?: string | null;
  artifact?: ModelKbArtifact | null;
  quant?: ModelKbQuant | null;
  serve?: ModelKbServe | null;
  accelerators?: ModelKbAccelerator[] | null;
  hardware?: ModelKbHardware | null;
  benchmarks?: ModelKbBenchmark[] | null;
  known_issues?: string[] | null;
  max_context?: number | null;
  max_output_tokens?: number | null;
  options?: Record<string, unknown> | null;
  quota?: Record<string, unknown> | null;
  roles_fit?: string[] | null;
  description?: string | null;
}

export interface HfFamilyRecord extends ModelKbBase {
  kind: 'hf_family';
  measurements?: {
    root_base?: string | null;
    family_levels?: Record<string, unknown> | null;
    recommend_serve?: unknown;
  } | null;
}

export interface HfCardRecord extends ModelKbBase {
  kind: 'hf_card';
  measurements?: {
    access?: string | null;
    token_source?: string | null;
    quality?: Record<string, number> | null;
    quants?: Array<Record<string, unknown>> | null;
  } | null;
  [key: string]: unknown;
}

/** One Hub child audited from its own config + safetensors index. */
export interface HfDerivativeMember {
  id?: string | null;
  relation?: string | null;
  access?: string | null;
  token_source?: string | null;
  formats?: string[] | null;
  total_bytes?: number | null;
  quant?: Record<string, unknown> | null;
  quant_parity?: { parity?: string | null; diverge_fields?: string[] | null } | null;
  components?: {
    status?: string | null;
    index?: string | null;
    key_count?: number | null;
    official_key_count?: number | null;
    dropped_groups?: string[] | null;
    shrunk_groups?: string[] | null;
  } | null;
  servability?: Record<string, unknown> | null;
  modification?: Record<string, unknown> | null;
}

export interface HfDerivativeRecord extends ModelKbBase {
  kind: 'hf_derivative';
  measurements?: {
    root_base?: string | null;
    official_status?: string | null;
    discovered?: number | null;
    truncated?: boolean | null;
    audited?: number | null;
    relation_counts?: Record<string, number> | null;
    component_drop_ids?: string[] | null;
    gated_unauthenticated?: string[] | null;
    token_source?: string | null;
    derivatives?: HfDerivativeMember[] | null;
  } | null;
}

export interface RecipeDerivativeSummary {
  root_base?: string | null;
  discovered?: number | null;
  audited?: number | null;
  truncated?: boolean | null;
  relation_counts?: Record<string, number> | null;
  component_drop_ids?: string[] | null;
  gated_unauthenticated?: string[] | null;
  official_status?: string | null;
  enrichment_id?: string | null;
}

export type ModelKbRecord =
  | ServeRecipeRecord
  | ResearchNoteRecord
  | CloudContextRecord
  | HfFamilyRecord
  | HfCardRecord
  | HfDerivativeRecord;

export interface ModelKbSearchHit {
  id: string;
  kind: ModelKbKind;
  recipe_id: string | null;
  title: string;
  narrative: string | null;
  score: number;
}

export type TransferRelation =
  | 'same_artifact'
  | 'same_base_quant'
  | 'same_base_finetune'
  | 'same_family_sibling'
  | 'same_arch_class'
  | 'cross_family';

export interface OptimizationCandidate {
  accel: string;
  relation: TransferRelation;
  proven_on_recipe: string;
  proven_on_model: string | null;
  measured_agg_tps: number | null;
  measured_at_concurrency: number | null;
  requires_custom_launcher: boolean;
  rdma_required: boolean;
  min_nodes: number | null;
  repo: string | null;
  note: string | null;
  proven_recipe_known_issues: string[];
}
