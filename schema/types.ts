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
  | 'hf_card';

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

export interface ModelKbServe {
  engine?: string | null;
  backend?: string | null;
  port?: number | null;
  decode_concurrency?: number | null;
  prefill_step_size?: number | null;
  prompt_cache_size?: number | null;
  draft_model?: string | null;
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
  [key: string]: unknown;
}

export type ModelKbRecord =
  | ServeRecipeRecord
  | ResearchNoteRecord
  | CloudContextRecord
  | HfFamilyRecord
  | HfCardRecord;

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
