#!/usr/bin/env python3
"""Export model-kb records.jsonl to an Obsidian notebook.

Default dest is the personal wiki atlas. Pass --dest to export into this
repo's `vault/` (or anywhere). Only generated subdirs are replaced:
recipes/, cloud/, hubs/, _index.md. Authored notes (research/, templates/,
engines/, HOWTO.md) are left alone.

  python3 export_obsidian.py
  python3 export_obsidian.py --records data/records.seed.jsonl --dest ../vault
  python3 export_obsidian.py --kinds serve_recipe
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

HOME = Path.home()
# Live fleet defaults — override with flags. Product home is ~/.r1o/model-kb;
# ~/infra/model-kb is the legacy path.
_DEFAULT_RECORDS = Path(os.environ.get("MODEL_KB_RECORDS", str(HOME / ".r1o/model-kb/records.jsonl")))
if not _DEFAULT_RECORDS.exists():
    _DEFAULT_RECORDS = HOME / "infra/model-kb/records.jsonl"
_DEFAULT_DEST = Path(os.environ.get("MODEL_KB_VAULT", str(HOME / "wiki/atlas/model-kb")))

KB_RECORDS = _DEFAULT_RECORDS
ATLAS = _DEFAULT_DEST

ENGINE_LINKS = {
    "ds4": "[[ds4-serve-ops]] · [[ds4-engine]]",
    "mlx_lm": "[[mlx-lm-serve-reference]]",
    "mlx_vlm": "[[mlx-lm-serve-reference]]",
    "dflash": "[[dflash-turboquant-mlx]]",
    "dflash-mlx": "[[dflash-turboquant-mlx]]",
    "mtplx": "[[mtplx-mlx]]",
    "llama_cpp": "[[llama-cpp]]",
}
BACKEND_LINKS = {
    "jaccl": "[[jaccl-rdma]] · [[ds4-jaccl-distributed]]",
}


def slug(rec):
    """Filesystem + Obsidian-safe note name from recipe_id."""
    return (rec.get("recipe_id") or "unnamed").replace("/", "--").replace(":", "-")


def _sv(v):
    """Slug a dimension value for a hub note filename."""
    return str(v).lower().replace("/", "-").replace(":", "-").replace(" ", "-").replace(".", "-")


def modality_of(rec):
    a = rec.get("artifact") or {}
    o = rec.get("options") or {}
    return "vision" if (a.get("is_vlm") or o.get("vision")) else "text"


_DATE_SUFFIX = re.compile(r"[-@](20\d{2})-?(\d{2})-?(\d{2})$")


def release_date_of(rec):
    """Release date derived from a dated model-id suffix (e.g. -20240307,
    -2025-01-15). Only unambiguous full dates count; otherwise None —
    never guessed from training data."""
    a = rec.get("artifact") or {}
    for cand in (a.get("model_id"), rec.get("recipe_id")):
        if not cand:
            continue
        m = _DATE_SUFFIX.search(str(cand))
        if m:
            y, mo, d = m.groups()
            if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
                return f"{y}-{mo}-{d}"
    return None


def provenance_of(rec):
    """live-verified if any source beyond the static specs snapshot."""
    su = set(rec.get("sources_used") or [])
    return "snapshot" if su and su <= {"cloud-model-specs.json"} else "live-verified"


def family_of(rec):
    """Family from the record's own data: explicit artifact.family wins;
    otherwise derive from the id shape — leading alpha token of the model id
    (claude-3-haiku-20240307 → claude, gpt-4o → gpt, Qwen3.6-35B → qwen).
    General algorithm, no family catalog."""
    a = rec.get("artifact") or {}
    fam = a.get("family")
    if fam and fam.lower() not in ("other", "local"):
        return fam.lower()
    raw = a.get("model_id") or a.get("hf_id") or a.get("name") or rec.get("recipe_id") or ""
    raw = str(raw).split("/")[-1].lower()
    m = re.match(r"([a-z]+)", raw)
    return m.group(1) if m else "other"


def lineage_of(rec):
    """Base model id with any dated suffix stripped — groups a model's
    dated releases (claude-3-haiku-20240307 → claude-3-haiku)."""
    a = rec.get("artifact") or {}
    mid = a.get("model_id")
    if not mid:
        return None
    base = _DATE_SUFFIX.sub("", str(mid))
    return base if base != mid else None


def architecture_of(rec):
    a = rec.get("artifact") or {}
    arch = a.get("architecture")
    if arch:
        # Normalize to the leading family term ("MoE+MLA+mHC" → "moe")
        return arch.split("+")[0].split(" ")[0].lower()
    if a.get("is_moe") is True:
        return "moe"
    if a.get("is_moe") is False:
        return "dense"
    return None


def context_class_of(rec):
    ctx = (rec.get("serve") or {}).get("max_context") or (rec.get("hardware") or {}).get("max_context")
    if not isinstance(ctx, (int, float)) or ctx <= 0:
        return None
    for label, floor in (("1m", 1_000_000), ("512k", 500_000), ("256k", 250_000),
                         ("200k", 190_000), ("128k", 125_000), ("64k", 60_000), ("32k", 30_000)):
        if ctx >= floor:
            return label
    return "short"


def accelerators_of(rec):
    return sorted({a.get("type") for a in rec.get("accelerators") or []
                   if a.get("type") and a.get("enabled")})


def _one(v):
    return [v] if v else []


# Relationship dimensions — each is (name, extractor -> list of values).
# Every value comes from record fields (no invented catalogs); hub notes are
# generated from what the records actually contain. Extend this list to add
# new relationship axes — nothing else needs to change.
DIMENSIONS = [
    ("family", lambda rec: [family_of(rec)] if family_of(rec) != "other" else []),
    ("provider", lambda rec: _one((rec.get("artifact") or {}).get("provider"))),
    ("modality", lambda rec: [modality_of(rec)]),
    ("lineage", lambda rec: _one(lineage_of(rec))),
    ("artifact", lambda rec: _one((rec.get("artifact") or {}).get("hf_id"))),
    ("architecture", lambda rec: _one(architecture_of(rec))),
    ("engine", lambda rec: [v for v in _one((rec.get("serve") or {}).get("engine")) if v != "cloud"]),
    ("backend", lambda rec: _one((rec.get("serve") or {}).get("backend"))),
    ("quant", lambda rec: [v for v in _one((rec.get("quant") or {}).get("label")) if v != "api"]),
    ("accelerator", accelerators_of),
    ("chip", lambda rec: [v for v in _one((rec.get("hardware") or {}).get("chip")) if v != "cloud"]),
    ("context", lambda rec: _one(context_class_of(rec))),
    ("params", lambda rec: _one((rec.get("artifact") or {}).get("params_total"))),
    ("login", lambda rec: _one((rec.get("options") or {}).get("login_label"))),
    ("role", lambda rec: list(rec.get("roles_fit") or [])),
    ("distributed", lambda rec: (["multi-node"] if ((rec.get("hardware") or {}).get("min_nodes") or 0) > 1 else [])
                                + (["rdma"] if (rec.get("hardware") or {}).get("rdma_required") else [])),
]


def relations_of(rec):
    """[(dimension, value), ...] for this record, deduped, stable order."""
    out, seen = [], set()
    for dim, fn in DIMENSIONS:
        for v in fn(rec):
            key = (dim, _sv(v))
            if key not in seen:
                seen.add(key)
                out.append((dim, v))
    return out


def hub_key(dim, value):
    return f"{dim}-{_sv(value)}"


# Hub keys with >=2 members — populated in main() before notes render, so
# cards never link to a hub note that won't exist (kb-verify checks targets).
VALID_HUBS = set()


def hub_link(dim, value):
    return f"[[model-kb/hubs/{hub_key(dim, value)}|{dim}: {value}]]"


def card_relations(rec):
    rels = [(d, v) for d, v in relations_of(rec) if hub_key(d, v) in VALID_HUBS]
    if not rels:
        return []
    return ["## Relations", "", " · ".join(hub_link(d, v) for d, v in rels), ""]


def load_records(path: Path):
    if not path.exists():
        raise SystemExit(f"records not found: {path}")
    recs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def fm(**kv):
    """Minimal YAML frontmatter (flat keys + flow lists only)."""
    out = ["---"]
    for k, v in kv.items():
        if v is None:
            continue
        if isinstance(v, list):
            items = ", ".join(f'"{x}"' if " " in str(x) else str(x) for x in v)
            out.append(f"{k}: [{items}]")
        elif isinstance(v, bool):
            out.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            out.append(f"{k}: {v}")
        else:
            s = str(v).replace('"', "'")
            out.append(f'{k}: "{s}"' if (":" in s or "#" in s) else f"{k}: {s}")
    out.append("---")
    return "\n".join(out)


def tbl(rows, headers):
    if not rows:
        return "_none_\n"
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append("| " + " | ".join(str(c) if c is not None else "—" for c in r) + " |")
    return "\n".join(lines) + "\n"


def accels(rec):
    rows = []
    for a in rec.get("accelerators") or []:
        params = a.get("params") or {}
        pstr = " ".join(f"{k}={v}" for k, v in params.items()) or "—"
        rows.append((a.get("type"), "on" if a.get("enabled") else "OFF", pstr, a.get("note") or ""))
    return rows


def recipe_note(rec):
    a = rec.get("artifact") or {}
    q = rec.get("quant") or {}
    s = rec.get("serve") or {}
    h = rec.get("hardware") or {}
    samp = rec.get("sampling") or {}
    engine, backend = s.get("engine"), s.get("backend")

    head = fm(
        tags=["model-kb", "serve-recipe", f"engine-{engine}", f"backend-{backend}"],
        aliases=[a.get("name")] if a.get("name") else None,
        entity_type="serve_recipe",
        recipe_id=rec.get("recipe_id"),
        family=family_of(rec),
        engine=engine,
        backend=backend,
        quant=q.get("label"),
        hf_id=a.get("hf_id"),
        modality=modality_of(rec),
        release_date=release_date_of(rec),
        retired=rec.get("retired"),
        provenance=provenance_of(rec),
        source="model-kb",
        created=rec.get("created"),
        status="active",
    )
    parts = [head, "", f"# {rec.get('recipe_id')}", ""]
    if rec.get("description"):
        parts += [rec["description"], ""]

    parts += ["## Artifact", ""]
    parts.append(tbl(
        [(a.get("name"), a.get("hf_id"), a.get("path"),
          f"{a.get('size_gb')} GB" if a.get("size_gb") else None,
          a.get("params_total"), a.get("params_active"), a.get("architecture"))],
        ["name", "hf_id", "path", "size", "params", "active", "arch"]))
    nodes = a.get("nodes") or []
    if nodes:
        parts.append(f"nodes: {', '.join(nodes)}\n")

    parts += ["## Serve", ""]
    parts.append(tbl(
        [(engine, backend, s.get("port"), s.get("decode_concurrency"),
          s.get("prefill_step_size"), s.get("prompt_cache_size"), s.get("draft_model"))],
        ["engine", "backend", "port", "decode_conc", "prefill_step", "prompt_cache", "draft"]))

    if h:
        parts += ["## Hardware", ""]
        parts.append(tbl(
            [(h.get("chip"), h.get("ram_gb"), h.get("min_nodes"),
              "yes" if h.get("rdma_required") else "no",
              h.get("peak_memory_gb"), h.get("max_context"), h.get("notes"))],
            ["chip", "ram_gb", "min_nodes", "rdma", "peak_mem_gb", "max_ctx", "notes"]))

    ar = accels(rec)
    if ar:
        parts += ["## Accelerators", ""]
        parts.append(tbl(ar, ["type", "state", "params", "note"]))

    if samp:
        parts += ["## Sampling", ""]
        parts.append(tbl(
            [(samp.get("temperature"), samp.get("min_p"), samp.get("top_p"),
              samp.get("top_k"), samp.get("max_tokens"), samp.get("seed"))],
            ["temperature", "min_p", "top_p", "top_k", "max_tokens", "seed"]))

    benches = rec.get("benchmarks") or []
    if benches:
        parts += ["## Benchmarks", ""]
        parts.append(tbl(
            [(b.get("id"), b.get("concurrency"), b.get("context_len"),
              b.get("agg_tps"), b.get("per_req_tps"), b.get("ttft_ms"), b.get("source"))
             for b in benches],
            ["id", "conc", "ctx_len", "agg_tps", "per_req_tps", "ttft_ms", "source"]))

    ki = rec.get("known_issues") or []
    if ki:
        parts += ["## Known Issues", ""]
        parts += [f"- {x}" for x in ki]
        parts.append("")

    vo = rec.get("verified_on") or []
    if vo:
        parts += ["## Verified On", ""]
        vrows = []
        for v in vo:
            if isinstance(v, str):
                vrows.append(("—", "—", v))
            else:
                vrows.append((v.get("date"), ", ".join(v.get("nodes") or []), v.get("note")))
        parts.append(tbl(vrows, ["date", "nodes", "note"]))

    su = rec.get("sources_used") or []
    if su:
        parts.append(f"sources: {', '.join(su)}\n")

    parts += card_relations(rec)

    related = []
    if engine in ENGINE_LINKS:
        related.append(ENGINE_LINKS[engine])
    if backend in BACKEND_LINKS:
        related.append(BACKEND_LINKS[backend])
    if related:
        parts += ["## See Also", "", " · ".join(related), ""]

    if rec.get("search_narrative"):
        parts += ["---", "", rec["search_narrative"], ""]
    return "\n".join(parts)


def cloud_note(rec):
    a = rec.get("artifact") or {}
    s = rec.get("serve") or {}
    o = rec.get("options") or {}
    quota = rec.get("quota") or {}

    head = fm(
        tags=["model-kb", "cloud-context", f"provider-{a.get('provider')}"],
        aliases=[a.get("name")] if a.get("name") else None,
        entity_type="cloud_context",
        recipe_id=rec.get("recipe_id"),
        provider=a.get("provider"),
        model_id=a.get("model_id"),
        family=family_of(rec),
        modality=modality_of(rec),
        release_date=release_date_of(rec),
        retired=rec.get("retired"),
        provenance=provenance_of(rec),
        source="model-kb",
        created=rec.get("created"),
        status="active",
    )
    parts = [head, "", f"# {a.get('name') or rec.get('recipe_id')}", ""]
    if rec.get("description"):
        parts += [rec["description"], ""]
    parts.append(tbl(
        [(a.get("provider"), a.get("model_id"), a.get("family"),
          s.get("api_surface"), s.get("max_context"), s.get("max_output_tokens"),
          "yes" if o.get("vision") else "no",
          "yes" if o.get("thinking") else ("—" if o.get("thinking") is None else "no"))],
        ["provider", "model_id", "family", "api", "max_ctx", "max_out", "vision", "thinking"]))
    if quota:
        parts += ["## Quota", ""]
        parts.append(tbl(
            [(quota.get("display"), quota.get("kind"), quota.get("detail"), quota.get("source"))],
            ["display", "kind", "detail", "source"]))
    rf = rec.get("roles_fit") or []
    if rf:
        parts.append(f"roles: {', '.join(rf)}\n")
    su = rec.get("sources_used") or []
    if su:
        parts.append(f"sources: {', '.join(su)}\n")
    parts += card_relations(rec)
    return "\n".join(parts)


def hub_note(dim, value, members):
    """A hub note per (dimension, value) — the shared node that connects
    every model in that group via wiki-ingest RELATES_TO edges."""
    head = fm(
        tags=["model-kb", "hub", f"dim-{dim}"],
        entity_type="model_hub",
        dimension=dim,
        value=str(value),
        member_count=len(members),
        source="model-kb",
        status="active",
    )
    parts = [head, "", f"# {dim}: {value}", "",
             f"{len(members)} model-kb records share `{dim} = {value}`.", ""]
    rows = []
    for rec in sorted(members, key=lambda r: r.get("recipe_id") or ""):
        a = rec.get("artifact") or {}
        sub = "recipes" if rec.get("kind") == "serve_recipe" else "cloud"
        rows.append((f"[[model-kb/{sub}/{slug(rec)}|{a.get('name') or rec.get('recipe_id')}]]",
                     rec.get("kind"), a.get("provider") or "local", modality_of(rec)))
    parts.append(tbl(rows, ["record", "kind", "provider", "modality"]))
    return "\n".join(parts)


def build_index(recipes, clouds):
    parts = [
        fm(tags=["model-kb", "index"], entity_type="index", source="model-kb", status="active"),
        "",
        "# model-kb — Obsidian Export",
        "",
        f"{len(recipes)} serve recipes · {len(clouds)} cloud contexts. "
        "Generated from `records.jsonl`. Authored research lives in `research/`. "
        "Regenerate: `python3 engine/export_obsidian.py --dest vault`.",
        "",
        "## Serve Recipes",
        "",
    ]
    fams = {}
    for r in sorted(recipes, key=lambda r: r.get("recipe_id") or ""):
        fams.setdefault(family_of(r), []).append(r)
    for fam in sorted(fams):
        parts.append(f"### {fam}")
        parts.append("")
        rows = []
        for r in fams[fam]:
            s, q, h = r.get("serve") or {}, r.get("quant") or {}, r.get("hardware") or {}
            rid = r.get("recipe_id")
            rows.append((f"[[model-kb/recipes/{slug(r)}|{rid}]]", s.get("engine"), s.get("backend"),
                         q.get("label"), s.get("port"), h.get("max_context")))
        parts.append(tbl(rows, ["recipe", "engine", "backend", "quant", "port", "max_ctx"]))
    parts += ["## Cloud Contexts", ""]
    provs = {}
    for r in sorted(clouds, key=lambda r: r.get("recipe_id") or ""):
        provs.setdefault(((r.get("artifact") or {}).get("provider")) or "other", []).append(r)
    for prov in sorted(provs):
        parts.append(f"### {prov}")
        parts.append("")
        rows = []
        for r in provs[prov]:
            a, s = r.get("artifact") or {}, r.get("serve") or {}
            rid = r.get("recipe_id")
            rows.append((f"[[model-kb/cloud/{slug(r)}|{a.get('name') or rid}]]",
                         a.get("model_id"), s.get("max_context"), s.get("max_output_tokens")))
        parts.append(tbl(rows, ["name", "model_id", "max_ctx", "max_out"]))
    parts += ["## See Also", "", "[[stubs-catalog-index]] · [[serve-optimization-architecture]] · [[asmi-serve]]", ""]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="export_obsidian")
    ap.add_argument("--records", type=Path, default=_DEFAULT_RECORDS,
                    help="records.jsonl (env MODEL_KB_RECORDS, else ~/.r1o then ~/infra)")
    ap.add_argument("--dest", type=Path, default=_DEFAULT_DEST,
                    help="Obsidian folder (env MODEL_KB_VAULT, else ~/wiki/atlas/model-kb)")
    ap.add_argument("--kinds", default="serve_recipe,cloud_context",
                    help="comma list of kinds to export, or 'all'")
    args = ap.parse_args(argv)

    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    if "all" in kinds:
        kinds = {"serve_recipe", "cloud_context"}

    recs = load_records(args.records)
    recipes = [r for r in recs if r.get("kind") == "serve_recipe"] if "serve_recipe" in kinds else []
    clouds = [r for r in recs if r.get("kind") == "cloud_context"] if "cloud_context" in kinds else []

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    # Only wipe generated subdirs — never authored research/templates/engines.
    for sub in ("recipes", "cloud", "hubs"):
        p = dest / sub
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)

    VALID_HUBS.clear()
    hubs = {}
    exportable = recipes + clouds
    for r in exportable:
        for dim, v in relations_of(r):
            hubs.setdefault(hub_key(dim, v), (dim, v, []))[2].append(r)
    singles = 0
    for key, (dim, v, members) in sorted(hubs.items()):
        if len(members) < 2:
            singles += 1  # a hub of one connects nothing — skip
            continue
        VALID_HUBS.add(key)
        (dest / "hubs" / f"{key}.md").write_text(hub_note(dim, v, members))

    for r in recipes:
        (dest / "recipes" / f"{slug(r)}.md").write_text(recipe_note(r))
    for r in clouds:
        (dest / "cloud" / f"{slug(r)}.md").write_text(cloud_note(r))

    (dest / "_index.md").write_text(build_index(recipes, clouds))

    written = len(hubs) - singles
    print(f"exported {len(recipes)} recipes + {len(clouds)} cloud contexts "
          f"+ {written} hubs ({singles} single-member skipped) → {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
