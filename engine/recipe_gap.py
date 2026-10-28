#!/usr/bin/env python3
"""recipe_gap.py — research/HF presence without a serve_recipe (MCP model_get fail).

Model-specific configs for the benchmarking agent / model-kb MCP are
serve_recipe rows. Research notes alone do NOT count.

Upstream SoT for recipes: ~/.r1o/serve-configs.json (via /serve-config skill).
NEVER hand-write records.jsonl.

Gaps split:
  missing_recipe  — weights exist on the cluster, no serve_recipe (fix via /serve-config)
  missing_weights — researched only (e.g. Muse Glimmer); do NOT invent a path/recipe

Exit 1 only for missing_recipe. missing_weights is deferred (download first).

Usage:
  python3 recipe_gap.py
  python3 recipe_gap.py --records records.jsonl --json
  python3 recipe_gap.py --require-hf-id   # only gaps with org/name repo ids
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDS = ROOT / "records.jsonl"
DEFAULT_INDEX_DB = Path.home() / "infra" / "graph-indexer" / "index.db"

# Strong local-serve family tokens that research notes track for r1o.
_FAMILY_NEEDLES = re.compile(
    r"muse-?glimmer|muse.?spark|nemotron.?3\.?5.?lightning|nemotron.?lightning|"
    r"qwen3\.?8|qwen3\.?6-?35b|qwen3\.?6-?27b|llama-?4-?scout|"
    r"mlx-community/[a-z0-9_.-]+",
    re.I,
)
_REPO_ID = re.compile(r"\b([a-z0-9_.-]+/[a-z0-9_./+-]{3,80})\b", re.I)
# Noise repos from cabling / code paths, not models
_NOISE_REPO = re.compile(
    r"^(hardware|tools|web|hub|src|buffer|distributed|mlx_lm)/|"
    r"\.(ts|sh|py|pt)$|^\d|"
    # technique notes that mention model classes, not deployable HF repos
    r"^(opt|bloom|llama|glm-class)/|opt/bloom",
    re.I,
)


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def _recipe_blobs(recipes: list[dict]) -> list[tuple[dict, str]]:
    out = []
    for r in recipes:
        blob = json.dumps(
            {
                "id": r.get("recipe_id") or r.get("id"),
                "artifact": r.get("artifact"),
                "serve": r.get("serve"),
                "search_text": r.get("search_text"),
                "quant": r.get("quant"),
            },
            default=str,
        ).lower()
        out.append((r, blob))
    return out


def _note_keys(note: dict) -> set[str]:
    keys: set[str] = set()
    st = (note.get("search_text") or "") + " " + (note.get("title") or "")
    st_l = st.lower()
    for m in _REPO_ID.findall(st_l):
        if _NOISE_REPO.search(m):
            continue
        # prefer model-ish orgs
        org = m.split("/")[0]
        if org in {
            "mlx-community",
            "meta-models",
            "meta-llama",
            "qwen",
            "nvidia",
            "google",
            "deepseek-ai",
            "moonshotai",
            "moonshot-ai",
            "minimaxai",
            "unsloth",
            "bartowski",
            "mradermacher",
        } or "muse" in m or "nemotron" in m or "qwen" in m:
            keys.add(m)
            keys.add(m.split("/")[-1])
    for g in (note.get("applies_to") or {}).get("model_globs") or []:
        cleaned = g.replace("*", "").strip()
        if len(cleaned) >= 4:
            keys.add(cleaned.lower())
    for m in _FAMILY_NEEDLES.findall(st_l):
        keys.add(m.lower().replace(" ", ""))
    return {k for k in keys if len(k) >= 4}


def _recipe_covers(blob: str, keys: set[str]) -> bool:
    for k in keys:
        tok = k.split("/")[-1].lower()
        if len(tok) < 4:
            continue
        # normalize dots/dashes for soft match
        if tok in blob or k.lower() in blob:
            return True
        soft = tok.replace("-", "").replace(".", "").replace("_", "")
        soft_blob = blob.replace("-", "").replace(".", "").replace("_", "")
        if len(soft) >= 6 and soft in soft_blob:
            return True
    return False


def _cluster_tokens(db: Path) -> set[str]:
    """Live weight names/paths from graph-indexer. Empty → fail closed (no 'on cluster' claims)."""
    if not db.exists():
        return set()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    toks: set[str] = set()
    for name, path in con.execute(
        "select name, path from entities where entity_type in ('model', 'model_instance')"
    ):
        for raw in (name, Path(path or "").name):
            t = re.sub(r"[^a-z0-9]+", "", (raw or "").lower())
            if len(t) >= 6:
                toks.add(t)
    return toks


def _same_model_token(a: str, b: str) -> bool:
    """Same artifact, not a shared family prefix.

    `qwen3vl` is contained in `qwen3vlembedding8braw`. Substring containment
    made verify.py report a missing serve recipe for a research note whose
    keys are the VL module and Qwen3-VL-8B-Instruct, and the desktop release
    died on that false positive. An exact match counts. Otherwise the shorter
    token has to be at least 12 characters and almost the whole of the longer
    one (a quant or pack suffix, not a different model).
    """
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < 12 or short not in long_:
        return False
    return len(short) / len(long_) >= 0.8


def _on_cluster(keys: set[str], cluster: set[str]) -> bool:
    if not cluster:
        return False
    for k in keys:
        tok = re.sub(r"[^a-z0-9]+", "", k.split("/")[-1].lower())
        if len(tok) < 6:
            continue
        if any(_same_model_token(tok, c) for c in cluster):
            return True
    return False


def find_gaps(
    records: list[dict],
    *,
    require_hf_id: bool = False,
    index_db: Path | None = None,
) -> list[dict]:
    recipes = [r for r in records if r.get("kind") == "serve_recipe"]
    notes = [r for r in records if r.get("kind") == "research_note"]
    blobs = _recipe_blobs(recipes)
    cluster = _cluster_tokens(index_db or DEFAULT_INDEX_DB)
    gaps: list[dict] = []
    for n in notes:
        keys = _note_keys(n)
        if not keys:
            continue
        hf_ids = [k for k in keys if "/" in k]
        if require_hf_id and not hf_ids:
            continue
        # Skip notes that are purely infrastructure (no model family needle + no hf id)
        st = ((n.get("search_text") or "") + " " + (n.get("title") or "")).lower()
        if not hf_ids and not _FAMILY_NEEDLES.search(st):
            continue
        covered = any(_recipe_covers(blob, keys) for _, blob in blobs)
        if covered:
            continue
        on_cluster = _on_cluster(keys, cluster)
        if on_cluster:
            kind = "missing_recipe"
            fix = (
                "Weights are on the cluster. Add a serve profile to "
                "~/.r1o/serve-configs.json (use /serve-config) with the live path, "
                "then run model-kb-ingest recipes in FULL ORDER "
                "(recipes→cloud→research→family→hf→index). NEVER hand-edit records.jsonl."
            )
        else:
            kind = "missing_weights"
            fix = (
                "Research-only — no matching artifact in graph-indexer model inventory. "
                "Download first (live path), then add a serve-config. "
                "Do NOT invent ~/Models/<name> or a recipe without weights."
            )
        gaps.append(
            {
                "kind": kind,
                "on_cluster": on_cluster,
                "note_id": n.get("id") or n.get("note_id"),
                "title": (n.get("title") or "")[:100],
                "path": n.get("path"),
                "hf_ids": sorted(hf_ids)[:8],
                "keys": sorted(keys)[:12],
                "globs": (n.get("applies_to") or {}).get("model_globs") or [],
                "fix": fix,
            }
        )
    return gaps


def weights_without_recipe(records: list[dict], index_db: Path) -> list[dict]:
    """Weights present in the live inventory that no serve_recipe points at.

    The note-driven pass above only sees families a note names; this pass starts
    from the disk. Advisory: embedding/TTS/diffusion weights appear too, so each
    row carries the config.json architecture for the reader to judge.
    Empty/missing inventory → [] (fail closed: no presence claims).
    """
    if not index_db.exists():
        return []
    recipes = [r for r in records if r.get("kind") == "serve_recipe"]
    norm = lambda s: re.sub(r"[^a-z0-9]+", "", (s or "").lower())
    covered: set[str] = set()
    for r in recipes:
        a = r.get("artifact") or {}
        p = Path(str(a.get("path") or ""))
        for s in (a.get("name"), p.name, p.parent.name if p.suffix else "", (a.get("hf_id") or "").split("/")[-1]):
            if norm(s):
                covered.add(norm(s))
    con = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    rows = con.execute(
        "select name, node, path, metadata from entities where entity_type = 'model'"
    ).fetchall()
    out = []
    for name, node, path, meta in rows:
        p = Path(path or "")
        # A component folder (vae/, text_encoder/) belongs to its parent model.
        folder = p.name if norm(p.name) == norm(name) else p.parent.name
        if norm(folder) in covered or norm(name) in covered:
            continue
        m = json.loads(meta or "{}")
        out.append({
            "name": name, "node": node, "path": path, "folder": folder,
            "model_type": m.get("model_type"),
            "architectures": m.get("architectures") or [],
        })
    # One row per folder (components of the same model collapse)
    seen: dict[tuple[str, str], dict] = {}
    for row in out:
        seen.setdefault((row["node"], row["folder"]), row)
    return sorted(seen.values(), key=lambda r: (r["node"], r["folder"].lower()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--require-hf-id", action="store_true")
    ap.add_argument("--index-db", type=Path, default=DEFAULT_INDEX_DB)
    args = ap.parse_args()
    if not args.records.exists():
        print(f"ERROR: records missing at {args.records}", file=sys.stderr)
        print(
            "Deploy the model-kb kit engine next to this script first "
            "(gateway servers.json points here).",
            file=sys.stderr,
        )
        return 2
    recs = _load(args.records)
    n_recipes = sum(1 for r in recs if r.get("kind") == "serve_recipe")
    n_notes = sum(1 for r in recs if r.get("kind") == "research_note")
    gaps = find_gaps(recs, require_hf_id=args.require_hf_id, index_db=args.index_db)
    missing_recipe = [g for g in gaps if g.get("kind") == "missing_recipe"]
    missing_weights = [g for g in gaps if g.get("kind") == "missing_weights"]
    orphans = weights_without_recipe(recs, args.index_db)
    if args.json:
        print(
            json.dumps(
                {
                    "records": str(args.records),
                    "serve_recipes": n_recipes,
                    "research_notes": n_notes,
                    "gaps": gaps,
                    "gap_count": len(gaps),
                    "missing_recipe": len(missing_recipe),
                    "missing_weights": len(missing_weights),
                    "weights_without_recipe": orphans,
                },
                indent=2,
            )
        )
    else:
        print(
            f"serve_recipes={n_recipes} research_notes={n_notes} "
            f"missing_recipe={len(missing_recipe)} missing_weights={len(missing_weights)}"
        )
        if orphans:
            print("\nWEIGHTS WITHOUT A RECIPE (live inventory; advisory — judge by architecture):\n")
            for o in orphans:
                arch = ", ".join(o["architectures"]) or o["model_type"] or "no config.json"
                print(f"- {o['node']}: {o['folder']}  [{arch}]  {o['path']}")
        if not gaps:
            print("OK — no research-tracked models missing a serve_recipe (heuristic).")
            return 0
        if missing_recipe:
            print(
                "\nMISSING RECIPE (weights on cluster — add serve-config, then ingest):\n"
            )
            for g in missing_recipe:
                print(f"- {g['title']}")
                print(f"  note: {g['note_id']}")
                if g["hf_ids"]:
                    print(f"  hf:   {', '.join(g['hf_ids'])}")
                print(f"  fix:  {g['fix']}")
        if missing_weights:
            print(
                "\nDEFERRED — missing weights (research-only; do not invent a recipe):\n"
            )
            for g in missing_weights:
                print(f"- {g['title']}")
                print(f"  note: {g['note_id']}")
                if g["hf_ids"]:
                    print(f"  hf:   {', '.join(g['hf_ids'])}")
                print(f"  fix:  {g['fix']}")
    return 1 if missing_recipe else 0


if __name__ == "__main__":
    raise SystemExit(main())
