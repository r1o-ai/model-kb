#!/usr/bin/env python3
"""verify.py — post-ingest health gate for model-kb.

Fails closed if a kind is zeroed (join trap), engines are unknown, embeddings
are misaligned, or recipe_gap reports missing_recipe.

Usage:
  python3 verify.py
  python3 verify.py --skip-embed --skip-gaps
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R1O_ROOT = Path(os.environ.get("R1O_ROOT", Path.home() / "Projects" / "r1o"))
ENGINE_TS = R1O_ROOT / "web" / "src" / "lib" / "engine-config.ts"


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def known_engines() -> set[str]:
    """Live SoT: r1o EngineId union. Empty file → fail closed."""
    if not ENGINE_TS.exists():
        raise SystemExit(f"FAIL: engine registry missing at {ENGINE_TS}")
    text = ENGINE_TS.read_text()
    ids = set(re.findall(r"^\s+\| '([a-z][a-z0-9_]*)'", text, re.M))
    if not ids:
        raise SystemExit(f"FAIL: parsed 0 EngineId values from {ENGINE_TS}")
    return ids


def verify(
    records_path: Path,
    *,
    skip_embed: bool = False,
    skip_gaps: bool = False,
    strict_gated: bool = False,
) -> int:
    recs = load_records(records_path)
    kinds = collections.Counter(r.get("kind") for r in recs)
    print("kinds:", dict(kinds), "total", len(recs))
    required = ("serve_recipe", "cloud_context", "research_note", "hf_family", "hf_card")
    for k in required:
        n = kinds.get(k, 0)
        if n == 0:
            print(f"FAIL: kind {k} is 0 — join trap or skipped ingest")
            return 1

    recipes = [r for r in recs if r.get("kind") == "serve_recipe"]
    print(f"serve_recipes: {len(recipes)}")
    print(f"  with benchmarks[]:       {sum(1 for r in recipes if r.get('benchmarks'))}")
    print(f"  with known_issues:       {sum(1 for r in recipes if r.get('known_issues'))}")
    print(f"  with hf_family:          {sum(1 for r in recipes if r.get('hf_family'))}")
    print(f"  with community_quality:  {sum(1 for r in recipes if r.get('community_quality'))}")
    print(
        f"  explicit model path:     {sum(1 for r in recipes if (r.get('artifact') or {}).get('path'))}"
    )

    engines = known_engines()
    bad = [
        (r.get("recipe_id"), (r.get("serve") or {}).get("engine"))
        for r in recipes
        if (r.get("serve") or {}).get("engine") not in engines
    ]
    if bad:
        print("FAIL: UNKNOWN ENGINES (advisor cannot build a command):", bad)
        return 1
    missing = [r.get("recipe_id") for r in recipes if not (r.get("serve") or {}).get("engine")]
    if missing:
        print("FAIL: ENGINE-LESS recipes:", missing)
        return 1
    print("engines: ok")

    if not skip_gaps:
        sys.path.insert(0, str(ROOT))
        from recipe_gap import find_gaps

        gaps = find_gaps(recs)
        missing_recipe = [g for g in gaps if g.get("kind") == "missing_recipe"]
        missing_weights = [g for g in gaps if g.get("kind") == "missing_weights"]
        print(f"recipe_gap: missing_recipe={len(missing_recipe)} missing_weights={len(missing_weights)}")
        if missing_recipe:
            for g in missing_recipe:
                print(f"  MISSING RECIPE: {g.get('title')}  {g.get('note_id')}")
            return 1

    if not skip_embed:
        import numpy as np

        ids_path = ROOT / "bm25_index" / "vectors.ids.json"
        vec_path = ROOT / "bm25_index" / "vectors.npy"
        if not ids_path.exists() or not vec_path.exists():
            print("FAIL: embeddings missing — run embed_model_kb.py")
            return 1
        ids = json.loads(ids_path.read_text())
        vec = np.load(vec_path)
        rec_ids = [str(r.get("id") or r.get("recipe_id") or i) for i, r in enumerate(recs)]
        if not (len(recs) == len(ids) == vec.shape[0] and ids == rec_ids):
            print("FAIL: vectors misaligned", len(recs), len(ids), vec.shape)
            return 1
        print("vectors aligned", vec.shape)

    families = [r for r in recs if r.get("kind") == "hf_family"]
    derivs = [r for r in recs if r.get("kind") == "hf_derivative"]
    deriv_roots = {(r.get("measurements") or {}).get("root_base") for r in derivs}
    missing_roots = []
    for fam in families:
        root = (fam.get("measurements") or {}).get("root_base")
        if (fam.get("measurements") or {}).get("truncated"):
            print(f"FAIL: truncated hf_family {root}")
            return 1
        if root not in deriv_roots:
            missing_roots.append(root)
    if families and missing_roots:
        print("FAIL: hf_family without hf_derivative for", missing_roots[:8])
        return 1
    for deriv in derivs:
        meas = deriv.get("measurements") or {}
        if meas.get("truncated"):
            print("FAIL: truncated derivative audit", meas.get("root_base"))
            return 1
        for child in meas.get("derivatives") or []:
            if not isinstance(child, dict):
                continue
            if child.get("access") in {"unreachable", "http_error"}:
                print("FAIL: unreachable derivative stored as a row", child.get("id"))
                return 1
            comp = (child.get("components") or {}).get("status")
            if comp in {"unreachable", "http_error"}:
                print("FAIL: unreachable component census stored", child.get("id"))
                return 1
        # Strict applies to the official root every comparison depends on. Gated third-party
        # derivatives stay listed in gated_unauthenticated; they are recorded, not skipped.
        if strict_gated and meas.get("official_status") == "gated":
            print("FAIL: --strict-gated official root unread", meas.get("root_base"))
            return 1
    print("derivative audit: ok" if derivs else "derivative audit: no hf_family rows")

    print("verify OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=ROOT / "records.jsonl")
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--skip-gaps", action="store_true")
    ap.add_argument("--strict-gated", action="store_true")
    args = ap.parse_args()
    if not args.records.exists():
        print(f"FAIL: records missing at {args.records}")
        return 2
    return verify(args.records, skip_embed=args.skip_embed, skip_gaps=args.skip_gaps, strict_gated=args.strict_gated)


if __name__ == "__main__":
    raise SystemExit(main())
