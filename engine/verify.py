#!/usr/bin/env python3
"""verify.py — post-ingest health gate for model-kb.

Fails closed if a kind is zeroed (join trap), a recipe has no engine or was not
folded (no computed servability), a serve config's field did not reach its
recipe unexplained (the asmi round trip), embeddings are misaligned, or
recipe_gap reports missing_recipe. A recipe whose engine no online node declares
is NOT a failure: the fold marks it servable=false with a reason, and it is
listed here. So is a recipe that cannot load as recorded (`problems`).

Usage:
  python3 verify.py
  python3 verify.py --skip-embed --skip-gaps
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ingest_serve_configs import DEFAULT_CONFIG as DEFAULT_SERVE_CONFIGS  # noqa: E402 — the recipes' upstream
from lib.asmi_load import LOAD_REQUEST_FIELDS  # noqa: E402


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


RECIPE_KINDS = ("serve_recipe", "authored_recipe")


def unfolded_recipes(recs: list[dict]) -> list[str]:
    """Recipe rows without computed servability — the fold step did not run on them."""
    return [
        str(r.get("recipe_id") or r.get("id"))
        for r in recs
        if r.get("kind") in RECIPE_KINDS and "servable" not in r
    ]


def serve_round_trip(recipes: list[dict], configs: list[dict]) -> dict:
    """Compare each config's `serve` with its recipe's `serve`.

    dropped   — set in the config, absent from the recipe
    changed   — present in both with a different value
    unmatched — a config with no recipe
    A dropped or changed key is explained when the recipe's problems/warnings
    name `serve.<key>`; anything else is a lost knob.
    """
    by_name = {
        (r.get("source") or {}).get("serve_config_name") or r.get("recipe_id"): r
        for r in recipes
    }
    out: dict = {"dropped": [], "changed": [], "unmatched": [], "unexplained": []}
    for cfg in configs:
        name = cfg.get("name")
        rec = by_name.get(name)
        if rec is None:
            out["unmatched"].append(name)
            continue
        src = cfg.get("serve") or {}
        got = rec.get("serve") or {}
        notes = " ".join(list(rec.get("problems") or []) + list(rec.get("warnings") or []))
        for key, value in src.items():
            if value is None:
                continue
            if key not in got:
                kind = "dropped"
            elif got[key] != value:
                kind = "changed"
            else:
                continue
            row = {"recipe": name, "key": key, "config": value, "recipe_value": got.get(key)}
            out[kind].append(row)
            if f"serve.{key}" not in notes:
                out["unexplained"].append({**row, "kind": kind})
    return out


def engine_request_fields(base: str | None = None, timeout_s: float = 5.0) -> set[str]:
    """Request fields some engine turns into a flag: asmi GET /serve/engines
    → engines[].fields[].request_field. Raises RuntimeError when asmi does not answer."""
    base = (base or os.getenv("ASMI_URL", "http://localhost:9090")).rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/serve/engines", timeout=timeout_s) as r:
            body = json.loads(r.read() or b"null")
    except (OSError, ValueError, urllib.error.URLError) as e:
        raise RuntimeError(f"asmi {base}/serve/engines unreachable: {e}") from e
    rows = body.get("engines") if isinstance(body, dict) else body
    if not isinstance(rows, list):
        raise RuntimeError(f"asmi {base}/serve/engines answered without an engines list")
    return {
        str(f["request_field"])
        for e in rows if isinstance(e, dict)
        for f in (e.get("fields") or []) if isinstance(f, dict) and f.get("request_field")
    }


def classify_key(key: str, engine_fields: set[str]) -> str:
    if key in engine_fields:
        return "engine-mapped (a lost knob)"
    if key in LOAD_REQUEST_FIELDS:
        return "core asmi LoadRequest field"
    return "config-only (not an asmi field)"


def check_round_trip(recipes: list[dict], config_path: Path) -> int:
    if not config_path.exists():
        print(f"FAIL: serve configs missing at {config_path} — cannot check the asmi round trip")
        return 1
    configs = list(json.loads(config_path.read_text()).get("configs") or [])
    rt = serve_round_trip(recipes, configs)
    print(f"serve round trip: configs={len(configs)} dropped={len(rt['dropped'])} "
          f"changed={len(rt['changed'])} unmatched={len(rt['unmatched'])}")
    for row in rt["dropped"] + rt["changed"]:
        print(f"  {row['recipe']}: serve.{row['key']} config={row['config']!r} recipe={row['recipe_value']!r}")
    for name in rt["unmatched"]:
        print(f"  UNMATCHED config {name} (no recipe)")
    if not rt["unexplained"]:
        return 0
    try:
        engine_fields = engine_request_fields()
    except RuntimeError as e:
        print(f"FAIL: {e} — cannot classify {len(rt['unexplained'])} unexplained serve key(s)")
        return 1
    for row in rt["unexplained"]:
        print(f"  LOST serve.{row['key']} ({row['kind']}, {classify_key(row['key'], engine_fields)}) "
              f"in {row['recipe']}")
    print("FAIL: serve config fields did not reach their recipes — fix ingest, never the corpus")
    return 1


def verify(
    records_path: Path,
    *,
    skip_embed: bool = False,
    skip_gaps: bool = False,
    strict_gated: bool = False,
    serve_configs: Path | None = DEFAULT_SERVE_CONFIGS,
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

    missing = [r.get("recipe_id") for r in recipes if not (r.get("serve") or {}).get("engine")]
    if missing:
        print("FAIL: ENGINE-LESS recipes:", missing)
        return 1
    unfolded = unfolded_recipes(recs)
    if unfolded:
        print("FAIL: recipes without servability (fold did not run):", unfolded[:12])
        return 1
    not_servable = [
        (r.get("recipe_id") or r.get("id"), r.get("servable_reason"))
        for r in recs
        if r.get("kind") in RECIPE_KINDS and r.get("servable") is False
    ]
    print(f"servable: {sum(1 for r in recs if r.get('kind') in RECIPE_KINDS and r.get('servable'))}"
          f"  not servable: {len(not_servable)}")
    for rid, why in not_servable:
        print(f"  NOT SERVABLE {rid}: {why}")
    print("engines: ok")

    not_loadable = [(r.get("recipe_id"), r.get("problems")) for r in recipes if r.get("loadable") is False]
    print(f"loadable: {sum(1 for r in recipes if r.get('loadable') is not False)}  not loadable: {len(not_loadable)}")
    for rid, problems in not_loadable:
        print(f"  NOT LOADABLE {rid}: " + "; ".join(problems or []))
    if serve_configs is not None and check_round_trip(recipes, serve_configs):
        return 1

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
    ap.add_argument("--serve-configs", type=Path, default=DEFAULT_SERVE_CONFIGS,
                    help="serve configs the recipes were ingested from (asmi round trip)")
    ap.add_argument("--skip-round-trip", action="store_true")
    args = ap.parse_args()
    if not args.records.exists():
        print(f"FAIL: records missing at {args.records}")
        return 2
    return verify(
        args.records,
        skip_embed=args.skip_embed,
        skip_gaps=args.skip_gaps,
        strict_gated=args.strict_gated,
        serve_configs=None if args.skip_round_trip else args.serve_configs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
