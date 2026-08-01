#!/usr/bin/env python3
"""enrich_optimization_transfer.py — attach optimization_transfer[] onto each recipe.

For every serve_recipe, find accelerators proven on related recipes (same family /
quant / finetune) and codify applicability + expected acceptance multiplier.

Usage:
  python3 enrich_optimization_transfer.py
  python3 enrich_optimization_transfer.py --recipes records.jsonl --out records.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from optimization_transfer import summarize_for_recipe

ROOT = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", type=Path, default=ROOT / "records.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "records.jsonl")
    args = ap.parse_args()

    rows = load_jsonl(args.recipes)
    recipes = [r for r in rows if r.get("kind") == "serve_recipe"]
    others = [r for r in rows if r.get("kind") != "serve_recipe"]

    n_with = 0
    for r in recipes:
        summary = summarize_for_recipe(r, recipes)
        actionable = summary.get("actionable_transfers") or []
        # store compact form on recipe
        r["optimization_transfer"] = {
            "by_status": summary.get("by_status") or {},
            "transfers": [
                {
                    "accel": t["accel"],
                    "relation": t["relation"],
                    "status": t["status"],
                    "acceptance_mult": t["acceptance_mult"],
                    "expected_avg_accepted_tokens": t.get("expected_avg_accepted_tokens"),
                    "confidence": t["confidence"],
                    "caveat": t.get("caveat"),
                    "serve_recommendation": t.get("serve_recommendation"),
                    "proven_on": t.get("proven_on"),
                }
                for t in actionable
                if t.get("status") != "deny"
            ],
        }
        # BM25 tokens so "mtp finetune degraded" is searchable
        st = r.get("search_text") or ""
        extra = []
        for t in r["optimization_transfer"]["transfers"][:12]:
            extra.append(f"opt:{t['accel']}")
            extra.append(f"transfer:{t['status']}")
            extra.append(f"relation:{t['relation']}")
        if extra:
            r["search_text"] = (st + " " + " ".join(extra)).lower()
            n_with += 1
        # narrative blurb
        degraded = [t for t in actionable if t.get("status") == "degraded"]
        if degraded:
            bits = ", ".join(
                f"{t['accel']} from {t['proven_on'].get('recipe_id')} (×{t['acceptance_mult']})"
                for t in degraded[:3]
            )
            sn = str(r.get("search_narrative") or "")
            add = f" Optimization transfer (degraded acceptance): {bits}."
            if add not in sn:
                r["search_narrative"] = (sn + add).strip()[:1200]

    out_rows = recipes + others
    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in out_rows) + "\n"
    )
    print(f"optimization_transfer: {n_with}/{len(recipes)} recipes with cross-links → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
