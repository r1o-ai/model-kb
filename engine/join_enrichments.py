#!/usr/bin/env python3
"""join_enrichments.py — attach research_note enrichments onto serve recipes.

Join algorithm from ARDD plan §2.2:
  exact hf_id 100 | name/basename 90 | fnmatch globs 70
  +15 accel subset | +10 engine match
  attach ≥85, review 70–84, unjoined <70 (still indexes as research_note)

Mutates recipe records in-memory:
  - enrichments: [enrichment_id, ...]
  - enrichment_links: [{id, score, title}, ...]
  - extends search_text / search_narrative with claim tokens

Usage:
  python3 join_enrichments.py --recipes records.jsonl --enrich enrichments.jsonl
  # writes merged records.jsonl (recipes + unjoined notes)
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def score_join(note: dict[str, Any], recipe: dict[str, Any]) -> int:
    applies = note.get("applies_to") or {}
    art = recipe.get("artifact") or {}
    name = str(art.get("name") or "")
    hf = str(art.get("hf_id") or "")
    basename = name
    path = str(art.get("path") or "")
    if path:
        basename = path.rstrip("/").split("/")[-1] or name

    score = 0
    globs = applies.get("model_globs") or []

    # HF family tree: recipe is in the member set or shares base
    kind = note.get("kind") or ""
    if kind in ("hf_family", "hf_card"):
        members = set(applies.get("family_member_ids") or [])
        bases = set(applies.get("base_models") or applies.get("hf_ids") or [])
        if hf and hf in members:
            score = max(score, 95)
        if hf and hf in bases:
            score = max(score, 100)
        if name and any(name in str(m) or str(m).endswith(name) for m in members):
            score = max(score, 90)
        # mlx-community/X-4bit under base Qwen/X
        if hf and bases:
            for b in bases:
                b_tail = str(b).split("/")[-1].lower()
                if b_tail and b_tail in hf.lower():
                    score = max(score, 88)
                if b_tail and b_tail in name.lower():
                    score = max(score, 88)

    # exact matches
    for g in globs:
        if g and not any(c in g for c in "*?[]"):
            if g == hf:
                score = max(score, 100)
            if g == name or g == basename:
                score = max(score, 90)

    # fnmatch globs
    for g in globs:
        if not g:
            continue
        for target in (name, hf, basename):
            if target and fnmatch.fnmatch(target, g):
                score = max(score, 70)
            # also try casefold
            if target and fnmatch.fnmatch(target.lower(), g.lower()):
                score = max(score, 70)

    # If no globs at all, try accel/engine-only soft match later — base 0
    if score < 70:
        # token overlap fallback for notes without model globs: title vs name
        title = (note.get("title") or "").lower()
        if name and name.lower().split("-")[0] in title:
            # weak — do not reach attach threshold alone
            score = max(score, 40)
        # ds4 inventory basenames (ds4-gguf, ds4-flash-q4) ↔ DeepSeek wiki notes
        name_l = (name or basename or "").lower()
        hf_l = hf.lower()
        note_blob = f"{title} {(note.get('search_text') or '')[:400]}".lower()
        if any(t in name_l or t in hf_l for t in ("ds4", "deepseek", "dwarfstar")) and any(
            t in note_blob for t in ("deepseek", "ds4", "dwarfstar", "v4 flash", "v4-pro")
        ):
            score = max(score, 75)
        if "deepseek" in hf_l and "deepseek" in note_blob:
            # base 80 (review band); engine match below pushes attach ≥85
            score = max(score, 80)

    if score >= 70:
        recipe_accels = {
            a.get("type")
            for a in (recipe.get("accelerators") or [])
            if a.get("enabled", True) and a.get("type")
        }
        note_accels = set(applies.get("accel_types") or [])
        if note_accels and note_accels.issubset(recipe_accels):
            score += 15
        elif note_accels and note_accels & recipe_accels:
            score += 8

        eng = (recipe.get("serve") or {}).get("engine")
        note_eng = set(applies.get("engines") or [])
        if eng and eng in note_eng:
            score += 10
        # ds4/deepseek recipes: engine match is strong signal even when
        # accel sets don't subset (ssd_streaming/kv_disk aren't in note hints)
        if eng in ("ds4",) and eng in note_eng and score < 85:
            score = max(score, 85)
        hf_id = str((recipe.get("artifact") or {}).get("hf_id") or "").lower()
        if hf_id.startswith("deepseek") and score >= 80:
            # measured wiki notes about deepseek attach to flash/pro recipes
            if any(t in (note.get("title") or "").lower() for t in ("deepseek", "ds4", "dwarfstar", "mhc")):
                score = max(score, 88)

    return score


def claim_tokens(note: dict[str, Any], limit: int = 24) -> list[str]:
    toks: list[str] = []
    for c in note.get("claims") or []:
        if isinstance(c, dict):
            m = c.get("metrics") or {}
            if m.get("agg_tps") is not None:
                toks.append(f"{m['agg_tps']} tok/s")
            if m.get("context_len"):
                toks.append(f"{m['context_len']}ctx")
            for w in re.findall(r"[a-z0-9./+-]+", (c.get("text") or "").lower())[:6]:
                toks.append(w)
        elif isinstance(c, str):
            for w in re.findall(r"[a-z0-9./+-]+", c.lower())[:8]:
                toks.append(w)
        if len(toks) >= limit:
            break
    # HF card quality map → searchable
    meas = note.get("measurements") or {}
    quality = meas.get("quality") or {}
    if isinstance(quality, dict):
        for k, v in list(quality.items())[:12]:
            toks.append(str(k))
            try:
                toks.append(f"{float(v):.1f}".rstrip("0").rstrip("."))
            except (TypeError, ValueError):
                pass
            toks.append(f"quality:{k}")
    # family levels
    levels = meas.get("family_levels") or {}
    if isinstance(levels, dict):
        toks.append("hf-family-tree")
        for lk, members in levels.items():
            toks.append(f"level:{lk}")
            if isinstance(members, list):
                for m in members[:5]:
                    if isinstance(m, dict) and m.get("id"):
                        toks.append(str(m["id"]).split("/")[-1].lower())
    for q in (meas.get("quants") or [])[:8]:
        if isinstance(q, dict) and q.get("quant"):
            toks.append(str(q["quant"]).lower())
    return toks[:limit]


def attach(note: dict[str, Any], recipe: dict[str, Any], score: int) -> None:
    eid = note.get("enrichment_id") or note.get("id")
    links = recipe.setdefault("enrichment_links", [])
    if any(l.get("id") == eid for l in links):
        return
    kind = note.get("kind") or "research_note"
    links.append(
        {
            "id": eid,
            "score": score,
            "title": note.get("title"),
            "confidence": note.get("confidence"),
            "kind": kind,
        }
    )
    links.sort(key=lambda x: -int(x.get("score") or 0))
    recipe["enrichments"] = [l["id"] for l in links]

    # structured community quality / family (for serve ranking UI)
    meas = note.get("measurements") or {}
    if kind == "hf_card" and isinstance(meas.get("quality"), dict) and meas["quality"]:
        cq = recipe.setdefault("community_quality", {})
        # merge preferring higher attach score notes (later links sort by score)
        for k, v in meas["quality"].items():
            cq[k] = v
        cq["_source"] = note.get("source", {}).get("repo") or eid
    if kind == "hf_family" and isinstance(meas.get("family_levels"), dict):
        recipe["hf_family"] = {
            "root_base": meas.get("root_base"),
            "member_count": meas.get("member_count"),
            "levels": {k: len(v) if isinstance(v, list) else 0 for k, v in meas["family_levels"].items()},
            "recommend_serve": meas.get("recommend_serve") or [],
            "enrichment_id": eid,
        }

    # extend search fields
    extra = claim_tokens(note)
    st = recipe.get("search_text") or ""
    recipe["search_text"] = re.sub(r"\s+", " ", (st + " " + " ".join(extra)).lower()).strip()
    sn = str(recipe.get("search_narrative") or "")
    cite = f" [{eid}]"
    claim_bits: list[str] = []
    for c in note.get("claims") or []:
        if isinstance(c, dict) and c.get("confidence") == "measured" and c.get("text"):
            claim_bits.append(c["text"])
        elif isinstance(c, str):
            claim_bits.append(c)
    add = " ".join(claim_bits[:2])
    if kind == "hf_family" and meas.get("root_base"):
        add = (add + f" Family root {meas['root_base']} ({meas.get('member_count')} variants).").strip()
    if kind == "hf_card" and meas.get("quality"):
        top = list(meas["quality"].items())[:4]
        add = (add + " Card quality: " + ", ".join(f"{k}={v}" for k, v in top) + ".").strip()
    eid_s = str(eid or "")
    if add and eid_s and eid_s not in sn:
        recipe["search_narrative"] = (sn + " Research: " + add + cite).strip()[:1200]

    sources = set(recipe.get("sources_used") or [])
    if kind.startswith("hf_"):
        sources.add("huggingface")
        if note.get("source", {}).get("unsloth"):
            sources.add("unsloth")
    else:
        sources.add("wiki")
    recipe["sources_used"] = sorted(sources)


def main() -> int:
    ap = argparse.ArgumentParser(prog="join_enrichments")
    ap.add_argument("--recipes", type=Path, default=ROOT / "records.jsonl")
    ap.add_argument("--enrich", type=Path, default=ROOT / "enrichments.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "records.jsonl")
    ap.add_argument("--review-out", type=Path, default=ROOT / "join_review.jsonl")
    args = ap.parse_args()

    all_rows = load_jsonl(args.recipes)
    recipes = [r for r in all_rows if r.get("kind") == "serve_recipe"]
    # Preserve every non-recipe, non-research row (cloud_context, etc.).
    # research_note rows from a prior join are dropped and re-added from --enrich
    # so the dual-index set stays a pure function of the current enrichment file.
    preserved = [
        r
        for r in all_rows
        if r.get("kind") not in ("serve_recipe", "research_note", "hf_family", "hf_card")
        and not str(r.get("kind") or "").startswith("hf_")
    ]
    notes = load_jsonl(args.enrich)
    if not notes:
        print("no enrichments — pass-through recipes + preserved kinds")
        out = list(recipes) + preserved
        args.out.write_text("".join(json.dumps(r, default=str) + "\n" for r in out))
        return 0

    # reset previous enrichment attachments on recipes
    for r in recipes:
        r.pop("enrichments", None)
        r.pop("enrichment_links", None)

    review: list[dict[str, Any]] = []
    attached_ids: set[str] = set()
    unjoined: list[dict[str, Any]] = []

    stats = {"attach": 0, "review": 0, "unjoined": 0, "links": 0}

    for note in notes:
        eid = note.get("enrichment_id") or note.get("id")
        best: list[tuple[int, dict[str, Any]]] = []
        for recipe in recipes:
            s = score_join(note, recipe)
            if s > 0:
                best.append((s, recipe))
        best.sort(key=lambda x: -x[0])

        did_attach = False
        for s, recipe in best:
            if s >= 85:
                attach(note, recipe, s)
                did_attach = True
                attached_ids.add(str(eid))
                stats["links"] += 1
            elif 70 <= s < 85:
                review.append(
                    {
                        "enrichment_id": eid,
                        "title": note.get("title"),
                        "recipe_id": recipe.get("recipe_id"),
                        "score": s,
                        "artifact": (recipe.get("artifact") or {}).get("name"),
                    }
                )
                stats["review"] += 1

        if did_attach:
            stats["attach"] += 1
        else:
            stats["unjoined"] += 1
            unjoined.append(note)

    # recipes + preserved (cloud_context, …) + all research notes
    # (attached ones stay as independent searchable rows — dual retrieval)
    out_rows: list[dict[str, Any]] = list(recipes) + preserved + list(notes)

    args.out.write_text("".join(json.dumps(r, default=str) + "\n" for r in out_rows))
    args.review_out.write_text("".join(json.dumps(r, default=str) + "\n" for r in review))

    recipes_with = sum(1 for r in recipes if r.get("enrichments"))
    print(
        f"join: attach_notes={stats['attach']} links={stats['links']} "
        f"review_pairs={stats['review']} unjoined_notes={stats['unjoined']} "
        f"recipes_enriched={recipes_with}/{len(recipes)} "
        f"preserved={len(preserved)} → {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
