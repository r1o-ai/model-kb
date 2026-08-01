#!/usr/bin/env python3
"""ingest_hf_family.py — HuggingFace model family tree → model-kb enrichments.

HF exposes the "family tree" via list filter (not a dedicated /tree of models):

  GET https://huggingface.co/api/models?filter=base_model:{namespace/name}
      &sort=downloads&direction=-1&limit=100

Each child carries tags like:
  base_model:Qwen/Qwen3.5-35B-A3B
  base_model:quantized:Qwen/Qwen3.5-35B-A3B
  4-bit | gguf | mlx | unsloth | fp8 | awq | gptq | ...

We also walk **up** from recipe hf_ids via cardData.base_model / tags so
mlx-community/* and unsloth/* resolve to the true base, then expand siblings.

Output enrichment kind: ``hf_family``
  measurements.family_levels = {
    "base": [...],
    "official_quant": [...],
    "mlx": [...],
    "gguf_unsloth": [...],
    "gguf_other": [...],
    "awq_gptq_fp8": [...],
    "finetune_abliterated": [...],
    "draft_dflash": [...],
    "other": [...]
  }

Usage:
  python3 ingest_hf_family.py
  python3 ingest_hf_family.py --limit 5 --join
  python3 ingest_hf_family.py --base Qwen/Qwen3.5-35B-A3B

Env: HF_TOKEN optional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
DEFAULT_RECIPES = ROOT / "records.jsonl"
DEFAULT_OUT = ROOT / "enrichments_hf_family.jsonl"
HF_API = "https://huggingface.co/api/models"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _http_json(url: str, timeout: float = 40.0) -> tuple[int, Any]:
    headers = {"User-Agent": "r1o-model-kb/1.0 (hf-family-tree)"}
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace") if e.fp else ""
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"error": raw[:400]}
    except Exception as e:  # noqa: BLE001
        return 0, {"error": str(e)}


def content_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_recipe_artifacts(path: Path) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") and r.get("kind") != "serve_recipe":
            continue
        art = r.get("artifact") or {}
        hf = (art.get("hf_id") or "").strip()
        name = (art.get("name") or "").strip()
        key = hf or name
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"hf_id": hf, "name": name, "recipe_id": str(r.get("recipe_id") or "")})
    return out


def extract_base_models(api: dict[str, Any]) -> list[str]:
    bases: list[str] = []
    card = api.get("cardData") or {}
    bm = card.get("base_model")
    if isinstance(bm, str) and bm:
        bases.append(bm)
    elif isinstance(bm, list):
        bases.extend(str(x) for x in bm if x)
    for t in api.get("tags") or []:
        if not isinstance(t, str):
            continue
        # base_model:Qwen/...  or base_model:quantized:Qwen/...
        if t.startswith("base_model:quantized:"):
            bases.append(t.split("base_model:quantized:", 1)[1])
        elif t.startswith("base_model:") and "quantized" not in t:
            bases.append(t.split("base_model:", 1)[1])
    # dedupe
    seen: set[str] = set()
    out: list[str] = []
    for b in bases:
        b = b.strip()
        if b and b not in seen and "/" in b:
            seen.add(b)
            out.append(b)
    return out


def resolve_root_base(hf_id: str, sleep_s: float) -> str:
    """Walk base_model links until root (or self if already root)."""
    if not hf_id or "/" not in hf_id:
        return hf_id
    current = hf_id
    seen: set[str] = set()
    for _ in range(4):
        if current in seen:
            break
        seen.add(current)
        code, api = _http_json(f"{HF_API}/{current}")
        time.sleep(sleep_s)
        if code != 200 or not isinstance(api, dict):
            break
        bases = extract_base_models(api)
        # prefer non-self base
        nxt = next((b for b in bases if b != current), None)
        if not nxt:
            return current
        current = nxt
    return current


def fetch_family(base_id: str, limit: int = 80) -> list[dict[str, Any]]:
    """Children declaring base_model:{base_id} via Hub filter API."""
    q = quote(f"base_model:{base_id}", safe="")
    # Hub uses filter= as tag filter
    url = (
        f"{HF_API}?filter={q}&limit={limit}&sort=downloads&direction=-1"
        f"&full=true"
    )
    code, data = _http_json(url)
    if code != 200 or not isinstance(data, list):
        # fallback: other= form used by some hub UIs
        url2 = f"{HF_API}?other={q}&limit={limit}&sort=downloads&direction=-1"
        code, data = _http_json(url2)
    if not isinstance(data, list):
        return []
    # also include the base itself
    return data


def classify_member(m: dict[str, Any], root_base: str) -> str:
    mid = m.get("id") or ""
    tags = [str(t).lower() for t in (m.get("tags") or [])]
    tagset = set(tags)
    name = mid.lower()

    if mid == root_base:
        return "base"

    # draft / dflash heads
    if "dflash" in name or "mtp" in name or "draft" in name:
        return "draft_dflash"

    # unsloth first (preferred community quant line)
    if mid.startswith("unsloth/") or "unsloth" in tagset:
        if "gguf" in tagset or name.endswith("-gguf") or "/gguf" in name:
            return "gguf_unsloth"
        return "unsloth_weights"

    # mlx community quants
    if mid.startswith("mlx-community/") or "mlx" in tagset or re.search(r"\b\d-bit\b", " ".join(tags)):
        if "mlx" in tagset or mid.startswith("mlx-community/"):
            return "mlx"

    # compressed / production quants
    if any(t in tagset or t in name for t in ("awq", "gptq", "fp8", "nvfp4", "mxfp4", "int4", "int8")):
        return "awq_gptq_fp8"
    if any(x in name for x in ("-awq", "-gptq", "-fp8", "-nvfp4", "mxfp4")):
        return "awq_gptq_fp8"

    # GGUF others
    if "gguf" in tagset or name.endswith("-gguf") or "-gguf" in name:
        return "gguf_other"

    # abliterated / uncensored finetunes
    if any(x in name for x in ("abliterat", "uncensor", "heretic", "wasserstein")):
        return "finetune_abliterated"

    # official org quants of same namespace
    root_ns = root_base.split("/")[0]
    if mid.startswith(root_ns + "/") and mid != root_base:
        return "official_quant"

    return "other"


def level_label(level: str) -> str:
    return {
        "base": "0_base",
        "official_quant": "1_official_quant",
        "mlx": "2_mlx",
        "gguf_unsloth": "2_gguf_unsloth",
        "gguf_other": "3_gguf_other",
        "awq_gptq_fp8": "2_compressed",
        "draft_dflash": "2_draft",
        "unsloth_weights": "2_unsloth_weights",
        "finetune_abliterated": "3_finetune",
        "other": "4_other",
    }.get(level, f"9_{level}")


def summarize_member(m: dict[str, Any], level: str) -> dict[str, Any]:
    tags = m.get("tags") or []
    quant_tags = [
        t
        for t in tags
        if isinstance(t, str)
        and (
            re.search(r"\d-bit|gguf|mlx|awq|gptq|fp8|mxfp4|nvfp4|bf16|int4|int8", t, re.I)
            or t.startswith("base_model:")
        )
    ]
    return {
        "id": m.get("id"),
        "downloads": m.get("downloads") or 0,
        "likes": m.get("likes") or 0,
        "pipeline_tag": m.get("pipeline_tag"),
        "level": level,
        "level_rank": level_label(level),
        "quant_tags": quant_tags[:12],
        "library": m.get("library_name"),
    }


def build_family_enrichment(
    root_base: str,
    members: list[dict[str, Any]],
    seed_artifacts: list[dict[str, str]],
) -> dict[str, Any]:
    levels: dict[str, list[dict[str, Any]]] = {}
    # ensure base present
    base_entry = next((m for m in members if m.get("id") == root_base), None)
    if not base_entry:
        base_entry = {"id": root_base, "downloads": 0, "likes": 0, "tags": []}
        members = [base_entry] + members

    for m in members:
        level = classify_member(m, root_base)
        levels.setdefault(level, []).append(summarize_member(m, level))

    # sort each level by downloads
    for k in levels:
        levels[k].sort(key=lambda x: (-(x.get("downloads") or 0), x.get("id") or ""))

    # pick recommended serve candidates for Apple Silicon
    recommend = []
    for key in ("mlx", "gguf_unsloth", "official_quant", "base", "draft_dflash"):
        for m in levels.get(key, [])[:3]:
            recommend.append({"id": m["id"], "level": key, "downloads": m["downloads"]})

    globs = [root_base, root_base.split("/")[-1], f"*{root_base.split('/')[-1]}*"]
    for a in seed_artifacts:
        if a.get("hf_id"):
            globs.append(a["hf_id"])
        if a.get("name"):
            globs.append(a["name"])
            globs.append(f"*{a['name']}*")

    flat_ids = [str(m.get("id")) for ms in levels.values() for m in ms if m.get("id")]
    search_bits = [
        "hf-family-tree",
        "model-family",
        root_base,
        " ".join(flat_ids[:40]),
        " ".join(levels.keys()),
        "unsloth" if levels.get("gguf_unsloth") else "",
        "mlx" if levels.get("mlx") else "",
        "gguf" if levels.get("gguf_unsloth") or levels.get("gguf_other") else "",
    ]

    eid = f"hffamily:{root_base.replace('/', '__')}"
    body = {
        "id": eid,
        "kind": "hf_family",
        "enrichment_id": eid,
        "title": f"HF family tree · {root_base}",
        "applies_to": {
            "model_globs": list(dict.fromkeys(g for g in globs if g)),
            "hf_ids": [root_base],
            "base_models": [root_base],
            "family_member_ids": flat_ids[:100],
            "accel_types": [],
            "engines": ["mlx_lm", "mlx_vlm", "ds4"],
        },
        "claims": [
            f"{len(flat_ids)} Hub variants under base_model:{root_base}",
            f"levels: {', '.join(f'{k}={len(v)}' for k, v in sorted(levels.items()))}",
        ],
        "measurements": {
            "root_base": root_base,
            "family_levels": levels,
            "member_count": len(flat_ids),
            "recommend_serve": recommend[:12],
            "api": f"GET /api/models?filter=base_model:{root_base}",
        },
        "confidence": "measured" if len(flat_ids) > 1 else "pointer",
        "search_text": " ".join(x for x in search_bits if x).lower(),
        "search_narrative": (
            f"Hugging Face family for {root_base}: {len(flat_ids)} variants. "
            f"Levels: " + ", ".join(f"{k}({len(v)})" for k, v in sorted(levels.items())) + ". "
            + (
                f"Top serve candidates: "
                + ", ".join(r["id"] for r in recommend[:5])
                + "."
                if recommend
                else ""
            )
        ),
        "source": {
            "type": "huggingface",
            "api": "filter=base_model:",
            "root_base": root_base,
            "url": f"https://huggingface.co/{root_base}",
        },
        "sources_used": ["huggingface"],
        "created": _now(),
    }
    body["content_hash"] = content_hash(
        {"root": root_base, "members": flat_ids[:50], "levels": {k: len(v) for k, v in levels.items()}}
    )
    return body


def merge_jsonl(paths: list[Path], out: Path) -> int:
    seen: set[str] = set()
    rows: list[dict] = []
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = r.get("id") or r.get("enrichment_id")
            if not eid or eid in seen:
                continue
            seen.add(eid)
            rows.append(r)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""))
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(prog="ingest_hf_family")
    ap.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--base", action="append", default=[], help="Explicit base id (repeatable)")
    ap.add_argument("--limit", type=int, default=0, help="Max bases from recipes (0=all)")
    ap.add_argument("--family-limit", type=int, default=60, help="Max children per base from Hub")
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--join", action="store_true")
    args = ap.parse_args()

    artifacts = load_recipe_artifacts(args.recipes)
    # map root_base -> seeds
    roots: dict[str, list[dict[str, str]]] = {}

    if args.base:
        for b in args.base:
            roots.setdefault(b, [])
    else:
        arts = artifacts[: args.limit] if args.limit else artifacts
        print(f"resolving bases for {len(arts)} artifacts…")
        for art in arts:
            seed = art.get("hf_id") or art.get("name") or ""
            if not seed:
                continue
            if "/" not in seed:
                # name only — try as search later; skip walk
                print(f"  skip non-hf id name={seed}")
                continue
            root = resolve_root_base(seed, args.sleep)
            roots.setdefault(root, []).append(art)
            print(f"  {seed} → root {root}")

    print(f"unique roots: {len(roots)}")
    written = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fout:
        for root, seeds in sorted(roots.items()):
            members = fetch_family(root, limit=args.family_limit)
            time.sleep(args.sleep)
            # enrich members with tags when list endpoint is sparse
            rich: list[dict[str, Any]] = []
            for m in members:
                mid = m.get("id")
                if not mid:
                    continue
                # list endpoint often already has tags/downloads
                if m.get("tags") is None:
                    code, full = _http_json(f"{HF_API}/{mid}")
                    time.sleep(args.sleep * 0.5)
                    if code == 200 and isinstance(full, dict):
                        m = full
                rich.append(m)
            if not rich:
                # still write stub with base only
                code, base_api = _http_json(f"{HF_API}/{root}")
                time.sleep(args.sleep)
                if code == 200 and isinstance(base_api, dict):
                    rich = [base_api]
            rec = build_family_enrichment(root, rich, seeds)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            mc = rec["measurements"]["member_count"]
            levels = {k: len(v) for k, v in rec["measurements"]["family_levels"].items()}
            print(f"  + {root}: members={mc} levels={levels}")

    print(f"wrote {written} → {args.out}")

    if args.join and written:
        import subprocess
        import sys

        merged_path = ROOT / "enrichments.jsonl"
        # merge existing research + hf cards + family
        n = merge_jsonl(
            [
                ROOT / "enrichments.jsonl",
                ROOT / "enrichments_hf.jsonl",
                args.out,
            ],
            ROOT / "enrichments.merged.jsonl",
        )
        ROOT.joinpath("enrichments.merged.jsonl").replace(merged_path)
        print(f"merged enrichments: {n}")
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "join_enrichments.py"),
                "--recipes",
                str(args.recipes),
                "--enrich",
                str(merged_path),
                "--out",
                str(args.recipes),
            ],
            cwd=str(ROOT),
        )
        subprocess.check_call(
            [sys.executable, str(ROOT / "build_bm25.py"), "--rebuild"],
            cwd=str(ROOT),
        )
        print("join + BM25 OK")

    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
