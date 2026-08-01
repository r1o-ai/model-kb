#!/usr/bin/env python3
"""ingest_hf_cards.py — HuggingFace / Unsloth model-card measurements → enrichments.

Pulls what Unsloth (and HF cards) actually publish for serve decisions:
  - base_model linkage (mlx-community → upstream)
  - quality benchmark tables (MMLU-Pro, SWE-bench, GPQA, …)
  - GGUF quant inventory + sizes when present
  - Dynamic/imatrix/KLD marketing signals (searchable claims)

Does NOT invent t/s — local tok/s stay in recipe.benchmarks from serve-configs.
Community quality scores inform *which quant/family to prefer*; local benches
inform *how we run it*.

Usage:
  python3 ingest_hf_cards.py
  python3 ingest_hf_cards.py --out enrichments_hf.jsonl --limit 5
  python3 ingest_hf_cards.py --join   # also join into records.jsonl + rebuild BM25

Env:
  HF_TOKEN optional (higher rate limits)
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

ROOT = Path(__file__).resolve().parent
DEFAULT_RECIPES = ROOT / "records.jsonl"
DEFAULT_OUT = ROOT / "enrichments_hf.jsonl"
HF_API = "https://huggingface.co/api/models"
HF_RAW = "https://huggingface.co"

# Quality metrics we care about for "which model to serve" ranking
QUALITY_KEYS = {
    "mmlu-pro",
    "mmlu_pro",
    "mmlu-redux",
    "gpqa",
    "gpqa diamond",
    "swe-bench",
    "swe-bench verified",
    "livecodebench",
    "ifeval",
    "aa-lcr",
    "hle",
    "bfcl",
    "tau2-bench",
    "mmmu",
    "mmmu-pro",
    "aime",
    "hmmt",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _http_get(url: str, timeout: float = 30.0) -> tuple[int, str]:
    headers = {"User-Agent": "r1o-model-kb/1.0 (unsloth-hf-card-ingest)"}
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace") if e.fp else ""
        return e.code, body
    except Exception as e:  # noqa: BLE001
        return 0, str(e)


def _http_json(url: str) -> tuple[int, Any]:
    code, raw = _http_get(url)
    try:
        return code, json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return code, {"_raw": raw[:500]}


def content_hash(obj: dict[str, Any]) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_recipe_hf_ids(path: Path) -> list[dict[str, str]]:
    """Unique artifact identities from serve recipes."""
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
        if r.get("kind") not in (None, "serve_recipe") and r.get("kind") != "serve_recipe":
            if r.get("kind") and r.get("kind") != "serve_recipe":
                continue
        art = r.get("artifact") or {}
        hf = (art.get("hf_id") or "").strip()
        name = (art.get("name") or "").strip()
        key = hf or name
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "hf_id": hf,
                "name": name,
                "recipe_id": str(r.get("recipe_id") or ""),
            }
        )
    return out


def resolve_unsloth_candidates(hf_id: str, name: str) -> list[str]:
    """Map a recipe artifact to likely Unsloth / HF card repos."""
    cands: list[str] = []
    if hf_id:
        cands.append(hf_id)
        # mlx-community/Foo-4bit → try unsloth/Foo-GGUF and base
        base = hf_id.split("/")[-1]
        base_clean = re.sub(
            r"-(?:mlx-)?(?:\d+bit|mixed-[\d-]+|mxfp4|nvfp4|bf16|abliterated).*$",
            "",
            base,
            flags=re.I,
        )
        base_clean = re.sub(r"-mlx$", "", base_clean, flags=re.I)
        if hf_id.startswith("mlx-community/") or "mlx" in hf_id.lower():
            cands.append(f"unsloth/{base_clean}-GGUF")
            cands.append(f"unsloth/{base_clean}")
        if not hf_id.startswith("unsloth/"):
            # also search API for unsloth variants of base
            cands.append(f"unsloth/{base}-GGUF")
            cands.append(f"unsloth/{base_clean}-GGUF")
    if name:
        n = re.sub(r"-(?:mlx-)?(?:\d+bit|mixed-[\d-]+|mxfp4).*$", "", name, flags=re.I)
        cands.append(f"unsloth/{n}-GGUF")
        cands.append(f"unsloth/{n}")
    # dedupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        c = c.strip().strip("/")
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def parse_html_quality_tables(readme: str) -> list[dict[str, Any]]:
    """Extract model-column quality scores from Unsloth HTML tables.

    Tables are wide: first column = metric category/name, subsequent columns = models.
    We keep the last column that matches our model when possible, else all numeric
    cells for metrics we care about.
    """
    tables = re.findall(r"<table[\s\S]*?</table>", readme, flags=re.I)
    measurements: list[dict[str, Any]] = []
    for ti, table in enumerate(tables):
        # rows as list of cell texts
        rows: list[list[str]] = []
        for tr in re.findall(r"<tr[\s\S]*?</tr>", table, flags=re.I):
            cells = re.findall(r"<t[dh][^>]*>([\s\S]*?)</t[dh]>", tr, flags=re.I)
            plain = []
            for c in cells:
                t = re.sub(r"<br\s*/?>", " ", c, flags=re.I)
                t = re.sub(r"<[^>]+>", "", t)
                t = re.sub(r"\s+", " ", t).strip()
                plain.append(t)
            if plain:
                rows.append(plain)
        if len(rows) < 2:
            continue
        # detect header row: models
        header = rows[0]
        # metrics rows
        current_section = ""
        for row in rows[1:]:
            if not row:
                continue
            # section headers often single non-numeric label spanning
            if len(row) == 1 or (len(row) >= 1 and not any(_is_score(x) for x in row[1:])):
                # might be section title if first cell has no score
                if row[0] and not _is_score(row[0]):
                    current_section = row[0]
                continue
            metric = row[0]
            if not metric or _is_score(metric):
                continue
            metric_l = metric.lower().strip()
            # collect scores per column
            for ci, cell in enumerate(row[1:], start=1):
                if not _is_score(cell):
                    continue
                model_col = header[ci] if ci < len(header) else f"col{ci}"
                if not _metric_interesting(metric_l):
                    continue
                val = _parse_score(cell)
                if val is None:
                    continue
                measurements.append(
                    {
                        "metric": metric,
                        "metric_norm": re.sub(r"[^a-z0-9]+", "_", metric_l).strip("_"),
                        "value": val,
                        "raw": cell,
                        "model_column": model_col,
                        "section": current_section,
                        "table": ti,
                        "source": "hf_readme_table",
                    }
                )
    return measurements


def _is_score(s: str) -> bool:
    s = s.strip()
    if not s or s in ("--", "—", "–", "N/A", "n/a"):
        return False
    # 85.3 or 40.5 / 34.1 or 2160
    return bool(re.match(r"^-?\d+(\.\d+)?(\s*/\s*-?\d+(\.\d+)?)?$", s))


def _parse_score(s: str) -> float | None:
    s = s.strip()
    if "/" in s:
        s = s.split("/")[0].strip()
    try:
        return float(s)
    except ValueError:
        return None


def _metric_interesting(metric_l: str) -> bool:
    if any(k in metric_l for k in QUALITY_KEYS):
        return True
    # broader useful buckets
    for k in (
        "mmlu",
        "gpqa",
        "swe",
        "code",
        "livecode",
        "ifeval",
        "bench",
        "mmmu",
        "aime",
        "math",
        "agent",
        "browse",
        "tau",
        "bfcl",
        "hle",
    ):
        if k in metric_l:
            return True
    return False


def parse_quant_inventory(api: dict[str, Any], readme: str) -> list[dict[str, Any]]:
    """GGUF / quant files from HF siblings + README mentions."""
    quants: list[dict[str, Any]] = []
    for sib in api.get("siblings") or []:
        fn = sib.get("rfilename") or ""
        if not re.search(r"\.(gguf|safetensors)$", fn, re.I):
            continue
        # skip multi-part shards listing as separate except first
        qlabel = None
        m = re.search(
            r"(Q\d+_K(?:_[A-Z]+)?|Q\d+_0|IQ\d+\w*|MXFP4\w*|BF16|F16|FP16|UD-\w+)",
            fn,
            re.I,
        )
        if m:
            qlabel = m.group(1).upper().replace("FP16", "F16")
        size = sib.get("size")
        quants.append(
            {
                "file": fn,
                "quant": qlabel,
                "size_bytes": size,
                "source": "hf_siblings",
            }
        )
    # README preferred quant mentions
    for m in re.finditer(
        r"\b(Q[0-9]+_K(?:_[A-Z]+)?|MXFP4(?:_MOE)?|BF16|Dynamic 2\.0|imatrix)\b",
        readme,
        re.I,
    ):
        tok = m.group(1)
        if not any(
            q.get("quant") == tok.upper()
            or (q.get("file") or "").find(tok) >= 0
            for q in quants
        ):
            quants.append({"file": None, "quant": tok, "size_bytes": None, "source": "readme_mention"})
    return quants


def parse_claims(readme: str) -> list[str]:
    claims: list[str] = []
    patterns = [
        r"Unsloth Dynamic 2\.0[^\n.]{0,80}",
        r"imatrix[^\n.]{0,80}",
        r"reduced the maximum KLD[^\n.]{0,80}",
        r"outperforms other leading quants[^\n.]{0,60}",
        r"tool-calling[^\n.]{0,80}",
        r"(\d+(?:\.\d+)?x faster)[^\n.]{0,40}",
        r"(\d+%\s*less)[^\n.]{0,40}",
    ]
    for pat in patterns:
        for m in re.finditer(pat, readme, re.I):
            c = re.sub(r"\s+", " ", m.group(0)).strip()
            if c and c not in claims:
                claims.append(c[:200])
    return claims[:20]


def pick_best_repo(cands: list[str]) -> tuple[str | None, dict, str]:
    """Return (repo_id, api_json, readme) for first existing candidate; prefer unsloth GGUF."""
    # Prefer unsloth *-GGUF then unsloth then original
    ordered = sorted(
        cands,
        key=lambda r: (
            0 if r.startswith("unsloth/") and r.endswith("-GGUF") else
            1 if r.startswith("unsloth/") else
            2
        ),
    )
    for repo in ordered:
        code, api = _http_json(f"{HF_API}/{repo}")
        if code != 200 or not isinstance(api, dict) or not api.get("id"):
            continue
        # README
        rc, readme = _http_get(f"{HF_RAW}/{repo}/raw/main/README.md")
        if rc != 200:
            readme = ""
        return repo, api, readme
    return None, {}, ""


def build_enrichment(
    artifact: dict[str, str],
    repo: str,
    api: dict[str, Any],
    readme: str,
) -> dict[str, Any]:
    card = api.get("cardData") or {}
    base = card.get("base_model") or []
    if isinstance(base, str):
        base = [base]
    # tags often include base_model:Foo
    tags = api.get("tags") or []
    for t in tags:
        if isinstance(t, str) and t.startswith("base_model:") and "quantized" not in t:
            b = t.split(":", 1)[1]
            if b and b not in base:
                base.append(b)

    quality = parse_html_quality_tables(readme)
    # Prefer columns matching our artifact name / base
    name = artifact.get("name") or ""
    focus = []
    for m in quality:
        col = (m.get("model_column") or "").lower()
        if name and any(tok.lower() in col for tok in re.split(r"[-_/]", name) if len(tok) > 3):
            focus.append(m)
        elif any(str(b).split("/")[-1].lower() in col for b in base):
            focus.append(m)
    if not focus and quality:
        # take last model column (often the card's own model)
        # group by metric keep last
        by_metric: dict[str, dict] = {}
        for m in quality:
            by_metric[m["metric_norm"]] = m
        focus = list(by_metric.values())

    quants = parse_quant_inventory(api, readme)
    claims = parse_claims(readme)

    # compact quality map for ranking
    quality_map: dict[str, float] = {}
    for m in focus:
        quality_map[m["metric_norm"]] = m["value"]

    applies_globs = []
    if artifact.get("hf_id"):
        applies_globs.append(artifact["hf_id"])
        applies_globs.append(artifact["hf_id"].split("/")[-1])
    if artifact.get("name"):
        applies_globs.append(artifact["name"])
        applies_globs.append(f"*{artifact['name']}*")
    for b in base:
        applies_globs.append(b)
        applies_globs.append(b.split("/")[-1])
        applies_globs.append(f"*{b.split('/')[-1]}*")

    search_bits = [
        "unsloth" if repo.startswith("unsloth/") else "hf-card",
        "model-card",
        "measurements",
        repo,
        artifact.get("name") or "",
        artifact.get("hf_id") or "",
        " ".join(base),
        " ".join(quality_map.keys()),
        " ".join(str(int(v)) if v == int(v) else f"{v:.1f}" for v in quality_map.values()),
        " ".join(q.get("quant") or "" for q in quants if q.get("quant")),
        " ".join(claims[:5]),
    ]
    if quality_map.get("mmlu_pro") or quality_map.get("mmlu-pro"):
        search_bits.append("quality:mmlu_pro")
    if any("swe" in k for k in quality_map):
        search_bits.append("quality:coding")
    if repo.startswith("unsloth/"):
        search_bits.append("unsloth-dynamic")

    eid = f"hfcard:{repo.replace('/', '__')}"
    rec = {
        "id": eid,
        "kind": "hf_card",
        "enrichment_id": eid,
        "title": f"HF/Unsloth card · {repo}",
        "applies_to": {
            "model_globs": list(dict.fromkeys(applies_globs)),
            "hf_ids": [artifact["hf_id"]] if artifact.get("hf_id") else [],
            "base_models": base,
            "accel_types": [],
            "engines": ["mlx_lm", "mlx_vlm", "ds4"],
        },
        "claims": claims,
        "measurements": {
            "quality": quality_map,
            "quality_rows": focus[:40],
            "quants": quants[:40],
            "downloads": api.get("downloads"),
            "likes": api.get("likes"),
            "pipeline_tag": api.get("pipeline_tag") or card.get("pipeline_tag"),
            "gguf_arch": (api.get("gguf") or {}).get("architecture"),
            "context_length": (api.get("gguf") or {}).get("context_length"),
        },
        "confidence": "measured" if quality_map else "pointer",
        "search_text": " ".join(x for x in search_bits if x).lower(),
        "search_narrative": (
            f"Model card for {repo}"
            + (f" (base {', '.join(base)})" if base else "")
            + (
                f". Quality: "
                + ", ".join(f"{k}={v}" for k, v in list(quality_map.items())[:8])
                if quality_map
                else ". No parseable quality table."
            )
            + (
                f" Quants: {', '.join(sorted({q['quant'] for q in quants if q.get('quant')})[:12])}."
                if quants
                else ""
            )
        ),
        "source": {
            "type": "huggingface",
            "repo": repo,
            "url": f"https://huggingface.co/{repo}",
            "card_url": f"https://huggingface.co/{repo}/blob/main/README.md",
            "unsloth": repo.startswith("unsloth/"),
        },
        "sources_used": ["huggingface", "unsloth"] if repo.startswith("unsloth/") else ["huggingface"],
        "created": _now(),
    }
    rec["content_hash"] = content_hash(
        {k: rec[k] for k in ("title", "measurements", "claims", "applies_to") if k in rec}
    )
    return rec


def merge_enrichment_files(paths: list[Path], out: Path) -> int:
    """Concatenate unique enrichment ids."""
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
    ap = argparse.ArgumentParser(prog="ingest_hf_cards")
    ap.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="Max artifacts (0=all)")
    ap.add_argument("--sleep", type=float, default=0.35, help="Delay between HF calls")
    ap.add_argument("--join", action="store_true", help="Merge into enrichments.jsonl + join + BM25")
    args = ap.parse_args()

    arts = load_recipe_hf_ids(args.recipes)
    if args.limit:
        arts = arts[: args.limit]
    print(f"artifacts: {len(arts)}")

    written = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fout:
        for art in arts:
            cands = resolve_unsloth_candidates(art.get("hf_id") or "", art.get("name") or "")
            repo, api, readme = pick_best_repo(cands)
            if not repo:
                print(f"  skip (no card): {art.get('hf_id') or art.get('name')}")
                time.sleep(args.sleep)
                continue
            rec = build_enrichment(art, repo, api, readme)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            qn = len(rec.get("measurements", {}).get("quality") or {})
            print(f"  + {repo}  quality_metrics={qn}  for {art.get('hf_id') or art.get('name')}")
            time.sleep(args.sleep)

    print(f"wrote {written} → {args.out}")

    if args.join and written:
        # merge research + hf into enrichments.jsonl
        merged = ROOT / "enrichments.jsonl"
        n = merge_enrichment_files(
            [ROOT / "enrichments.jsonl", args.out],
            ROOT / "enrichments.merged.jsonl",
        )
        ROOT.joinpath("enrichments.merged.jsonl").replace(merged)
        print(f"merged enrichments: {n}")
        # join + bm25
        import subprocess
        import sys

        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "join_enrichments.py"),
                "--recipes",
                str(args.recipes),
                "--enrich",
                str(merged),
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
