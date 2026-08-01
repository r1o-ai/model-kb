#!/usr/bin/env python3
"""ingest_research.py — wiki/research corpus → research_note enrichments.

Phase C of model-kb ARDD plan (2026-07-11). Reads research-corpus.glob,
parses markdown tables for measured metrics (no invented numbers), and
emits kind=research_note records for BM25 + join onto serve recipes.

Usage:
  python3 ingest_research.py
  python3 ingest_research.py --glob research-corpus.glob --out enrichments.jsonl
"""

from __future__ import annotations

import argparse
import glob as globmod
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
HOME = Path.home()
DEFAULT_GLOB = ROOT / "research-corpus.glob"
DEFAULT_OUT = ROOT / "enrichments.jsonl"

# Heuristic applies_to from path/title/body tokens (never invent metrics).
_ACCEL_HINTS = {
    "dflash": "dflash",
    "turboquant": "turboquant",
    "turbo-quant": "turboquant",
    "jaccl": "jaccl",
    "mtp": "mtp",
    "mtplx": "mtplx",
    "triattention": "triattention",
    "speculative": "dflash",
}
_ENGINE_HINTS = {
    "mlx-lm": "mlx_lm",
    "mlx_lm": "mlx_lm",
    "mlx-vlm": "mlx_vlm",
    "mlx_vlm": "mlx_vlm",
    "dflash-mlx": "dflash-mlx",
    "ds4": "ds4",
    "mtplx": "mtplx",
}
_MODEL_GLOBS = [
    (re.compile(r"qwen3\.?6.*35b|qwen3\.6-35b|35b-a3b", re.I), "*Qwen3.6*35B*A3B*"),
    (re.compile(r"qwen3\.?6.*27b|qwen3\.6-27b", re.I), "*Qwen3.6*27B*"),
    (re.compile(r"qwen3\.?5.*35b|qwen3\.5-35b", re.I), "*Qwen3.5*35B*A3B*"),
    (re.compile(r"qwen3\.?5.*27b|qwen3\.5-27b", re.I), "*Qwen3.5*27B*"),
    (re.compile(r"qwen3\.?5.*a3b|qwen3\.5-a3b", re.I), "*Qwen3.5*A3B*"),
    (re.compile(r"minimax", re.I), "*MiniMax*"),
    (re.compile(r"kimi", re.I), "*Kimi*"),
    # DeepSeek family — prefer Flash/Pro-specific globs before generic
    (re.compile(r"deepseek.?v4.?flash|ds4.?flash|ds4-gguf|v4-flash", re.I), "*DeepSeek*Flash*"),
    (re.compile(r"deepseek.?v4.?pro|v4-pro", re.I), "*DeepSeek*Pro*"),
    (re.compile(r"deepseek|ds4|dwarfstar", re.I), "*DeepSeek*"),
    (re.compile(r"ds4-gguf", re.I), "*ds4-gguf*"),
    (re.compile(r"ds4-flash-q4", re.I), "*ds4-flash*"),
    (re.compile(r"gemma", re.I), "*Gemma*"),
]


def content_hash(obj: dict[str, Any]) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def expand_corpus(glob_file: Path) -> list[Path]:
    hits: list[Path] = []
    for line in glob_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # graph-indexer py is not a research note — skip for note parse
        if line.endswith("param_researcher.py"):
            continue
        pattern = str(HOME / line) if not line.startswith("/") else line
        for m in globmod.glob(pattern):
            p = Path(m)
            if p.is_file() and p.suffix.lower() in {".md", ".markdown"}:
                hits.append(p)
    # unique, stable
    return sorted(set(hits), key=lambda p: str(p))


def _slug_from_path(path: Path) -> str:
    rel = str(path)
    for prefix in (str(HOME / "wiki/atlas/"), str(HOME / "wiki/"), str(HOME) + "/"):
        if rel.startswith(prefix):
            rel = rel[len(prefix) :]
            break
    rel = re.sub(r"\.md$", "", rel, flags=re.I)
    return "wiki:" + re.sub(r"[^a-zA-Z0-9]+", "-", rel).strip("-").lower()


def _title_from_md(text: str, path: Path) -> str:
    for line in text.splitlines()[:40]:
        m = re.match(r"^#\s+(.+)$", line.strip())
        if m:
            return m.group(1).strip()
    return path.stem.replace("-", " ")


def _infer_applies_to(text: str, path: Path) -> dict[str, Any]:
    blob = f"{path.name}\n{text[:8000]}"
    accels: list[str] = []
    for needle, name in _ACCEL_HINTS.items():
        if needle in blob.lower() and name not in accels:
            accels.append(name)
    engines: list[str] = []
    for needle, name in _ENGINE_HINTS.items():
        if needle in blob.lower() and name not in engines:
            engines.append(name)
    globs: list[str] = []
    for pat, g in _MODEL_GLOBS:
        if pat.search(blob) and g not in globs:
            globs.append(g)
    return {
        "model_globs": globs,
        "accel_types": accels,
        "engines": engines,
    }


def _parse_number(cell: str) -> float | int | None:
    s = cell.strip().replace(",", "")
    s = re.sub(r"[*%×x]$", "", s, flags=re.I).strip()
    # strip bold markers
    s = s.replace("**", "").strip()
    # "2:17" wall clock — skip
    if re.match(r"^\d+:\d+", s):
        return None
    # "106.9s (516 tok/s)" → take first number
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        v = float(m.group(1))
        if v.is_integer():
            return int(v)
        return v
    except ValueError:
        return None


def _split_table_row(line: str) -> list[str]:
    line = line.strip()
    if not line.startswith("|"):
        return []
    parts = [c.strip() for c in line.strip("|").split("|")]
    return parts


def _is_separator(cells: list[str]) -> bool:
    if not cells:
        return True
    return all(re.match(r"^:?-+:?$", c.replace(" ", "")) for c in cells if c)


def extract_table_claims(text: str, enrichment_id: str) -> list[dict[str, Any]]:
    """Parse GFM tables; emit measured claims only when numeric cells exist."""
    lines = text.splitlines()
    claims: list[dict[str, Any]] = []
    i = 0
    table_idx = 0
    while i < len(lines):
        header = _split_table_row(lines[i])
        if len(header) < 2:
            i += 1
            continue
        if i + 1 >= len(lines) or not _is_separator(_split_table_row(lines[i + 1])):
            i += 1
            continue
        # consume separator + body
        i += 2
        headers_l = [h.lower() for h in header]
        # detect metric columns
        tps_cols = [
            j
            for j, h in enumerate(headers_l)
            if any(k in h for k in ("tok/s", "tps", "gen tok", "throughput", "speed"))
            and "prefill" not in h
            and "accept" not in h
        ]
        prefill_cols = [j for j, h in enumerate(headers_l) if "prefill" in h]
        ctx_hint = None
        # section heading above table may have 55K / context
        for back in range(max(0, i - 8), i):
            hm = re.search(r"(\d+)\s*[Kk]\s*(?:tokens?|ctx|context)?", lines[back])
            if hm:
                ctx_hint = int(hm.group(1)) * (1000 if hm.group(0).lower().find("k") >= 0 else 1)
                if "k" in lines[back][hm.start() : hm.end()].lower():
                    ctx_hint = int(hm.group(1)) * 1000
                break

        row_n = 0
        while i < len(lines):
            cells = _split_table_row(lines[i])
            if not cells or len(cells) != len(header):
                break
            if _is_separator(cells):
                i += 1
                continue
            row_n += 1
            config = cells[0].replace("**", "").strip()
            metrics: dict[str, Any] = {}
            for j in tps_cols:
                if j < len(cells):
                    n = _parse_number(cells[j])
                    if n is not None:
                        metrics["agg_tps"] = n
                        break
            for j in prefill_cols:
                if j < len(cells):
                    # prefer parenthetical tok/s in prefill cell if present
                    m = re.search(r"(\d+(?:\.\d+)?)\s*tok/s", cells[j], re.I)
                    if m:
                        metrics.setdefault("prefill_tps", float(m.group(1)))
                    n = _parse_number(cells[j])
                    if n is not None and "s" in cells[j].lower():
                        metrics.setdefault("prefill_s", n)
            if ctx_hint:
                metrics["context_len"] = ctx_hint
            # accept rate
            for j, h in enumerate(headers_l):
                if "accept" in h and j < len(cells):
                    n = _parse_number(cells[j])
                    if n is not None:
                        metrics["acceptance_pct"] = n

            if not metrics:
                i += 1
                continue

            claim_id = f"claim:{enrichment_id.split('wiki:')[-1][:40]}-t{table_idx}-r{row_n}"
            text_parts = [config] if config else []
            if "agg_tps" in metrics:
                text_parts.append(f"{metrics['agg_tps']} tok/s")
            if metrics.get("context_len"):
                text_parts.append(f"@ {metrics['context_len']} context")
            claims.append(
                {
                    "id": claim_id,
                    "text": " — ".join(text_parts) if text_parts else f"table row {row_n}",
                    "metrics": metrics,
                    "hardware": None,
                    "date": None,
                    "confidence": "measured",
                    "source_span": f"table[{table_idx}] row[{row_n}] config={config!r}",
                }
            )
            i += 1
        table_idx += 1
        if len(claims) >= 12:
            break
    return claims[:12]


def extract_prose_pointers(text: str, enrichment_id: str) -> list[dict[str, Any]]:
    """Non-numeric bullets that still help search (narrative, no fake metrics)."""
    claims: list[dict[str, Any]] = []
    # Key finding lines
    for n, line in enumerate(text.splitlines()):
        s = line.strip()
        if s.startswith("**Key finding:**") or s.startswith("**TQ's value"):
            claims.append(
                {
                    "id": f"claim:{enrichment_id.split('wiki:')[-1][:40]}-p{n}",
                    "text": re.sub(r"^\*\*[^*]+\*\*:\s*", "", s)[:400],
                    "metrics": {},
                    "confidence": "narrative",
                    "source_span": f"line:{n}",
                }
            )
        if len(claims) >= 4:
            break
    return claims


def note_to_record(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if len(text.strip()) < 40:
        return None

    eid = _slug_from_path(path)
    title = _title_from_md(text, path)
    applies = _infer_applies_to(text, path)
    claims = extract_table_claims(text, eid)
    if len(claims) < 12:
        claims.extend(extract_prose_pointers(text, eid))
        claims = claims[:12]

    conf = "pointer"
    if any(c.get("confidence") == "measured" for c in claims):
        conf = "measured"
    elif claims:
        conf = "narrative"

    # BM25 tokens from title + applies + claim metrics (max ~40 claim tokens)
    tokens: list[str] = [title.lower(), path.stem.replace("-", " ")]
    for g in applies.get("model_globs") or []:
        tokens.append(g.replace("*", " ").lower())
    for a in applies.get("accel_types") or []:
        tokens.append(str(a))
        tokens.append(f"accel:{a}")
    for e in applies.get("engines") or []:
        tokens.append(str(e))
    claim_toks: list[str] = []
    for c in claims:
        m = c.get("metrics") or {}
        if m.get("agg_tps") is not None:
            claim_toks.append(f"{m['agg_tps']} tok/s")
        if m.get("context_len"):
            claim_toks.append(f"{m['context_len']} context")
        for w in re.findall(r"[a-zA-Z0-9./+-]+", (c.get("text") or "").lower())[:8]:
            claim_toks.append(w)
    tokens.extend(claim_toks[:40])
    search_text = re.sub(r"\s+", " ", " ".join(tokens)).strip().lower()

    narrative_bits = [f"Research note: {title}."]
    measured = [c for c in claims if c.get("confidence") == "measured"][:3]
    for c in measured:
        narrative_bits.append(c.get("text", ""))
    search_narrative = " ".join(x for x in narrative_bits if x)[:800]

    try:
        rel_path = "~/" + str(path.relative_to(HOME))
    except ValueError:
        rel_path = str(path)

    rec: dict[str, Any] = {
        "id": eid,
        "kind": "research_note",
        "enrichment_id": eid,
        "path": rel_path,
        "title": title,
        "applies_to": applies,
        "claims": claims,
        "confidence": conf,
        "search_text": search_text,
        "search_narrative": search_narrative,
        "source": {"type": "wiki", "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()},
        "sources_used": ["wiki"],
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    rec["content_hash"] = content_hash(
        {k: v for k, v in rec.items() if k not in ("created", "content_hash")}
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(prog="ingest_research")
    ap.add_argument("--glob", type=Path, default=DEFAULT_GLOB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json", action="store_true", help="print summary JSON")
    args = ap.parse_args()

    paths = expand_corpus(args.glob)
    records: list[dict[str, Any]] = []
    for p in paths:
        rec = note_to_record(p)
        if rec:
            records.append(rec)

    args.out.write_text("".join(json.dumps(r, default=str) + "\n" for r in records))
    measured = sum(1 for r in records if r.get("confidence") == "measured")
    claims = sum(len(r.get("claims") or []) for r in records)
    summary = {
        "files_scanned": len(paths),
        "notes_written": len(records),
        "measured_notes": measured,
        "claims_total": claims,
        "out": str(args.out),
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"research ingest: {summary['notes_written']} notes "
            f"({summary['measured_notes']} measured) from {summary['files_scanned']} files → {args.out}"
        )
        print(f"  claims_total={claims}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
