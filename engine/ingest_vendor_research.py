#!/usr/bin/env python3
"""Ingest vendor-research JSONL findings into model-kb records.jsonl.

  READ    vendor-research/*.jsonl (agent findings, cloud_context shape)
  VALIDATE required fields, kind, provenance (sources_used non-empty)
  MERGE   by id into records.jsonl — never drops records it didn't produce
  INDEX   rebuild BM25 with --rebuild-index

Fail-closed: no findings files, or zero valid records → exit 1, no write.

Usage:
  python3 ingest_vendor_research.py                    # all vendor files
  python3 ingest_vendor_research.py --vendor mistral   # one file
  python3 ingest_vendor_research.py --dry-run
  python3 ingest_vendor_research.py --rebuild-index
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESEARCH_DIR = ROOT / "vendor-research"
DEFAULT_OUT = ROOT / "records.jsonl"

REQUIRED_FIELDS = ("id", "kind", "artifact", "serve", "sources_used", "description")


def content_hash(obj: dict[str, Any]) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"WARN {path.name}:{i} invalid JSON ({e}) — skipped", file=sys.stderr)
    return out


def validate(rec: dict[str, Any]) -> str | None:
    """Return rejection reason, or None if valid."""
    for f in REQUIRED_FIELDS:
        if f not in rec or rec[f] in (None, "", [], {}):
            return f"missing/empty required field '{f}'"
    if rec["kind"] != "cloud_context":
        return f"kind={rec['kind']!r} (vendor research must be cloud_context)"
    if not rec["id"].startswith("cloud:"):
        return f"id {rec['id']!r} must start with 'cloud:'"
    if not isinstance(rec["sources_used"], list) or not rec["sources_used"]:
        return "sources_used must be a non-empty list of URLs (provenance required)"
    # provenance for numeric specs: numbers with no sources = invented
    serve = rec.get("serve") or {}
    for k in ("max_context", "max_output_tokens"):
        v = serve.get(k)
        if v is not None and not isinstance(v, int):
            return f"serve.{k} must be int or null, got {type(v).__name__}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(prog="ingest_vendor_research")
    ap.add_argument("--vendor", help="single vendor slug (file stem in vendor-research/)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.vendor:
        files = [RESEARCH_DIR / f"{args.vendor}.jsonl"]
        if not files[0].exists():
            print(f"FAIL: {files[0]} not found", file=sys.stderr)
            return 1
    else:
        files = sorted(RESEARCH_DIR.glob("*.jsonl"))
    if not files:
        print(f"FAIL: no findings files in {RESEARCH_DIR} — nothing to ingest (fail closed)", file=sys.stderr)
        return 1

    incoming: dict[str, dict[str, Any]] = {}
    rejected = 0
    for f in files:
        recs = load_jsonl(f)
        ok = 0
        for rec in recs:
            reason = validate(rec)
            if reason:
                print(f"REJECT {f.name} {rec.get('id', '<no id>')}: {reason}", file=sys.stderr)
                rejected += 1
                continue
            rec.setdefault("source", {})
            rec["source"].setdefault("kind", "vendor_research")
            rec["content_hash"] = content_hash({k: v for k, v in rec.items() if k != "content_hash"})
            incoming[rec["id"]] = rec
            ok += 1
        print(f"{f.name}: {ok}/{len(recs)} valid")

    if not incoming:
        print("FAIL: zero valid records across all findings — refusing to write (fail closed)", file=sys.stderr)
        return 1

    existing = load_jsonl(args.out) if args.out.exists() else []
    by_id: dict[str, dict[str, Any]] = {r["id"]: r for r in existing if r.get("id")}
    new_ids = [i for i in incoming if i not in by_id]
    updated_ids = [i for i in incoming if i in by_id and by_id[i].get("content_hash") != incoming[i]["content_hash"]]
    by_id.update(incoming)

    print(f"merge: {len(new_ids)} new, {len(updated_ids)} updated, {rejected} rejected, total {len(by_id)} (was {len(existing)})")

    if args.dry_run:
        for i in new_ids[:10]:
            print("  NEW", i)
        return 0

    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in by_id.values()) + "\n")
    print(f"wrote {args.out}")

    if args.rebuild_index:
        import subprocess

        subprocess.check_call([sys.executable, str(ROOT / "build_bm25.py"), "--rebuild"], cwd=str(ROOT))
        print("bm25 index rebuilt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
