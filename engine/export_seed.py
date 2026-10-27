#!/usr/bin/env python3
"""export_seed.py — anonymize live records.jsonl for the shipped DMG seed.

Never copy ~/infra/model-kb/records.jsonl into the app. The seed is a derived
view: no cluster hostnames, no /Users/ma, no node inventories, no NAS paths.

Usage:
  python3 export_seed.py
  python3 export_seed.py --out ~/Projects/model-kb-kit/data/records.seed.jsonl
  python3 export_seed.py --check-only   # leak-scan an existing seed
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = Path.home() / "Projects" / "model-kb-kit" / "data" / "records.seed.jsonl"

# Same cluster-id shape as desktop/scripts/scrub-seed.mjs (must not ship).
HOST_RE = re.compile(r"\b(?:mini2|minipro1|m3u[1-4]|marmac|m4m1|mbk)\b", re.I)
# Dict keys like mlx_versions.hub / mlx_versions.m3u1 — exact token only (not "huggingface hub").
HOST_KEY_RE = re.compile(r"^(?:hub|mini2|minipro1|m3u[1-4]|marmac|m4m1|mbk)$", re.I)
HOME_RE = re.compile(r"/Users/ma\b")
EMAIL_RE = re.compile(r"\b(?:maigcg@gmail|mario@fib0|m@fib0)\b", re.I)


def portable_path(raw: str | None) -> str | None:
    if not raw:
        return None
    s = HOME_RE.sub("~", str(raw)).replace("\\", "/")
    if s.startswith("~/Models/") or s == "~/Models":
        return s
    marker = "/Models/"
    if marker in s:
        return "~/Models/" + s.split(marker, 1)[1]
    return None


def scrub_str(s: str) -> str:
    s = HOST_RE.sub("a-node", s)
    s = HOME_RE.sub("~", s)
    s = EMAIL_RE.sub("redacted", s)
    return s


def scrub(obj: object, *, key: str | None = None) -> object:
    if isinstance(obj, str):
        if key == "path":
            return portable_path(obj)
        return scrub_str(obj)
    if isinstance(obj, list):
        if key == "nodes":
            return []
        return [scrub(x) for x in obj]
    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            dk = "a-node" if HOST_KEY_RE.match(str(k)) else k
            if k in {"asmi_path", "inventory_model_id"}:
                out[dk] = None
            elif k == "nodes":
                out[dk] = []
            elif k == "path":
                out[dk] = portable_path(v) if isinstance(v, str) else None
            else:
                out[dk] = scrub(v, key=k)
        return out
    return obj


def leak_scan(blob: str) -> list[str]:
    hits: list[str] = []
    if HOST_RE.search(blob):
        hits.append("cluster hostname")
    if "/Users/ma" in blob:
        hits.append("/Users/ma")
    if EMAIL_RE.search(blob):
        hits.append("author email")
    return hits


def export(src: Path, dest: Path) -> int:
    rows: list[str] = []
    for line in src.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        clean = scrub(rec)
        rows.append(json.dumps(clean, ensure_ascii=False, separators=(",", ":")))
    blob = "\n".join(rows) + ("\n" if rows else "")
    leaks = leak_scan(blob)
    if leaks:
        print(f"FAIL: seed still contains {leaks} — refusing to write {dest}", file=sys.stderr)
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(blob)
    tmp.replace(dest)
    print(f"wrote {len(rows)} anonymized records → {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=ROOT / "records.jsonl")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if args.check_only:
        target = args.out if args.out.exists() else args.records
        leaks = leak_scan(target.read_text())
        if leaks:
            print(f"FAIL: {target} leaks {leaks}")
            return 1
        print(f"OK: {target} has no cluster/home/email leaks")
        return 0
    if not args.records.exists():
        print(f"FAIL: records missing at {args.records}")
        return 2
    return export(args.records, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
