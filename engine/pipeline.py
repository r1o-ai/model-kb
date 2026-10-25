#!/usr/bin/env python3
"""pipeline.py — canonical full rebuild for model-kb.

Order is load-bearing (ingest_serve_configs.py clobbers records.jsonl):

  backup → fill-paths → recipes → cloud → research → family → hf cards
  → merge enrichments → join → transfer → BM25 → embed → verify

Usage:
  python3 pipeline.py                 # full, fail closed
  python3 pipeline.py --release       # same + fatal network + seed export
  python3 pipeline.py --allow-partial # network/embed failures become warnings
  python3 pipeline.py --skip-embed --skip-fill-paths
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(script: str, *args: str, required: bool = True) -> bool:
    cmd = [PY, str(ROOT / script), *args]
    print("+", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd, cwd=str(ROOT))
        return True
    except subprocess.CalledProcessError as e:
        if required:
            raise
        print(f"WARN {script} failed (continuing): {e}", flush=True)
        return False


def backup(records: Path) -> Path | None:
    if not records.exists():
        print("no existing records.jsonl — first ingest, skip backup")
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dest = records.with_name(f"records.jsonl.bak-{stamp}")
    shutil.copy2(records, dest)
    print(f"backup → {dest}")
    return dest


def counts(path: Path) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    if not path.exists():
        return c
    for line in path.open():
        if line.strip():
            c[json.loads(line).get("kind")] += 1
    return c


def merge_enrichments() -> int:
    seen: set[str] = set()
    rows: list[str] = []
    for p in (
        ROOT / "enrichments.jsonl",
        ROOT / "enrichments_hf_family.jsonl",
        ROOT / "enrichments_hf_derivative.jsonl",
        ROOT / "enrichments_hf.jsonl",
    ):
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
            rows.append(line)
    (ROOT / "enrichments.jsonl").write_text("\n".join(rows) + ("\n" if rows else ""))
    print(f"merged enrichments: {len(rows)}")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(prog="model-kb-pipeline")
    ap.add_argument("--release", action="store_true", help="Fatal network/embed; export anonymized seed")
    ap.add_argument("--allow-partial", action="store_true", help="Network/embed failures are warnings")
    ap.add_argument("--skip-fill-paths", action="store_true")
    ap.add_argument("--skip-research", action="store_true")
    ap.add_argument("--skip-network", action="store_true", help="Skip cloud/family/hf (keep existing enrichments)")
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    records = ROOT / "records.jsonl"
    net_required = args.release or not args.allow_partial
    if args.skip_network:
        net_required = False

    baseline = counts(records)
    print("BASELINE:", dict(baseline), "total", sum(baseline.values()))
    bak = backup(records)

    if not args.skip_fill_paths:
        run("fill_recipe_paths.py")

    # 1) recipes CLOBBERS — must be first
    run("ingest_serve_configs.py", "--out", str(records))

    # 2) cloud merge
    if not args.skip_network:
        run("ingest_cloud_context.py", "--out", str(records), required=net_required)

    if not args.skip_research:
        run("ingest_research.py", "--out", str(ROOT / "enrichments.jsonl"))
        if not args.skip_network:
            family_ok = run(
                "ingest_hf_family.py",
                "--out",
                str(ROOT / "enrichments_hf_family.jsonl"),
                required=net_required,
            )
            if family_ok:
                audit_args = [
                    "--family",
                    str(ROOT / "enrichments_hf_family.jsonl"),
                    "--out",
                    str(ROOT / "enrichments_hf_derivative.jsonl"),
                ]
                if args.release:
                    audit_args.append("--strict-gated")
                run("ingest_derivative_audit.py", *audit_args, required=net_required)
            run("ingest_hf_cards.py", "--out", str(ROOT / "enrichments_hf.jsonl"), required=net_required)
        merge_enrichments()
        run(
            "join_enrichments.py",
            "--recipes",
            str(records),
            "--enrich",
            str(ROOT / "enrichments.jsonl"),
            "--out",
            str(records),
        )

    run("enrich_optimization_transfer.py", "--recipes", str(records), required=False)

    if not args.skip_index:
        run("build_bm25.py", "--rebuild")

    if not args.skip_embed:
        run("embed_model_kb.py", required=net_required)

    after = counts(records)
    print("AFTER:", dict(after), "total", sum(after.values()))
    if baseline:
        for kind, n in baseline.items():
            if n and after.get(kind, 0) == 0:
                print(f"FAIL: kind {kind} dropped {n} → 0 (ordering trap). Restore {bak}")
                return 1

    if not args.skip_verify:
        verify_args = []
        if args.skip_embed:
            verify_args.append("--skip-embed")
        if args.release:
            verify_args.append("--strict-gated")
        run("verify.py", *verify_args)

    if args.release:
        run("export_seed.py")

    print("pipeline OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
