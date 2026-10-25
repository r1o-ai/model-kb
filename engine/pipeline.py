#!/usr/bin/env python3
"""pipeline.py — canonical full rebuild for model-kb.

Order is load-bearing (ingest_serve_configs.py clobbers records.jsonl):

  lock → embed preflight → snapshot → backup → fill-paths → recipes → cloud
  → research → family → derivative → hf cards → merge enrichments → join
  → transfer → BM25 → embed → verify

Usage:
  python3 pipeline.py                 # full, fail closed
  python3 pipeline.py --release       # same + fatal network + seed export
  python3 pipeline.py --allow-partial # network failures become warnings (embed never does)
  python3 pipeline.py --research-only # research → merge → join → BM25 → embed → verify (no recipes clobber)
  python3 pipeline.py --skip-embed --skip-fill-paths

Records and vectors are one generation. records.jsonl rows must equal
vectors.npy rows and vectors.ids.json (verify.py; hybrid search falls back to
BM25-only when they differ). So:
  * the embed server is discovered BEFORE anything is written; none → exit 3,
    the corpus untouched;
  * if any step fails before a new aligned generation is complete, the
    previous records.jsonl + bm25_index/ are restored and the run exits non-zero;
  * one ingest at a time: an exclusive lock on .ingest.lock (two crons share
    this corpus).
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
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


# The files that make one searchable generation. Restored together.
GENERATION = ("records.jsonl", "bm25_index/index.pkl", "bm25_index/vectors.npy",
              "bm25_index/vectors.ids.json", "bm25_index/vectors.meta.json")
EXIT_NO_EMBED = 3
EXIT_LOCKED = 75  # EX_TEMPFAIL


def acquire_lock(timeout_s: float):
    """Exclusive ingest lock; waits up to timeout_s for the other writer."""
    fh = open(ROOT / ".ingest.lock", "a+")
    deadline = time.monotonic() + timeout_s
    announced = False
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fh.seek(0)
            fh.truncate()
            fh.write(f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}\n")
            fh.flush()
            return fh
        except BlockingIOError:
            if time.monotonic() >= deadline:
                fh.seek(0)
                holder = fh.read().strip() or "?"
                fh.close()
                return holder
            if not announced:
                fh.seek(0)
                print(f"waiting for ingest lock (held by {fh.read().strip() or '?'})", flush=True)
                announced = True
            time.sleep(5)


def snapshot_generation() -> Path:
    snap = Path(tempfile.mkdtemp(prefix=".gen-snapshot-", dir=ROOT))
    for rel in GENERATION:
        src = ROOT / rel
        if src.exists():
            dst = snap / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return snap


def restore_generation(snap: Path) -> None:
    for rel in GENERATION:
        src, dst = snap / rel, ROOT / rel
        if src.exists():
            tmp = dst.with_name(dst.name + ".restore-tmp")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        elif dst.exists():
            dst.unlink()  # did not exist in the previous generation
    print(f"ROLLED BACK records.jsonl + bm25_index/ to the previous generation ({snap})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(prog="model-kb-pipeline")
    ap.add_argument("--release", action="store_true", help="Fatal network/embed; export anonymized seed")
    ap.add_argument("--allow-partial", action="store_true", help="Network failures are warnings (embed stays fatal)")
    ap.add_argument("--research-only", action="store_true",
                    help="research → merge → join → BM25 → embed → verify; keeps recipes + existing HF enrichments")
    ap.add_argument("--skip-fill-paths", action="store_true")
    ap.add_argument("--skip-research", action="store_true")
    ap.add_argument("--skip-network", action="store_true", help="Skip cloud/family/hf (keep existing enrichments)")
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--lock-timeout", type=float, default=float(os.getenv("MODEL_KB_LOCK_TIMEOUT_S", "7200")),
                    help="Seconds to wait for another ingest to finish (default 7200)")
    args = ap.parse_args()
    if args.research_only:
        if args.skip_research:
            ap.error("--research-only and --skip-research contradict")
        args.skip_fill_paths = True
        args.skip_network = True

    records = ROOT / "records.jsonl"
    net_required = args.release or not args.allow_partial
    if args.skip_network:
        net_required = False

    lock = acquire_lock(args.lock_timeout)
    if not hasattr(lock, "fileno"):
        print(f"FAIL: another ingest still holds {ROOT / '.ingest.lock'} ({lock}) after {args.lock_timeout:.0f}s")
        return EXIT_LOCKED

    # Preflight: a run that cannot embed must not change records (it would leave
    # records ≠ vectors until some later run happens to find a server).
    if not args.skip_embed:
        print("+ embed preflight", flush=True)
        if subprocess.call([PY, str(ROOT / "embed_model_kb.py"), "--check"], cwd=str(ROOT)) != 0:
            print("FAIL: no embedding server — nothing written; records + vectors stay the previous aligned pair")
            return EXIT_NO_EMBED

    baseline = counts(records)
    print("BASELINE:", dict(baseline), "total", sum(baseline.values()))
    snap = snapshot_generation()
    bak = backup(records)
    complete = False
    try:
        if not args.skip_fill_paths:
            run("fill_recipe_paths.py")

        # 1) recipes CLOBBERS — must be first
        if not args.research_only:
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

        if args.skip_embed:
            print("WARN --skip-embed: vectors not rebuilt; hybrid search is BM25-only until the next embed")
        else:
            # Always required: embedding is the records↔vectors contract, not
            # optional network enrichment, so --allow-partial does not soften it.
            run("embed_model_kb.py")

        after = counts(records)
        print("AFTER:", dict(after), "total", sum(after.values()))
        if baseline:
            for kind, n in baseline.items():
                if n and after.get(kind, 0) == 0:
                    print(f"FAIL: kind {kind} dropped {n} → 0 (ordering trap)")
                    raise RuntimeError(f"kind {kind} zeroed")
        complete = True
    except (subprocess.CalledProcessError, RuntimeError) as e:
        print(f"FAIL: {e}", flush=True)
        return 1
    finally:
        if not complete:
            restore_generation(snap)
        shutil.rmtree(snap, ignore_errors=True)

    if not args.skip_verify:
        verify_args = []
        if args.skip_embed:
            verify_args.append("--skip-embed")
        if args.release:
            verify_args.append("--strict-gated")
        # The generation is aligned and kept; a verify failure (e.g. a recipe gap)
        # still fails the run so the scheduler reports it.
        if not run("verify.py", *verify_args, required=False):
            print("FAIL: verify.py reported problems (records + vectors are aligned and kept)")
            return 1

    if args.release:
        run("export_seed.py")

    print("pipeline OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
