#!/usr/bin/env python3
"""pipeline.py — full model-kb rebuild: serve-configs + research + join + BM25.

Usage:
  python3 pipeline.py              # full ingest + index
  python3 pipeline.py --skip-index
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(script: str, *args: str) -> None:
    cmd = [PY, str(ROOT / script), *args]
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(prog="model-kb-pipeline")
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--skip-research", action="store_true")
    args = ap.parse_args()

    # 1) recipes from serve-configs (rewrites recipe:* rows)
    run("ingest_serve_configs.py", "--out", str(ROOT / "records.jsonl"))

    # 1b) cloud model context packs (max_context, options, quota) — not wiki
    try:
        run("ingest_cloud_context.py", "--out", str(ROOT / "records.jsonl"))
    except subprocess.CalledProcessError as e:
        print(f"WARN ingest_cloud_context failed (continuing): {e}")

    if not args.skip_research:
        # 2) research notes (wiki)
        run("ingest_research.py", "--out", str(ROOT / "enrichments.jsonl"))
        # 2b) HF/Unsloth model cards (quality tables + quant inventory)
        try:
            run("ingest_hf_cards.py", "--out", str(ROOT / "enrichments_hf.jsonl"))
        except subprocess.CalledProcessError as e:
            print(f"WARN ingest_hf_cards failed (continuing): {e}")
        # 2c) HF family tree levels per base_model filter API
        try:
            run("ingest_hf_family.py", "--out", str(ROOT / "enrichments_hf_family.jsonl"))
        except subprocess.CalledProcessError as e:
            print(f"WARN ingest_hf_family failed (continuing): {e}")
        # merge enrichment streams
        def _merge_enrich() -> None:
            seen: set[str] = set()
            rows: list[str] = []
            for p in (
                ROOT / "enrichments.jsonl",
                ROOT / "enrichments_hf.jsonl",
                ROOT / "enrichments_hf_family.jsonl",
            ):
                if not p.exists():
                    continue
                for line in p.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        import json as _json

                        r = _json.loads(line)
                    except Exception:
                        continue
                    eid = r.get("id") or r.get("enrichment_id")
                    if not eid or eid in seen:
                        continue
                    seen.add(eid)
                    rows.append(line)
            (ROOT / "enrichments.jsonl").write_text("\n".join(rows) + ("\n" if rows else ""))
            print(f"merged enrichments: {len(rows)}")

        _merge_enrich()
        # 3) join onto recipes + merge enrichment rows into records
        run(
            "join_enrichments.py",
            "--recipes",
            str(ROOT / "records.jsonl"),
            "--enrich",
            str(ROOT / "enrichments.jsonl"),
            "--out",
            str(ROOT / "records.jsonl"),
        )

    # 4) codify accel transfer (dflash/mtp/…) across family quant/finetune
    try:
        run("enrich_optimization_transfer.py", "--recipes", str(ROOT / "records.jsonl"))
    except subprocess.CalledProcessError as e:
        print(f"WARN enrich_optimization_transfer failed (continuing): {e}")

    if not args.skip_index:
        run("build_bm25.py", "--rebuild")

    print("pipeline OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
