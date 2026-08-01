#!/usr/bin/env python3
"""build_bm25.py — BM25 index over records.jsonl search_text.

Phase B task 9. Uses cluster-kb BM25Index (rank_bm25).
Embed channel optional — BM25-only is a valid single-channel RRF path.

Usage:
  python3 build_bm25.py
  python3 build_bm25.py --query "4bit dflash qwen 35b"
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lib.bm25_index import BM25Index, HAS_BM25  # noqa: E402

DEFAULT_RECORDS = ROOT / "records.jsonl"
DEFAULT_INDEX = ROOT / "bm25_index" / "index.pkl"


def load_records(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build(records: list[dict]) -> BM25Index:
    if not HAS_BM25:
        raise SystemExit("rank_bm25 missing: pip install rank-bm25")
    idx = BM25Index()
    # Prefer search_text; fall back to narrative
    n = idx.build(
        records,
        id_field="id",
        text_field="search_text",
        text_extractor=lambda r: r.get("search_text") or r.get("search_narrative") or "",
    )
    print(f"BM25 built: {n} docs")
    return idx


def save_index(idx: BM25Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # pickle the serializable state
    state = {
        "corpus": idx.corpus,
        "record_ids": idx.record_ids,
        "records": idx.records,
        "texts": idx.texts,
    }
    path.write_bytes(pickle.dumps(state))
    print(f"saved → {path}")


def load_index(path: Path) -> BM25Index:
    from rank_bm25 import BM25Okapi

    state = pickle.loads(path.read_bytes())
    idx = BM25Index()
    idx.corpus = state["corpus"]
    idx.record_ids = state["record_ids"]
    idx.records = state["records"]
    idx.texts = state["texts"]
    idx.bm25 = BM25Okapi(idx.corpus)
    return idx


def search(idx: BM25Index, query: str, top_k: int = 10) -> list[dict]:
    results = idx.search(query, top_k=top_k)
    out = []
    for r in results:
        rec = r.record or {}
        out.append(
            {
                "id": r.record_id,
                "score": r.score,
                "rank": r.rank,
                "name": (rec.get("artifact") or {}).get("name") or rec.get("title"),
                "title": rec.get("title"),
                "recipe_id": rec.get("recipe_id"),
                "engine": (rec.get("serve") or {}).get("engine"),
                "backend": (rec.get("serve") or {}).get("backend"),
                "quant": (rec.get("quant") or {}).get("label"),
                "accels": [
                    a.get("type")
                    for a in (rec.get("accelerators") or [])
                    if a.get("enabled", True)
                ],
                "kind": rec.get("kind"),
                "search_text": (rec.get("search_text") or "")[:120],
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--query", type=str, default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if not args.records.exists():
        print(f"missing {args.records}; run ingest_serve_configs.py first")
        return 1

    if args.rebuild or not args.index.exists() or args.query is None:
        records = load_records(args.records)
        idx = build(records)
        save_index(idx, args.index)
    else:
        idx = load_index(args.index)

    if args.query:
        hits = search(idx, args.query, top_k=args.top_k)
        for h in hits:
            print(
                f"#{h['rank']} {h['score']:.3f} {h['recipe_id']} "
                f"q={h['quant']} e={h['engine']}/{h['backend']} accel={h['accels']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
