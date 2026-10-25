#!/usr/bin/env python3
"""RRF fusion of BM25 + Qwen3-VL embedding ranks for model-kb search.

Load once per process:
    from hybrid_search import HybridModelKb
    hk = HybridModelKb()            # falls back to BM25-only if vectors missing
    hits = hk.search("dflash 2 acceptance", top_k=5)

Fail-closed policy: vectors absent / stale id-map mismatch / embed server down →
BM25-only with a stderr note. Never a wrong-but-silent ranking.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from build_bm25 import load_index, load_records, search as bm25_search  # noqa: E402

RRF_K = 60
EMBED_DIMS = 256


def _rec_key(rec: dict) -> str:
    return str(rec.get("recipe_id") or rec.get("id") or rec.get("title"))


class HybridModelKb:
    def __init__(self, index_path: Path | None = None):
        self.idx = load_index(index_path or (ROOT / "bm25_index" / "index.pkl"))
        self.records = load_records(ROOT / "records.jsonl")
        self.vectors = None
        self._server = None
        vec_path = ROOT / "bm25_index" / "vectors.npy"
        ids_path = ROOT / "bm25_index" / "vectors.ids.json"
        meta_path = ROOT / "bm25_index" / "vectors.meta.json"
        if vec_path.exists() and ids_path.exists():
            ids = json.loads(ids_path.read_text())
            rec_ids = [str(r.get("id") or r.get("recipe_id") or i) for i, r in enumerate(self.records)]
            fresh = ids == rec_ids
            if fresh and meta_path.exists():
                import hashlib
                # Content check: identical ids with reworded records must NOT pass
                meta_sha = json.loads(meta_path.read_text()).get("records_sha256")
                live_sha = hashlib.sha256((ROOT / "records.jsonl").read_bytes()).hexdigest()
                if meta_sha and meta_sha != live_sha:
                    fresh = False
                    print("[hybrid] vectors stale (content changed) — BM25 only", file=sys.stderr)
            if fresh:
                self.vectors = np.load(vec_path)
            else:
                if meta_path.exists() and ids == rec_ids:
                    pass  # message printed above
                else:
                    print("[hybrid] vectors id-map stale — BM25 only", file=sys.stderr)
        else:
            print("[hybrid] no vectors.npy — BM25 only (run embed_model_kb.py)", file=sys.stderr)

    def _embed_query(self, query: str) -> np.ndarray | None:
        if self.vectors is None:
            return None
        try:
            if self._server is None:
                # Raises EmbedServerUnavailable (caught below) — no localhost guess.
                from lib.embed_discovery import discover_embed_server
                self._server = discover_embed_server()
            import requests
            resp = requests.post(
                f"{self._server}/v1/embeddings",
                json={"input": [query], "dimensions": EMBED_DIMS},
                timeout=30,
            )
            resp.raise_for_status()
            v = np.asarray(resp.json()["data"][0]["embedding"], dtype=np.float32)[:EMBED_DIMS]
            return v / max(float(np.linalg.norm(v)), 1e-12)
        except Exception as e:
            print(f"[hybrid] embed server error ({e}) — BM25 only this query", file=sys.stderr)
            return None

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        bm25_hits = bm25_search(self.idx, query, top_k=max(top_k * 5, 50))
        qv = self._embed_query(query)
        if qv is None:
            return bm25_hits[:top_k]

        # BM25 ranks keyed by record key
        bm25_rank = {}
        bm25_by_key = {}
        for rank, hit in enumerate(bm25_hits):
            key = _rec_key({"recipe_id": hit.get("recipe_id"), "id": hit.get("id"), "title": hit.get("title")})
            bm25_rank[key] = rank
            bm25_by_key[key] = hit

        # Embedding ranks keyed the same way (vectors rows == records order)
        sims = self.vectors @ qv
        emb_order = np.argsort(-sims)[: max(top_k * 5, 50)]
        emb_rank = {}
        for rank, row in enumerate(emb_order):
            emb_rank[_rec_key(self.records[int(row)])] = rank

        keys = set(bm25_rank) | set(emb_rank)
        fused = sorted(
            keys,
            key=lambda k: -(1.0 / (RRF_K + bm25_rank.get(k, 10_000)) + 1.0 / (RRF_K + emb_rank.get(k, 10_000))),
        )
        out = []
        for k in fused[:top_k]:
            hit = dict(bm25_by_key.get(k) or {})
            if not hit:
                rec = next((r for r in self.records if _rec_key(r) == k), {})
                hit = {
                    "id": rec.get("id"),
                    "name": (rec.get("artifact") or {}).get("name") or rec.get("title"),
                    "title": rec.get("title"),
                    "recipe_id": rec.get("recipe_id"),
                    "kind": rec.get("kind"),
                }
            hit["score"] = 1.0 / (RRF_K + bm25_rank.get(k, 10_000)) + 1.0 / (RRF_K + emb_rank.get(k, 10_000))
            hit["fused"] = True
            out.append(hit)
        return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()
    hk = HybridModelKb()
    for i, hit in enumerate(hk.search(args.query, top_k=args.top_k), 1):
        label = hit.get("recipe_id") or hit.get("title") or hit.get("name") or hit.get("id")
        print(f"#{i} {hit.get('score', 0):.4f} {label} [{hit.get('kind') or ''}]")
