#!/usr/bin/env python3
"""Embed model-kb records.jsonl via the Qwen3-VL embedding server → vectors.npy.

Usage: python3 embed_model_kb.py [--dims 256] [--records records.jsonl] [--out vectors.npy]

Narrative field = search_narrative if present else search_text (the BM25 target stays
separate by design — see qwen-bm25-hybrid skill). Re-run after any records change;
the id map (ids.json) must stay in lockstep with vectors.npy rows.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

# The engine's own directory, as pipeline.py resolves it. The engine moved from
# ~/infra/model-kb to ~/.r1o/model-kb, and the old home-relative defaults
# failed `pipeline.py --release` (FileNotFoundError on the dead path) inside the
# desktop build:signed. Defaults follow the engine wherever it is installed.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lib.embed_discovery import EmbedServerUnavailable, discover_embed_server  # noqa: E402

BATCH = 8
# Exit code when no embedding server proves itself: nothing was written, the
# previous (vectors, ids, meta) generation is untouched.
EXIT_NO_SERVER = 2


def narrative_of(rec: dict) -> str:
    text = (
        rec.get("search_narrative")
        or rec.get("search_text")
        or " ".join(str(rec.get(k) or "") for k in ("title", "description", "recipe_id"))
    ).strip()
    # Qwen3-VL embed server stalls on multi-KB wiki dumps; cap for serve-search.
    return text[:2000]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(ROOT / "records.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "bm25_index" / "vectors.npy"))
    ap.add_argument("--dims", type=int, default=256)
    ap.add_argument("--check", action="store_true", help="Discover the embed server, print it, write nothing")
    args = ap.parse_args()

    trace: list[str] = []
    try:
        server = discover_embed_server(trace=trace)
    except EmbedServerUnavailable as e:
        print(f"FAIL: embed server unavailable — vectors NOT rebuilt: {e}", file=sys.stderr)
        return EXIT_NO_SERVER
    if args.check:
        print(server)
        return 0
    # Read the records once: the vectors, the id map and the sha all describe
    # exactly these bytes, even if another writer replaces the file meanwhile.
    records_bytes = Path(args.records).read_bytes()
    recs = [json.loads(line) for line in records_bytes.decode().splitlines() if line.strip()]
    ids = [str(r.get("id") or r.get("recipe_id") or i) for i, r in enumerate(recs)]
    texts = [narrative_of(r) for r in recs]
    print(f"[embed] {len(texts)} records @ {server} dims={args.dims}", file=sys.stderr)

    vecs = np.zeros((len(texts), args.dims), dtype=np.float32)
    tic = time.perf_counter()

    def embed_batch(batch: list[str]) -> list[list[float]]:
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                resp = requests.post(
                    f"{server}/v1/embeddings",
                    json={"input": batch, "dimensions": args.dims},
                    timeout=180,
                )
                resp.raise_for_status()
                return [item["embedding"] for item in resp.json()["data"]]
            except Exception as e:  # noqa: BLE001 — retry timeouts from the VL server
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        if len(batch) > 1:
            # Stall is usually one fat note; split so the rest of the corpus still lands.
            out: list[list[float]] = []
            for one in batch:
                out.extend(embed_batch([one]))
            return out
        raise RuntimeError(f"embed failed after retries: {last_err}") from last_err

    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        data = embed_batch(batch)
        for j, emb in enumerate(data):
            # Server may ignore `dimensions` (full 4096d) — Matryoshka slice client-side
            vecs[i + j] = np.asarray(emb, dtype=np.float32)[: args.dims]
        if (i // BATCH) % 10 == 0:
            print(f"[embed] {i + len(batch)}/{len(texts)} ({(i+len(batch))/max(time.perf_counter()-tic,0.01):.0f}/s)", file=sys.stderr)

    # L2-normalize so dot == cosine
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-12)

    # Atomic pair-write: tmp files then rename — a crash mid-write must never
    # leave a mixed-generation (vectors, ids) pair that still passes the check.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import hashlib
    records_sha = hashlib.sha256(records_bytes).hexdigest()
    if hashlib.sha256(Path(args.records).read_bytes()).hexdigest() != records_sha:
        print("FAIL: records.jsonl changed while embedding — vectors NOT written; re-run", file=sys.stderr)
        return 1
    np.save(out.with_suffix(".tmp.npy"), vecs)
    (out.parent / "vectors.ids.tmp.json").write_text(json.dumps(ids))
    (out.parent / "vectors.meta.tmp.json").write_text(
        json.dumps({"records_sha256": records_sha, "rows": len(ids), "dims": args.dims, "server": server})
    )
    out.with_suffix(".tmp.npy").rename(out)
    (out.parent / "vectors.ids.tmp.json").rename(out.parent / "vectors.ids.json")
    (out.parent / "vectors.meta.tmp.json").rename(out.parent / "vectors.meta.json")
    print(f"[done] {vecs.shape} → {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
