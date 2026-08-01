#!/usr/bin/env python3
"""ingest_serve_configs.py — serve-configs.json v3 → L0 recipe records.

Phase B task 6 of model-kb ARDD plan (2026-07-11).
SoT: ~/.r1o/serve-configs.json (never invent profiles).
Output: records.jsonl (append-safe via rewrite of recipe:* rows only when run standalone).

Usage:
  python3 ingest_serve_configs.py
  python3 ingest_serve_configs.py --out records.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path.home() / ".r1o" / "serve-configs.json"
DEFAULT_OUT = ROOT / "records.jsonl"

# Quant vocabulary (schema §Quant vocabulary)
# Order matters: decimal-bit/bpw BEFORE integer Nbit so "3.6bit" ≠ "6bit".
# Integer patterns are word-bound (no digit immediately before the N).
_QUANT_MAP = [
    # decimal bit / bpw first (Kimi 3.6bit, dynamic 2.7bpw, …)
    (re.compile(r"(?<!\d)(\d+\.\d+)-?bit\b", re.I), None),  # keep full match e.g. 3.6bit
    (re.compile(r"\d+\.\d+bpw[-\w]*", re.I), None),
    (re.compile(r"\d+bpw[-\w]*", re.I), None),
    (re.compile(r"mxfp4", re.I), "mxfp4"),
    (re.compile(r"nvfp4", re.I), "nvfp4"),
    # GGUF / ds4 quants (DeepSeek Flash)
    (re.compile(r"iq2xxs|iq2_xxs", re.I), "iq2xxs"),
    (re.compile(r"iq2|q2_k|q2-k", re.I), "q2"),
    (re.compile(r"q4[_-]?k|q4-imatrix", re.I), "q4"),
    (re.compile(r"mixed[-_]?\d+-\d+", re.I), None),  # keep full match
    (re.compile(r"\bmixed\b", re.I), "mixed"),
    (re.compile(r"\bbf16\b", re.I), "bf16"),
    (re.compile(r"\bfp16\b|\bf16\b", re.I), "fp16"),
    # integer bits — (?<!\d) blocks the "6" inside "3.6bit"
    (re.compile(r"(?<!\d)4-?bit\b|\bq4\b|\bmlx-4bit\b|\bint4\b", re.I), "4bit"),
    (re.compile(r"(?<!\d)8-?bit\b|\bq8\b|\bint8\b", re.I), "8bit"),
    (re.compile(r"(?<!\d)6-?bit\b|\bq6\b", re.I), "6bit"),
    (re.compile(r"(?<!\d)3-?bit\b|\bq3\b", re.I), "3bit"),
    (re.compile(r"(?<!\d)2-?bit\b|\bq2\b", re.I), "2bit"),
]


def normalize_quant(raw: str | None) -> dict[str, Any]:
    raw = (raw or "").strip() or "unknown"
    label = "unknown"
    for pat, canon in _QUANT_MAP:
        m = pat.search(raw)
        if not m:
            continue
        label = m.group(0).lower() if canon is None else canon
        break
    aliases = list({raw.lower(), label} - {""})
    return {"label": label, "raw": raw, "aliases": aliases, "scheme": None, "bits": None, "bpw": None}


def content_hash(obj: dict[str, Any]) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_search_text(rec: dict[str, Any]) -> str:
    """BM25-only keyword field (schema §L1 search_text)."""
    art = rec.get("artifact") or {}
    quant = rec.get("quant") or {}
    serve = rec.get("serve") or {}
    accels = rec.get("accelerators") or []
    hw = rec.get("hardware") or {}
    benches = rec.get("benchmarks") or []

    tokens: list[str] = []
    for v in (art.get("name"), art.get("hf_id"), rec.get("id"), rec.get("recipe_id")):
        if v:
            tokens.append(str(v).lower())
    tokens.append(quant.get("label", "unknown"))
    tokens.append(quant.get("raw", ""))
    tokens.extend(quant.get("aliases") or [])
    tokens.append(str(serve.get("engine") or ""))
    tokens.append(str(serve.get("backend") or ""))
    for a in accels:
        if not a.get("enabled", True):
            continue
        t = a.get("type")
        if t:
            tokens.append(f"accel:{t}")
            tokens.append(str(t))
        draft = a.get("draft_model")
        if draft:
            base = str(draft).rstrip("/").split("/")[-1]
            tokens.append(f"draft:{base.lower()}")
            tokens.append(base.lower())
    for role in rec.get("roles_fit") or []:
        tokens.append(str(role))
    for n in art.get("nodes") or []:
        tokens.append(str(n).lower())
    if art.get("params_total"):
        tokens.append(str(art["params_total"]).lower())
    if art.get("params_active"):
        tokens.append(str(art["params_active"]).lower())
    if art.get("architecture"):
        tokens.append(str(art["architecture"]).lower())
        if "moe" in str(art["architecture"]).lower():
            tokens.append("moe")
        else:
            tokens.append("dense")
    for b in benches:
        tps = b.get("agg_tps")
        if isinstance(tps, (int, float)) and tps > 0:
            tokens.append(f"{int(round(tps))} tok/s")
            tokens.append(f"{tps} tok/s")
    if hw.get("peak_memory_gb"):
        tokens.append(f"{hw['peak_memory_gb']}gb")
        tokens.append("peak_memory")
    if hw.get("min_nodes", 1) and hw.get("min_nodes", 1) > 1:
        tokens.append(f"{hw['min_nodes']}node")
        tokens.append("multinode")
    tokens.append(f"recipe:{rec.get('recipe_id') or rec.get('id')}")
    # collapse whitespace, lower
    text = " ".join(str(t) for t in tokens if t)
    return re.sub(r"\s+", " ", text.lower()).strip()


def build_search_narrative(rec: dict[str, Any], cfg: dict[str, Any]) -> str:
    art = rec.get("artifact") or {}
    serve = rec.get("serve") or {}
    quant = rec.get("quant") or {}
    accels = [a for a in (rec.get("accelerators") or []) if a.get("enabled", True)]
    accel_s = ", ".join(a.get("type", "?") for a in accels) or "none"
    benches = rec.get("benchmarks") or []
    best = max((b.get("agg_tps") or 0 for b in benches), default=0)
    tps_s = f" Measured ~{best:.1f} tok/s." if best else ""
    desc = (cfg.get("description") or "").strip()
    return (
        f"{art.get('name')} ({quant.get('label')}) via {serve.get('engine')}/{serve.get('backend')} "
        f"on min {rec.get('hardware', {}).get('min_nodes', 1)} node(s). "
        f"Accelerators: {accel_s}.{tps_s} "
        f"Recipe `{rec.get('recipe_id')}`. {desc}"
    ).strip()


def benches_from_config(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    br = cfg.get("benchmark_results") or {}
    if not isinstance(br, dict):
        return out
    for key, val in br.items():
        if key in ("pending", "planned_matrix"):
            continue
        if not isinstance(val, dict):
            continue
        if "agg_tps" not in val and "per_req_tps" not in val:
            continue
        # key often "1x" / "4x" / "8x" or "c1"
        conc = None
        m = re.search(r"(\d+)", str(key))
        if m:
            conc = int(m.group(1))
        out.append(
            {
                "id": str(key),
                "concurrency": conc,
                "max_tokens": val.get("max_tokens"),
                "context_len": val.get("context_len"),
                "agg_tps": val.get("agg_tps"),
                "per_req_tps": val.get("per_req_tps"),
                "ttft_ms": val.get("ttft_ms"),
                "wall_s": val.get("wall_s"),
                "source": "serve-configs",
            }
        )
    return out


def recipe_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    name = cfg.get("name") or "unnamed"
    model = cfg.get("model") or {}
    serve = cfg.get("serve") or {}
    hardware = cfg.get("hardware") or {}
    node_profile = hardware.get("node_profile") or {}

    # backend hard ban: only single|jaccl
    backend = serve.get("backend") or "single"
    if backend not in ("single", "jaccl"):
        backend = "single"

    recipe_id = name
    record_id = f"recipe:{recipe_id}"

    quant = normalize_quant(model.get("quant"))
    accels = []
    for a in cfg.get("accelerators") or []:
        if not isinstance(a, dict):
            continue
        accels.append(
            {
                "type": a.get("type") or "other",
                "enabled": bool(a.get("enabled", True)),
                "params": a.get("params") or {},
                "draft_model": a.get("draft_model"),
                "repo": a.get("repo"),
                "note": a.get("note"),
            }
        )

    rec: dict[str, Any] = {
        "id": record_id,
        "kind": "serve_recipe",
        "recipe_id": recipe_id,
        "artifact": {
            "name": model.get("name") or recipe_id,
            "hf_id": model.get("hf_id"),
            # Prefer explicit model.path (GGUF / ds4 absolute paths); else None
            "path": model.get("path"),
            "nodes": [],
            "size_gb": model.get("size_gb"),
            "params_total": model.get("params_total"),
            "params_active": model.get("params_active"),
            "architecture": model.get("architecture"),
            "is_vlm": None,
            "is_moe": (str(model.get("architecture") or "").lower().find("moe") >= 0)
            or bool(model.get("params_active")),
        },
        "quant": quant,
        "serve": {
            "engine": serve.get("engine") or "mlx_lm",
            "backend": backend,
            "port": serve.get("port"),
            "decode_concurrency": serve.get("decode_concurrency") or 1,
            "prefill_step_size": serve.get("prefill_step_size"),
            "prompt_cache_size": serve.get("prompt_cache_size"),
            "draft_model": serve.get("draft_model"),
        },
        "accelerators": accels,
        "hardware": {
            "min_nodes": hardware.get("min_nodes") or 1,
            "chip": node_profile.get("chip"),
            "ram_gb": node_profile.get("ram_gb"),
            "rdma_required": bool(hardware.get("rdma_required")),
            "peak_memory_gb": hardware.get("peak_memory_gb"),
            "max_context": hardware.get("max_context"),
            "notes": hardware.get("notes"),
        },
        "benchmarks": benches_from_config(cfg),
        "roles_fit": [],
        "known_issues": cfg.get("known_issues") or [],
        "verified_on": cfg.get("verified_on") or [],
        "source": {
            "serve_config_name": name,
            "asmi_path": None,
            "inventory_model_id": None,
        },
        "sources_used": ["serve-configs"],
        "description": cfg.get("description"),
        "created": cfg.get("created"),
        "sampling": cfg.get("sampling"),
    }

    # Draft safety: never task/smol/default if dflash draft arch or name
    is_draft = False
    arch = str(model.get("architecture") or "")
    if "dflash" in (model.get("name") or "").lower() and "draft" in (model.get("name") or "").lower():
        is_draft = True
    if "DFlashDraft" in arch:
        is_draft = True
    if is_draft:
        rec["roles_fit"] = ["draft"]
    else:
        # light default roles for real models
        rec["roles_fit"] = ["task", "active"]

    rec["search_text"] = build_search_text(rec)
    rec["search_narrative"] = build_search_narrative(rec, cfg)
    # hash without live/hash fields
    hashable = {k: v for k, v in rec.items() if k not in ("content_hash",)}
    rec["content_hash"] = content_hash(hashable)
    return rec


def recipe_to_serve_request(rec: dict[str, Any], hostfile: str | None = None) -> dict[str, Any]:
    """Materialize asmi /serve/load body (parity with r1o apply route).

    Required keys: model_path, engine, backend, port.
    backend ∈ {single, jaccl} only — never auto.
    Propagates draft_model / prefill / decode / accelerators so load keeps the recipe.
    """
    serve = rec.get("serve") or {}
    art = rec.get("artifact") or {}
    backend = serve.get("backend") or "single"
    if backend not in ("single", "jaccl"):
        backend = "single"

    # Prefer real path; else ~/Models/<name> (same as apply route).
    name = art.get("name") or rec.get("recipe_id") or "unknown"
    # serve-configs may stash path under model.path (copied into artifact when present)
    model_path = art.get("path") or f"~/Models/{name}"

    body: dict[str, Any] = {
        "model_path": model_path,
        "engine": serve.get("engine") or "mlx_lm",
        "backend": backend,
        "port": serve.get("port"),
        "decode_concurrency": serve.get("decode_concurrency") or 1,
        "recipe_id": rec.get("recipe_id") or rec.get("id"),
    }

    draft = serve.get("draft_model")
    if not draft:
        # fall back to first enabled accel draft_model (dflash profiles)
        for a in rec.get("accelerators") or []:
            if a.get("enabled", True) and a.get("draft_model"):
                draft = a.get("draft_model")
                break
    if draft:
        body["draft_model"] = draft
    if serve.get("prefill_step_size") is not None:
        body["prefill_step_size"] = serve["prefill_step_size"]
    if serve.get("prompt_cache_size") is not None:
        body["prompt_cache_size"] = serve["prompt_cache_size"]

    # Pass enabled accelerator types so downstream can enable TQ/DFlash/etc.
    accels = [a for a in (rec.get("accelerators") or []) if a.get("enabled", True)]
    if accels:
        body["accelerators"] = [
            {"type": a.get("type"), "params": a.get("params") or {}, "draft_model": a.get("draft_model")}
            for a in accels
        ]

    if backend == "jaccl":
        if not hostfile:
            raise ValueError("jaccl requires explicit hostfile path")
        body["hostfile"] = hostfile
    return body


def load_configs(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return list(data.get("configs") or [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.config.exists():
        print(f"missing {args.config}")
        return 1

    configs = load_configs(args.config)
    records = [recipe_from_config(c) for c in configs]

    # backend ban unit check
    for r in records:
        b = r["serve"]["backend"]
        assert b in ("single", "jaccl"), r["id"]
        req = recipe_to_serve_request(r, hostfile="~/.r1o/hostfiles/default.json" if b == "jaccl" else None)
        assert req["backend"] != "auto"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} recipes → {args.out}")
    # sample stats
    dflash = sum(1 for r in records if any(a.get("type") == "dflash" and a.get("enabled") for a in r["accelerators"]))
    jaccl = sum(1 for r in records if r["serve"]["backend"] == "jaccl")
    print(f"  dflash={dflash} jaccl={jaccl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
