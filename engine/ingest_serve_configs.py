#!/usr/bin/env python3
"""ingest_serve_configs.py — serve-configs.json v3 → L0 recipe records.

Phase B task 6 of model-kb ARDD plan (2026-07-11).
SoT: ~/.r1o/serve-configs.json (never invent profiles).
Output: records.jsonl (append-safe via rewrite of recipe:* rows only when run standalone).

The `serve` section is carried whole, under its own names: asmi's
`POST /serve/load` is the contract (lib/asmi_load.py), so every asmi field a
config sets reaches the recipe and the load unchanged. Nothing here picks a
subset, and nothing defaults an engine: a config without one gives a recipe
that is not loadable, with the reason in `problems`.

Usage:
  python3 ingest_serve_configs.py
  python3 ingest_serve_configs.py --out records.jsonl
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lib.asmi_load import asmi_body_of, describe_type, serve_load_fields, type_ok  # noqa: E402
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
    (re.compile(r"\bfp8\b|fp8[-\w]*", re.I), "fp8"),
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
    tokens.append(str(serve.get("hardware") or ""))
    tokens.append(str(serve.get("os") or ""))
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


BACKENDS = ("single", "jaccl")  # model-kb policy: never "auto" (asmi accepts it; model-kb does not send it)
ENGINE_MISSING = (
    "serve.engine missing: the serve config names no engine, so asmi cannot load it "
    "(model-kb never guesses one); add serve.engine as asmi GET /serve/engines names it"
)


def recipe_serve(serve: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    """The config's `serve` section as the recipe records it, plus why it can't load.

    Every key is kept under its own name: asmi LoadRequest fields reach the load
    unchanged, and config-only keys (hardware, os, …) stay on the record for
    search. Three things differ from the config, each one named:
      - `extra_args` that is not a list of strings is dropped (never coerced)
        and recorded as a problem: asmi takes pre-split argv only.
      - `backend` other than single|jaccl is recorded as single, with a warning.
      - no `engine` → engine null and a problem (never a default engine).
    """
    rec_serve = copy.deepcopy(serve) if isinstance(serve, dict) else {}
    problems: list[str] = []
    warnings: list[str] = []

    if "extra_args" in rec_serve and rec_serve["extra_args"] is not None:
        xa = rec_serve["extra_args"]
        if not type_ok("string_list", xa):
            del rec_serve["extra_args"]
            problems.append(
                f"serve.extra_args dropped: it is {xa!r} ({type(xa).__name__}), but asmi "
                f"LoadRequest.extra_args takes {describe_type('string_list')}; fix the serve config"
            )

    engine = rec_serve.get("engine") or None
    rec_serve["engine"] = engine
    if not engine:
        problems.append(ENGINE_MISSING)

    raw_backend = rec_serve.get("backend")
    backend = raw_backend or "single"
    if backend not in BACKENDS:
        warnings.append(
            f"serve.backend {raw_backend!r} is loaded as 'single': model-kb sends only "
            f"{' or '.join(BACKENDS)}, never auto"
        )
        backend = "single"
    rec_serve["backend"] = backend

    checked = serve_load_fields(rec_serve)
    problems += checked["problems"]
    warnings += checked["warnings"]
    return rec_serve, _dedupe(problems), _dedupe(warnings)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def recipe_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    name = cfg.get("name") or "unnamed"
    model = cfg.get("model") or {}
    serve = cfg.get("serve") or {}
    hardware = cfg.get("hardware") or {}
    node_profile = hardware.get("node_profile") or {}

    rec_serve, problems, warnings = recipe_serve(serve)

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
        "serve": rec_serve,
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
    # Why this recipe cannot load as recorded (empty = loadable). The body
    # builder (materialize_load) applies the same rules again at load time.
    rec["problems"] = problems
    rec["warnings"] = warnings
    rec["loadable"] = not problems

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


def materialize_load(rec: dict[str, Any], hostfile: str | None = None) -> dict[str, Any]:
    """The asmi /serve/load request a recipe makes, and why it can't load.

    Returns {asmi_body, port, problems, warnings, not_sent, loadable}:
      asmi_body — asmi LoadRequest fields only: every one the recipe's `serve`
                  sets, unchanged, plus model_path (artifact.path, else
                  ~/Models/<name>, the apply route's convention) and the draft
                  model from the first enabled accelerator when serve has none.
      port      — asmi takes it as ?port=N, never in the body.
      not_sent  — serve keys that are not asmi fields (config metadata).
    `hostfile` is the caller's fallback for a jaccl recipe that records none;
    a recipe's own serve.hostfile wins. backend is single|jaccl only.
    """
    serve = rec.get("serve") or {}
    art = rec.get("artifact") or {}
    problems: list[str] = list(rec.get("problems") or [])
    warnings: list[str] = list(rec.get("warnings") or [])

    checked = serve_load_fields(serve)
    problems += checked["problems"]
    warnings += checked["warnings"]
    fields = dict(checked["fields"])

    if not fields.get("engine"):
        problems.append(ENGINE_MISSING)
    backend = fields.get("backend") or "single"
    if backend not in BACKENDS:
        warnings.append(f"serve.backend {backend!r} is loaded as 'single' (model-kb never sends auto)")
        backend = "single"
    fields["backend"] = backend

    name = art.get("name") or rec.get("recipe_id") or "unknown"
    fields["model_path"] = art.get("path") or f"~/Models/{name}"

    if not fields.get("draft_model"):
        for a in rec.get("accelerators") or []:
            if a.get("enabled", True) and a.get("draft_model"):
                fields["draft_model"] = a["draft_model"]
                break

    if backend == "jaccl" and not fields.get("hostfile"):
        if hostfile:
            fields["hostfile"] = hostfile
        else:
            problems.append("jaccl requires explicit hostfile path (serve.hostfile or the caller's)")

    problems = _dedupe(problems)
    return {
        "asmi_body": fields,
        "port": serve.get("port"),
        "problems": problems,
        "warnings": _dedupe(warnings),
        "not_sent": checked["not_sent"],
        "loadable": not problems,
    }


def recipe_to_serve_request(rec: dict[str, Any], hostfile: str | None = None) -> dict[str, Any]:
    """model-kb's serve request for a recipe: the asmi body + port + recipe_id + accelerators.

    Raises ValueError naming every problem when the recipe cannot load as
    recorded. POST only `asmi_body_of(request)` to asmi, with the port as
    ?port= — `port`, `recipe_id` and `accelerators` are model-kb keys.
    """
    plan = materialize_load(rec, hostfile=hostfile)
    if plan["problems"]:
        raise ValueError(f"recipe {rec.get('recipe_id') or rec.get('id')} is not loadable: " + "; ".join(plan["problems"]))
    req: dict[str, Any] = dict(plan["asmi_body"])
    req["port"] = plan["port"]
    req["recipe_id"] = rec.get("recipe_id") or rec.get("id")
    accels = [a for a in (rec.get("accelerators") or []) if a.get("enabled", True)]
    if accels:
        req["accelerators"] = [
            {"type": a.get("type"), "params": a.get("params") or {}, "draft_model": a.get("draft_model")}
            for a in accels
        ]
    return req


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

    # backend ban unit check (an unloadable recipe is recorded, not fatal)
    for r in records:
        b = r["serve"]["backend"]
        assert b in BACKENDS, r["id"]
        plan = materialize_load(r, hostfile="~/.r1o/hostfiles/default.json" if b == "jaccl" else None)
        assert plan["asmi_body"]["backend"] != "auto"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} recipes → {args.out}")
    # sample stats
    dflash = sum(1 for r in records if any(a.get("type") == "dflash" and a.get("enabled") for a in r["accelerators"]))
    jaccl = sum(1 for r in records if r["serve"]["backend"] == "jaccl")
    print(f"  dflash={dflash} jaccl={jaccl}")
    for r in records:
        if not r["loadable"]:
            print(f"  NOT LOADABLE {r['recipe_id']}: " + "; ".join(r["problems"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
