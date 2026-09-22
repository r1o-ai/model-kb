#!/usr/bin/env python3
"""model-kb MCP — search / get / serving / load serve recipes.

Tools:
  model_search   — BM25 over recipes (quant, accel, engine, t/s tokens)
  model_get      — full recipe + ServeRequest materialization
  model_serving  — live asmi /serve/status (never from KB)
  model_load     — load recipe via r1o guarded apply (or dry_run)
  model_serve    — reap stale servers, pick best recipe, asmi /serve/load, reap leftovers

Env:
  ASMI_URL  default http://127.0.0.1:9090
  R1O_URL   default http://localhost:59408
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from build_bm25 import load_index, load_records, search as bm25_search  # noqa: E402
from ingest_serve_configs import recipe_to_serve_request  # noqa: E402

mcp = FastMCP("model-kb")

RECORDS = Path(os.getenv("MODEL_KB_RECORDS", str(ROOT / "records.jsonl")))
INDEX = Path(os.getenv("MODEL_KB_INDEX", str(ROOT / "bm25_index" / "index.pkl")))
ASMI_URL = os.getenv("ASMI_URL", "http://127.0.0.1:9090")
R1O_URL = os.getenv("R1O_URL", "http://localhost:59408")


def _recipe_score(rec: dict) -> float:
    """Measured agg_tps dominates. Unmeasured is not slow — it just loses to a sibling with numbers."""
    benches = rec.get("benchmarks") or []
    tps = 0.0
    for b in benches:
        if not isinstance(b, dict):
            continue
        v = b.get("agg_tps")
        if isinstance(v, (int, float)) and v > tps:
            tps = float(v)
    s = tps * 100.0
    if benches:
        s += 50.0
    s += 20.0 * len(rec.get("verified_on") or [])
    if (rec.get("artifact") or {}).get("path"):
        s += 15.0
    s += 8.0 * sum(1 for a in (rec.get("accelerators") or []) if isinstance(a, dict) and a.get("enabled"))
    if "active" in (rec.get("roles_fit") or []):
        s += 5.0
    return s


def _find(records: list[dict], recipe_id: str) -> dict | None:
    rid = recipe_id.removeprefix("recipe:")
    for r in records:
        if r.get("recipe_id") == rid or r.get("id") in (recipe_id, f"recipe:{rid}"):
            return r
        if (r.get("source") or {}).get("serve_config_name") in (rid, recipe_id):
            return r
    lower = rid.lower()
    exact: list[dict] = []
    partial: list[dict] = []
    for r in records:
        if r.get("kind") not in (None, "serve_recipe"):
            continue
        art = r.get("artifact") or {}
        name = str(art.get("name") or "").lower()
        hf = str(art.get("hf_id") or "").lower()
        if hf == lower or name == lower:
            exact.append(r)
        elif lower in name or lower in hf:
            partial.append(r)
    pool = exact or partial
    if not pool:
        return None
    return max(pool, key=_recipe_score)


_DOT_VER = re.compile(r"(?<![\w.])(?:v)?(\d+\.\d+)(?![\w.])", re.I)


def _query_versions(query: str) -> set[str]:
    return {m.group(1) for m in _DOT_VER.finditer(query or "")}


def _recipe_blob(rec: dict) -> str:
    art = rec.get("artifact") or {}
    return " ".join(
        str(x)
        for x in (
            rec.get("recipe_id"),
            rec.get("id"),
            art.get("name"),
            art.get("hf_id"),
            art.get("path"),
        )
        if x
    ).lower()


def _versions_ok(query: str, rec: dict) -> bool:
    need = _query_versions(query)
    if not need:
        return True
    blob = _recipe_blob(rec)
    return all(v in blob or f"v{v}" in blob for v in need)


def _resolve_best(records: list[dict], query: str) -> dict | None:
    """Best serve_recipe for a family name or recipe id. Dotted versions must match.

    'DeepSeek 4.1' must not resolve to DeepSeek-V4-Flash.
    """
    q = (query or "").strip()
    if not q:
        return None
    rec = _find(records, q)
    if rec and rec.get("kind") in (None, "serve_recipe") and _versions_ok(q, rec):
        return rec
    lower = q.lower()
    tokens = [t for t in re.split(r"[^a-z0-9.]+", lower) if len(t) > 1]
    cands: list[dict] = []
    for r in records:
        if r.get("kind") not in (None, "serve_recipe"):
            continue
        if not _versions_ok(q, r):
            continue
        blob = _recipe_blob(r)
        if lower in blob or (tokens and all(t in blob for t in tokens)):
            cands.append(r)
    if not cands:
        return None
    return max(cands, key=_recipe_score)


def _expand_path(raw: str) -> Path:
    s = str(raw or "")
    if s.startswith("~/"):
        return Path.home() / s[2:]
    return Path(s).expanduser()


def _reap_stale(*, keep_ports: set[int] | None = None) -> dict:
    """Stop asmi servers that are bare / model-less. Leave ready serves on other ports."""
    keep = keep_ports or set()
    code, data = _http_json("GET", f"{ASMI_URL.rstrip('/')}/serve/status", timeout=5)
    stopped: list[dict] = []
    noted: list[dict] = []
    if not isinstance(data, dict):
        return {"http_status": code, "stopped": stopped, "noted": [{"error": str(data)[:200]}]}
    for s in data.get("servers") or []:
        port = s.get("port")
        state = s.get("state")
        model = s.get("model")
        pid = s.get("pid")
        row = {"port": port, "state": state, "pid": pid, "model": model}
        if port in keep:
            continue
        if state in ("idle", None) and not pid:
            continue
        stale = state == "bare" or (not model and state not in ("starting", "ready"))
        if stale and port:
            sc, body = _http_json(
                "POST",
                f"{ASMI_URL.rstrip('/')}/serve/stop?port={port}",
                timeout=20,
            )
            stopped.append({**row, "stop_http": sc, "stop": body})
        else:
            noted.append(row)
    return {"http_status": code, "stopped": stopped, "noted": noted}


def _health_chat(port: int) -> dict:
    url = f"http://127.0.0.1:{int(port)}/v1/chat/completions"
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 4,
        "stream": False,
    }
    code, data = _http_json("POST", url, body, timeout=30)
    return {"http_status": code, "ok": code in (200, 201), "body": data if isinstance(data, dict) else str(data)[:300]}


def _http_json(method: str, url: str, body: dict | None = None, timeout: float = 120.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else str(e)
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw
    except Exception as e:  # noqa: BLE001
        return 0, str(e)


@mcp.tool()
def model_search(query: str, top_k: int = 8) -> dict:
    """BM25 search over serve recipes (quant, accelerators, engine, model names)."""
    if not INDEX.exists() or not RECORDS.exists():
        return {"error": "index missing — run ingest_serve_configs.py && build_bm25.py --rebuild"}
    idx = load_index(INDEX)
    hits = bm25_search(idx, query, top_k=top_k)
    return {"query": query, "hits": hits}


@mcp.tool()
def model_get(recipe_id: str) -> dict:
    """Get a recipe by id/name and materialize the asmi ServeRequest body.

    Includes local serve benchmarks, community_quality (Unsloth/HF card scores),
    and hf_family (Hub base_model tree levels) when enrichments are joined.
    """
    records = load_records(RECORDS)
    rec = _find(records, recipe_id)
    if not rec:
        return {"error": f"not found: {recipe_id}"}
    backend = (rec.get("serve") or {}).get("backend") or "single"
    hostfile = str(Path.home() / ".r1o" / "hostfiles" / "default.json") if backend == "jaccl" else None
    try:
        body = recipe_to_serve_request(rec, hostfile=hostfile)
    except ValueError as e:
        return {"error": str(e), "recipe_id": rec.get("recipe_id")}
    return {
        "recipe_id": rec.get("recipe_id"),
        "artifact": rec.get("artifact"),
        "quant": rec.get("quant"),
        "serve": rec.get("serve"),
        "accelerators": rec.get("accelerators"),
        "hardware": rec.get("hardware"),
        "benchmarks": rec.get("benchmarks"),
        "community_quality": rec.get("community_quality"),
        "hf_family": rec.get("hf_family"),
        "enrichment_links": rec.get("enrichment_links"),
        "serve_request": body,
        "asmi_body": {k: v for k, v in body.items() if k != "recipe_id"},
        "source": "model-kb",
    }


@mcp.tool()
def model_optimization_transfer(recipe_id: str = "", target_hf_id: str = "") -> dict:
    """Which optimizations proven elsewhere apply to this recipe/model?

    Codifies transfer of dflash / mtp / mtplx / triattention / etc. across
    same-base quant or finetune with expected acceptance_mult (e.g. MTP on
    base → finetune often status=degraded, mult≈0.55). Use before serving a
    variant with an accel measured only on the base.

    Provide recipe_id and/or target_hf_id.
    """
    records = load_records(RECORDS)
    recipes = [r for r in records if r.get("kind") == "serve_recipe"]
    target = None
    if recipe_id:
        target = _find(records, recipe_id)
    if not target and target_hf_id:
        for r in recipes:
            art = r.get("artifact") or {}
            if art.get("hf_id") == target_hf_id or target_hf_id in str(art.get("name") or ""):
                target = r
                break
    if not target:
        return {"error": "target recipe not found", "recipe_id": recipe_id, "target_hf_id": target_hf_id}

    # Prefer precomputed enrichment; else compute live
    pre = target.get("optimization_transfer")
    if pre and pre.get("transfers"):
        return {
            "recipe_id": target.get("recipe_id"),
            "artifact": target.get("artifact"),
            "optimization_transfer": pre,
            "source": "model-kb/precomputed",
            "hint": "degraded = try but re-bench acceptance; deny = do not transfer draft/MTP heads",
        }

    try:
        from optimization_transfer import summarize_for_recipe
    except Exception as e:  # noqa: BLE001
        return {"error": f"optimization_transfer module: {e}"}

    summary = summarize_for_recipe(target, recipes)
    return {
        "recipe_id": target.get("recipe_id"),
        "artifact": target.get("artifact"),
        "optimization_transfer": {
            "by_status": summary.get("by_status"),
            "transfers": summary.get("actionable_transfers"),
        },
        "source": "model-kb/live",
        "hint": "degraded = try but re-bench acceptance; deny = do not transfer draft/MTP heads",
    }


@mcp.tool()
def model_family(hf_id: str = "", recipe_id: str = "") -> dict:
    """Show Hugging Face family-tree levels for a base model or recipe.

    Uses joined hf_family enrichment (from GET /api/models?filter=base_model:…).
    Levels: base, official_quant, mlx, gguf_unsloth, gguf_other, awq_gptq_fp8,
    draft_dflash, finetune_abliterated, other — so you can pick which quant/line to serve.
    """
    records = load_records(RECORDS)
    rec = None
    if recipe_id:
        rec = _find(records, recipe_id)
    if not rec and hf_id:
        for r in records:
            if r.get("kind") != "serve_recipe":
                continue
            art = r.get("artifact") or {}
            if art.get("hf_id") == hf_id or (art.get("name") or "") in hf_id:
                rec = r
                break
    if not rec:
        # search enrichment rows stored as kind=hf_family in records
        for r in records:
            if r.get("kind") == "hf_family":
                root = (r.get("measurements") or {}).get("root_base") or ""
                if hf_id and (hf_id == root or hf_id in root or root in hf_id):
                    return {
                        "root_base": root,
                        "family": (r.get("measurements") or {}).get("family_levels"),
                        "recommend_serve": (r.get("measurements") or {}).get("recommend_serve"),
                        "source": "model-kb/hf_family",
                    }
        return {"error": "no recipe/family found", "hf_id": hf_id, "recipe_id": recipe_id}

    fam = rec.get("hf_family")
    return {
        "recipe_id": rec.get("recipe_id"),
        "artifact": rec.get("artifact"),
        "hf_family": fam,
        "community_quality": rec.get("community_quality"),
        "enrichment_links": rec.get("enrichment_links"),
        "hint": "Use recommend_serve / mlx / gguf_unsloth levels to pick a quant; local benchmarks[] for tok/s.",
        "source": "model-kb",
    }


@mcp.tool()
def model_serving() -> dict:
    """Live asmi /serve/status — never from the KB index."""
    code, data = _http_json("GET", f"{ASMI_URL.rstrip('/')}/serve/status", timeout=5)
    return {"http_status": code, "asmi": data}


@mcp.tool()
def model_load(recipe_id: str, dry_run: bool = True) -> dict:
    """Load a recipe via r1o guarded apply route (guardedServeLoad).

    dry_run=True (default) only materializes + checks model_path exists.
    dry_run=False POSTs /api/hermes/serve-configs/{name}/apply.
    """
    records = load_records(RECORDS)
    rec = _find(records, recipe_id)
    if not rec:
        return {"error": f"not found: {recipe_id}"}
    backend = (rec.get("serve") or {}).get("backend") or "single"
    if backend not in ("single", "jaccl"):
        backend = "single"
    hostfile = str(Path.home() / ".r1o" / "hostfiles" / "default.json") if backend == "jaccl" else None
    try:
        body = recipe_to_serve_request(rec, hostfile=hostfile if backend == "jaccl" else None)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    if body.get("backend") == "auto" or "model_path" not in body:
        return {"ok": False, "error": "invalid ServeRequest (auto backend or missing model_path)"}

    asmi_body = {k: v for k, v in body.items() if k != "recipe_id"}
    name = rec.get("recipe_id")
    mp = str(asmi_body.get("model_path") or "")
    expanded = Path.home() / mp[2:] if mp.startswith("~/") else Path(mp).expanduser()

    out = {
        "recipe_id": name,
        "asmi_body": asmi_body,
        "model_path_exists": expanded.exists(),
        "model_path_resolved": str(expanded),
        "via": "dry-run" if dry_run else "r1o-guarded-apply",
        "apply_url": f"{R1O_URL.rstrip('/')}/api/hermes/serve-configs/{name}/apply",
    }

    if dry_run:
        out["ok"] = True
        out["note"] = "dry-run — no load"
        return out

    code, data = _http_json(
        "POST",
        out["apply_url"],
        {},
        timeout=180,
    )
    out["http_status"] = code
    out["apply"] = data
    out["ok"] = code in (200, 201) and isinstance(data, dict) and data.get("ok") is True
    sc, status = _http_json("GET", f"{ASMI_URL.rstrip('/')}/serve/status", timeout=5)
    out["serve_status_http"] = sc
    if isinstance(status, dict):
        port = asmi_body.get("port")
        match = [s for s in (status.get("servers") or []) if s.get("port") == port]
        out["port_status"] = match[0] if match else None
    return out


@mcp.tool()
def model_serve(query: str, dry_run: bool = False) -> dict:
    """Pick the best recipe for `query`, reap stale/bare servers, load via asmi.

    One-shot serve. Query can be a family name ('DeepSeek 4.1') or a recipe_id.
    Dotted versions must match — '4.1' will not load V4-Flash.
    dry_run=True materializes + reaps report only (no load).
    CHANGES CLUSTER STATE when dry_run is false.
    """
    records = load_records(RECORDS)
    rec = _resolve_best(records, query)
    if not rec:
        return {
            "ok": False,
            "error": f"no serve_recipe matching {query!r}",
            "kind": "missing_recipe_or_weights",
            "hint": "A research note is not a config. Add weights + a serve-config, then ingest.",
        }

    backend = (rec.get("serve") or {}).get("backend") or "single"
    if backend not in ("single", "jaccl"):
        backend = "single"
    hostfile = str(Path.home() / ".r1o" / "hostfiles" / "default.json") if backend == "jaccl" else None
    try:
        body = recipe_to_serve_request(rec, hostfile=hostfile if backend == "jaccl" else None)
    except ValueError as e:
        return {"ok": False, "error": str(e), "recipe_id": rec.get("recipe_id")}

    mp = str(body.get("model_path") or "")
    expanded = _expand_path(mp)
    port = body.get("port")
    before = _reap_stale(keep_ports={int(port)} if port else set())

    out: dict = {
        "recipe_id": rec.get("recipe_id"),
        "query": query,
        "score": _recipe_score(rec),
        "asmi_body": {k: v for k, v in body.items() if k != "recipe_id"},
        "model_path_exists": expanded.exists(),
        "model_path_resolved": str(expanded),
        "reap_before": before,
        "via": "dry-run" if dry_run else "asmi-serve-load",
    }

    if not expanded.exists():
        out["ok"] = False
        out["kind"] = "missing_weights"
        out["error"] = f"weights not on disk: {expanded}"
        return out

    if dry_run:
        out["ok"] = True
        out["note"] = "dry-run — reaped stale, did not load"
        return out

    code, data = _http_json(
        "POST",
        f"{ASMI_URL.rstrip('/')}/serve/load",
        out["asmi_body"],
        timeout=180,
    )
    out["load_http"] = code
    out["load"] = data
    keep = {int(port)} if port else set()
    out["reap_after"] = _reap_stale(keep_ports=keep)
    if port:
        out["health"] = _health_chat(int(port))
        sc, status = _http_json("GET", f"{ASMI_URL.rstrip('/')}/serve/status", timeout=5)
        out["serve_status_http"] = sc
        if isinstance(status, dict):
            match = [s for s in (status.get("servers") or []) if s.get("port") == port]
            out["port_status"] = match[0] if match else None
    out["ok"] = code in (200, 201)
    return out


if __name__ == "__main__":
    mcp.run()
