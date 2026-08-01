#!/usr/bin/env python3
"""ingest_cloud_context.py — cloud model *context packs* → model-kb records.

wiki-mass-ingest analog for **models**, not wiki pages:

  EXTRACT  cloud catalogs + live probes (context, options, quota, benches)
  ROUTE    kind=cloud_context (not serve_recipe)
  WRITE    records.jsonl (merge by id; leave recipe:* alone)
  INDEX    rebuild BM25 (caller or --rebuild-index)

Sources (best-effort; never invent numbers):
  - web/src/lib/cloud-models.ts  (static offline catalog)
  - web/src/data/cloud-model-specs.json  (LiteLLM-ish specs)
  - web/src/lib/cloud-model-specs.ts overrides (GLM-5.x)
  - GET /api/pi/providers when R1O_URL reachable (OMP live)
  - GET /api/usage/status (remaining chips for providers)
  - Z.AI Coding Plan monitor when ZAI_API_KEY set
  - optional CLIProxy /v1/models when CLIPROXY_API_KEY / conf key present

Usage:
  python3 ingest_cloud_context.py
  python3 ingest_cloud_context.py --out records.jsonl --rebuild-index
  R1O_URL=http://localhost:59408 python3 ingest_cloud_context.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
R1O = Path(os.getenv("R1O_ROOT", Path.home() / "Projects" / "r1o"))
DEFAULT_OUT = ROOT / "records.jsonl"
CLOUD_MODELS_TS = R1O / "web" / "src" / "lib" / "cloud-models.ts"
CLOUD_SPECS_JSON = R1O / "web" / "src" / "data" / "cloud-model-specs.json"
CLOUD_SPECS_TS = R1O / "web" / "src" / "lib" / "cloud-model-specs.ts"
R1O_URL = os.getenv("R1O_URL", "http://127.0.0.1:59408").rstrip("/")


def content_hash(obj: dict[str, Any]) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def http_json(url: str, headers: dict[str, str] | None = None, timeout: float = 8.0) -> Any | None:
    req = urllib.request.Request(url, headers=headers or {"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        print(f"WARN {url}: {type(e).__name__} {e}", file=sys.stderr)
        return None


def parse_cloud_models_ts(path: Path) -> list[dict[str, Any]]:
    """Lightweight extract of CLOUD_PROVIDERS models from cloud-models.ts."""
    if not path.exists():
        return []
    text = path.read_text()
    # Split by provider blocks id: 'anthropic' | 'openai' | 'zai' | 'xai'
    providers: list[dict[str, Any]] = []
    # Find each provider id + name + models array entries
    for m in re.finditer(
        r"id:\s*'(?P<id>anthropic|openai|zai|xai)'\s*,\s*\n\s*name:\s*'(?P<name>[^']+)'",
        text,
    ):
        providers.append({"id": m.group("id"), "name": m.group("name"), "start": m.start()})
    if not providers:
        # fallback: just model lines
        providers = [{"id": "unknown", "name": "unknown", "start": 0}]

    models: list[dict[str, Any]] = []
    # model object lines
    model_re = re.compile(
        r"\{\s*id:\s*'(?P<id>[^']+)'\s*,\s*shortName:\s*'(?P<short>[^']+)'\s*,\s*"
        r"provider:\s*'(?P<provider>[^']+)'\s*,\s*"
        r"contextWindow:\s*(?P<ctx>[\d_]+)\s*,\s*"
        r"maxOutputTokens:\s*(?P<max>[\d_]+)\s*,\s*"
        r"vision:\s*(?P<vision>true|false)"
    )
    for m in model_re.finditer(text):
        models.append(
            {
                "id": m.group("id"),
                "shortName": m.group("short"),
                "provider": m.group("provider"),
                "contextWindow": int(m.group("ctx").replace("_", "")),
                "maxOutputTokens": int(m.group("max").replace("_", "")),
                "vision": m.group("vision") == "true",
                "source": "cloud-models.ts",
            }
        )
    return models


# fix typo findIter
def load_specs_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_spec_overrides_ts(path: Path) -> dict[str, Any]:
    """Parse GLM override table in cloud-model-specs.ts (simple)."""
    if not path.exists():
        return {}
    text = path.read_text()
    out: dict[str, Any] = {}
    for m in re.finditer(
        r"'(?P<id>glm-[^']+)':\s*\{\s*contextWindow:\s*(?P<ctx>[\d_]+)\s*,\s*"
        r"maxOutputTokens:\s*(?P<max>[\d_]+)\s*,\s*vision:\s*(?P<vis>true|false)",
        text,
    ):
        out[m.group("id")] = {
            "contextWindow": int(m.group("ctx").replace("_", "")),
            "maxOutputTokens": int(m.group("max").replace("_", "")),
            "vision": m.group("vis") == "true",
            "source": "cloud-model-specs.ts",
        }
    return out


def fetch_pi_providers() -> list[dict[str, Any]]:
    data = http_json(f"{R1O_URL}/api/pi/providers")
    if not isinstance(data, dict):
        return []
    models = data.get("models") or []
    out = []
    for m in models:
        if not isinstance(m, dict):
            continue
        if m.get("local"):
            continue
        out.append(
            {
                "id": m.get("id") or "",
                "value": m.get("value"),
                "provider": m.get("provider"),
                "shortName": m.get("shortName") or m.get("id"),
                "family": m.get("family"),
                "contextWindow": m.get("contextWindow"),
                "maxOutputTokens": m.get("maxOutputTokens") or m.get("maxTokens"),
                "loginLabel": m.get("loginLabel"),
                "email": m.get("email"),
                "source": "api/pi/providers",
            }
        )
    return out


def fetch_usage_status() -> dict[str, Any]:
    data = http_json(f"{R1O_URL}/api/usage/status")
    return data if isinstance(data, dict) else {}


def fetch_zai_quota() -> dict[str, Any] | None:
    key = os.getenv("ZAI_API_KEY") or os.getenv("BIGMODEL_API_KEY")
    if not key:
        return None
    for base in (
        "https://api.z.ai/api/monitor/usage/quota/limit",
        "https://open.bigmodel.cn/api/monitor/usage/quota/limit",
    ):
        data = http_json(base, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
        if isinstance(data, dict) and data.get("success"):
            return data
    return None


def merge_model_rows(
    static: list[dict[str, Any]],
    specs: dict[str, Any],
    overrides: dict[str, Any],
    live: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Key by provider/id."""
    by: dict[str, dict[str, Any]] = {}

    def key(provider: str, mid: str) -> str:
        return f"{provider}/{mid}".lower()

    for m in static:
        k = key(m["provider"], m["id"])
        by[k] = {
            "provider": m["provider"],
            "model_id": m["id"],
            "short_name": m["shortName"],
            "context_window": m["contextWindow"],
            "max_output_tokens": m["maxOutputTokens"],
            "vision": m["vision"],
            "sources": [m["source"]],
            "options": {
                "thinking": None,
                "vision": m["vision"],
            },
            "benchmarks": [],
            "quota": None,
            "family": None,
            "login_label": None,
        }

    for mid, spec in specs.items():
        # try match any provider
        for prov in ("anthropic", "openai", "zai", "xai", "google"):
            k = key(prov, mid)
            if k in by:
                by[k]["context_window"] = spec.get("contextWindow") or by[k]["context_window"]
                by[k]["max_output_tokens"] = spec.get("maxOutputTokens") or by[k]["max_output_tokens"]
                if "vision" in spec:
                    by[k]["vision"] = spec["vision"]
                    by[k]["options"]["vision"] = spec["vision"]
                by[k]["sources"].append("cloud-model-specs.json")
                break
        else:
            # store under unknown/mid for searchability
            k = key("litellm", mid)
            by[k] = {
                "provider": "litellm",
                "model_id": mid,
                "short_name": mid,
                "context_window": spec.get("contextWindow"),
                "max_output_tokens": spec.get("maxOutputTokens"),
                "vision": spec.get("vision"),
                "sources": ["cloud-model-specs.json"],
                "options": {"vision": spec.get("vision")},
                "benchmarks": [],
                "quota": None,
                "family": None,
                "login_label": None,
            }

    for mid, spec in overrides.items():
        k = key("zai", mid)
        row = by.get(k) or {
            "provider": "zai",
            "model_id": mid,
            "short_name": mid,
            "sources": [],
            "options": {},
            "benchmarks": [],
            "quota": None,
            "family": "zai",
            "login_label": None,
        }
        row["context_window"] = spec.get("contextWindow")
        row["max_output_tokens"] = spec.get("maxOutputTokens")
        row["vision"] = spec.get("vision")
        row["options"]["vision"] = spec.get("vision")
        row["sources"] = list(set(row.get("sources") or []) | {spec.get("source") or "override"})
        by[k] = row

    for m in live:
        mid = m.get("id") or ""
        prov = (m.get("provider") or m.get("family") or "unknown").split("/")[0]
        # normalize omp families
        fam = (m.get("family") or "").lower()
        if fam == "claude":
            prov = "anthropic"
        elif fam == "codex":
            prov = "openai"
        elif fam == "xai":
            prov = "xai"
        elif fam == "zai":
            prov = "zai"
        k = key(prov, mid)
        row = by.get(k) or {
            "provider": prov,
            "model_id": mid,
            "short_name": m.get("shortName") or mid,
            "sources": [],
            "options": {},
            "benchmarks": [],
            "quota": None,
            "family": fam or None,
            "login_label": None,
            "context_window": None,
            "max_output_tokens": None,
            "vision": None,
        }
        # Prefer the larger max_context (static catalog often has official
        # limits; OMP models.yml sometimes ships a lower operational cap).
        live_ctx = m.get("contextWindow")
        if live_ctx:
            try:
                live_ctx_i = int(live_ctx)
                prev = row.get("context_window")
                prev_i = int(prev) if prev is not None else 0
                if live_ctx_i > prev_i:
                    row["context_window"] = live_ctx_i
            except (TypeError, ValueError):
                pass
        live_max = m.get("maxOutputTokens")
        if live_max:
            try:
                live_max_i = int(live_max)
                prev = row.get("max_output_tokens")
                prev_i = int(prev) if prev is not None else 0
                if live_max_i > prev_i:
                    row["max_output_tokens"] = live_max_i
            except (TypeError, ValueError):
                pass
        if m.get("loginLabel"):
            row["login_label"] = m["loginLabel"]
        if m.get("email"):
            row["email"] = m["email"]
        if m.get("value"):
            row["omp_value"] = m["value"]
        row["family"] = fam or row.get("family")
        row["sources"] = list(set(row.get("sources") or []) | {"api/pi/providers"})
        by[k] = row

    return by


def to_record(row: dict[str, Any], usage: dict[str, Any], zai_quota: dict[str, Any] | None) -> dict[str, Any]:
    provider = row["provider"]
    mid = row["model_id"]
    rid = f"cloud:{provider}:{mid}".lower().replace(" ", "-")

    # attach provider-level remaining from usage status
    quota = None
    by_p = (usage or {}).get("byProvider") or {}
    if provider in by_p:
        bar = by_p[provider]
        quota = {
            "kind": bar.get("kind"),
            "display": bar.get("display"),
            "fraction": bar.get("fraction"),
            "detail": bar.get("detail"),
            "source": "api/usage/status",
        }
    if provider == "zai" and zai_quota and isinstance(zai_quota.get("data"), dict):
        limits = zai_quota["data"].get("limits") or []
        time = next((x for x in limits if x.get("type") == "TIME_LIMIT"), limits[0] if limits else None)
        if time:
            used = time.get("percentage")
            rem_frac = None
            if isinstance(time.get("remaining"), (int, float)) and time.get("usage"):
                rem_frac = time["remaining"] / max(1, time["usage"])
            elif isinstance(used, (int, float)):
                rem_frac = 1 - used / 100
            quota = {
                "kind": "remaining",
                "level": zai_quota["data"].get("level"),
                "window": f"{time.get('unit')}h" if time.get("type") == "TIME_LIMIT" else time.get("type"),
                "remaining": time.get("remaining"),
                "usage_cap": time.get("usage"),
                "used_percent": used,
                "fraction": rem_frac,
                "source": "zai/monitor/usage/quota/limit",
            }

    options = {
        "vision": row.get("vision"),
        "thinking": row.get("options", {}).get("thinking"),
        "max_output_tokens": row.get("max_output_tokens"),
        "context_window": row.get("context_window"),
        "omp_value": row.get("omp_value"),
        "login_label": row.get("login_label"),
    }

    art = {
        "name": row.get("short_name") or mid,
        "hf_id": None,
        "path": None,
        "nodes": [],
        "provider": provider,
        "model_id": mid,
        "family": row.get("family"),
        "is_vlm": bool(row.get("vision")),
        "is_moe": None,
        "is_cloud": True,
    }

    # Max context is first-class (same role as hardware.max_context on local recipes)
    max_context = row.get("context_window")
    try:
        max_context = int(max_context) if max_context is not None else None
    except (TypeError, ValueError):
        max_context = None

    hardware = {
        "min_nodes": 0,
        "chip": "cloud",
        "ram_gb": None,
        "rdma_required": False,
        "max_context": max_context,
        "notes": f"Cloud model via {provider}",
    }

    serve = {
        "engine": "cloud",
        "backend": provider,
        "port": None,
        "max_context": max_context,
        "max_output_tokens": row.get("max_output_tokens"),
        "api_surface": {
            "anthropic": "Messages API / OAuth or API key",
            "openai": "Chat Completions / Codex OAuth",
            "xai": "OpenAI-compatible api.x.ai",
            "zai": "Anthropic-compat open.bigmodel.cn + Coding Plan",
        }.get(provider, "provider API"),
    }

    # benches: leave empty unless we have real numbers later
    benches: list[dict[str, Any]] = list(row.get("benchmarks") or [])

    body: dict[str, Any] = {
        "id": rid,
        "kind": "cloud_context",
        "recipe_id": rid.removeprefix("cloud:").replace(":", "-"),
        "artifact": art,
        "quant": {"label": "api", "raw": "api", "aliases": ["cloud", "api"]},
        "serve": serve,
        "accelerators": [],
        "hardware": hardware,
        # Explicit top-level for chat UI / SessionStatsBar / context meter
        "max_context": max_context,
        "max_output_tokens": row.get("max_output_tokens"),
        "options": options,
        "quota": quota,
        "benchmarks": benches,
        "roles_fit": ["chat", "agent"] if provider in ("anthropic", "openai", "xai", "zai") else ["chat"],
        "known_issues": [],
        "verified_on": [],
        "sources_used": sorted(set(row.get("sources") or [])),
        "created": date.today().isoformat(),
        "description": (
            f"{art['name']} ({provider}/{mid}) cloud context pack — "
            f"max_context={max_context} max_out={row.get('max_output_tokens')} "
            f"vision={row.get('vision')}"
        ),
    }
    # search fields for BM25 — include max_context as token for "1m context" queries
    ctx_label = ""
    if max_context:
        if max_context >= 1_000_000:
            ctx_label = f"{max_context // 1_000_000}m_context"
        elif max_context >= 1000:
            ctx_label = f"{max_context // 1000}k_context"
    tokens = [
        provider,
        mid,
        art["name"],
        "cloud",
        "context",
        "max_context",
        str(max_context or ""),
        ctx_label,
        "vision" if row.get("vision") else "text",
        row.get("family") or "",
        row.get("login_label") or "",
    ]
    if quota and quota.get("display"):
        tokens.append(str(quota["display"]))
    body["search_text"] = " ".join(t for t in tokens if t)
    body["search_narrative"] = (
        f"{art['name']} is a cloud model on {provider}. "
        f"Context window {row.get('context_window') or 'unknown'} tokens; "
        f"max output {row.get('max_output_tokens') or 'unknown'}. "
        f"{'Supports vision. ' if row.get('vision') else ''}"
        f"Sources: {', '.join(body['sources_used'])}."
    )
    if quota:
        body["search_narrative"] += f" Quota signal: {quota.get('display')} ({quota.get('kind')})."
    body["content_hash"] = content_hash({k: v for k, v in body.items() if k != "content_hash"})
    body["source"] = {"kind": "cloud_context_ingest", "date": date.today().isoformat()}
    return body


def merge_records(existing: list[dict[str, Any]], cloud_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for r in existing:
        rid = r.get("id")
        if rid:
            by_id[rid] = r
    # drop old cloud_context then re-add
    by_id = {k: v for k, v in by_id.items() if v.get("kind") != "cloud_context"}
    for r in cloud_rows:
        by_id[r["id"]] = r
    # stable-ish order: recipes first then cloud
    recipes = [v for v in by_id.values() if v.get("kind") == "serve_recipe"]
    clouds = sorted(
        [v for v in by_id.values() if v.get("kind") == "cloud_context"],
        key=lambda x: x.get("id") or "",
    )
    other = [v for v in by_id.values() if v.get("kind") not in ("serve_recipe", "cloud_context")]
    return recipes + other + clouds


def main() -> int:
    ap = argparse.ArgumentParser(prog="ingest_cloud_context")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    static = parse_cloud_models_ts(CLOUD_MODELS_TS)
    specs = load_specs_json(CLOUD_SPECS_JSON)
    overrides = load_spec_overrides_ts(CLOUD_SPECS_TS)
    live = fetch_pi_providers()
    usage = fetch_usage_status()
    zai_q = fetch_zai_quota()

    print(
        f"sources: static={len(static)} specs={len(specs)} "
        f"overrides={len(overrides)} live_pi={len(live)} "
        f"usage_providers={list((usage.get('byProvider') or {}).keys())} "
        f"zai_quota={'yes' if zai_q else 'no'}"
    )

    merged = merge_model_rows(static, specs, overrides, live)
    cloud_recs = [to_record(row, usage, zai_q) for row in merged.values() if row.get("model_id")]
    print(f"cloud_context packs: {len(cloud_recs)}")

    existing: list[dict[str, Any]] = []
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                existing.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    final = merge_records(existing, cloud_recs)
    print(f"records total: {len(final)} (was {len(existing)})")

    if args.dry_run:
        # print a few
        for r in cloud_recs[:5]:
            print(" ", r["id"], r.get("hardware", {}).get("max_context"), r.get("quota"))
        return 0

    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in final) + "\n")
    print(f"wrote {args.out}")

    if args.rebuild_index:
        import subprocess

        subprocess.check_call([sys.executable, str(ROOT / "build_bm25.py"), "--rebuild"], cwd=str(ROOT))
        print("bm25 index rebuilt")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
