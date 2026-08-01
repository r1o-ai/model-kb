#!/usr/bin/env python3
"""model_kb.py — search / get / load serve recipes (Phase C).

End-to-end path:
  search "4bit dflash qwen 35b"
  get recipe:qwen35-35b-a3b-dflash-1node
  load <recipe_id> --dry-run
  load <recipe_id>                 # POST asmi /serve/load

SoT: records.jsonl + bm25_index (from serve-configs). Live serving always asmi.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from build_bm25 import load_index, load_records, search as bm25_search  # noqa: E402
from ingest_serve_configs import recipe_to_serve_request  # noqa: E402

DEFAULT_RECORDS = ROOT / "records.jsonl"
DEFAULT_INDEX = ROOT / "bm25_index" / "index.pkl"
ASMI_URL = os.getenv("ASMI_URL", "http://127.0.0.1:9090")
# Guarded load choke point (r1o web) — preferred over raw asmi.
R1O_URL = os.getenv("R1O_URL", "http://localhost:59408")


def _find_record(records: list[dict], recipe_id: str) -> dict | None:
    rid = recipe_id.removeprefix("recipe:")
    for r in records:
        if r.get("recipe_id") == rid or r.get("id") == recipe_id or r.get("id") == f"recipe:{rid}":
            return r
        # also allow serve-config name match
        src = (r.get("source") or {}).get("serve_config_name")
        if src == rid or src == recipe_id:
            return r
    return None


def cmd_search(args: argparse.Namespace) -> int:
    if not args.index.exists() or not args.records.exists():
        print("missing index/records — run: python3 ingest_serve_configs.py && python3 build_bm25.py --rebuild", file=sys.stderr)
        return 1
    idx = load_index(args.index)
    hits = bm25_search(idx, args.query, top_k=args.top_k)
    if args.json:
        print(json.dumps(hits, indent=2))
        return 0
    if not hits:
        print("no hits")
        return 1
    for h in hits:
        acc = ",".join(h.get("accels") or []) or "-"
        rid = h.get("recipe_id") or h.get("id") or "?"
        kind = h.get("kind") or "serve_recipe"
        if kind != "serve_recipe":
            print(
                f"#{h['rank']:<2} {h['score']:7.3f}  {str(rid)[:50]:<50} "
                f"kind={kind}  {(h.get('title') or h.get('name') or h.get('search_text') or '')[:60]}"
            )
            continue
        print(
            f"#{h['rank']:<2} {h['score']:7.3f}  {str(rid):<42} "
            f"q={h.get('quant') or '?':8} {h.get('engine')}/{h.get('backend')} "
            f"accel=[{acc}]  {h.get('name')}"
        )
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    records = load_records(args.records)
    rec = _find_record(records, args.recipe_id)
    if not rec:
        print(f"not found: {args.recipe_id}", file=sys.stderr)
        return 1
    hostfile = args.hostfile
    if rec.get("serve", {}).get("backend") == "jaccl" and not hostfile:
        hostfile = str(Path.home() / ".r1o" / "hostfiles" / "default.json")
    try:
        body = recipe_to_serve_request(rec, hostfile=hostfile if rec.get("serve", {}).get("backend") == "jaccl" else None)
    except ValueError as e:
        print(f"materialize error: {e}", file=sys.stderr)
        return 1

    # Optional port override for load tests
    if args.port:
        body["port"] = args.port

    out = {
        "recipe_id": rec.get("recipe_id"),
        "kind": rec.get("kind"),
        "artifact": rec.get("artifact"),
        "quant": rec.get("quant"),
        "serve": rec.get("serve"),
        "accelerators": rec.get("accelerators"),
        "hardware": rec.get("hardware"),
        "benchmarks": rec.get("benchmarks"),
        "serve_request": body,
        "asmi_body": {k: v for k, v in body.items() if k != "recipe_id"},
        "source": "model-kb",
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


def _asmi_json(method: str, path: str, body: dict | None = None, timeout: float = 120.0) -> tuple[int, dict | list | str]:
    url = ASMI_URL.rstrip("/") + path
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


def cmd_serving(_: argparse.Namespace) -> int:
    code, data = _asmi_json("GET", "/serve/status", timeout=5)
    print(json.dumps({"status": code, "asmi": data}, indent=2, default=str))
    return 0 if code == 200 else 1


def cmd_load(args: argparse.Namespace) -> int:
    records = load_records(args.records)
    rec = _find_record(records, args.recipe_id)
    if not rec:
        print(f"not found: {args.recipe_id}", file=sys.stderr)
        return 1

    backend = (rec.get("serve") or {}).get("backend") or "single"
    if backend not in ("single", "jaccl"):
        backend = "single"
    hostfile = args.hostfile
    if backend == "jaccl":
        hostfile = hostfile or str(Path.home() / ".r1o" / "hostfiles" / "default.json")
        if not Path(hostfile).expanduser().exists() and not args.dry_run:
            print(f"jaccl hostfile missing: {hostfile}", file=sys.stderr)
            return 1

    try:
        body = recipe_to_serve_request(rec, hostfile=hostfile if backend == "jaccl" else None)
    except ValueError as e:
        print(f"materialize error: {e}", file=sys.stderr)
        return 1

    if args.port:
        body["port"] = int(args.port)

    # Hard bans
    if body.get("backend") == "auto":
        print("REFUSE: backend=auto", file=sys.stderr)
        return 1
    if "model_path" not in body:
        print("REFUSE: missing model_path", file=sys.stderr)
        return 1

    asmi_body = {k: v for k, v in body.items() if k != "recipe_id"}
    recipe_name = rec.get("recipe_id") or (rec.get("source") or {}).get("serve_config_name")
    # Only explicit --direct bypasses the r1o guarded apply path.
    # --port alone is recorded on the body for dry-run / direct tests, but
    # live loads without --direct always go through guardedServeLoad.
    use_direct = bool(getattr(args, "direct", False))

    result = {
        "recipe_id": recipe_name,
        "asmi_body": asmi_body,
        "dry_run": bool(args.dry_run),
        "via": "dry-run" if args.dry_run else ("asmi-direct" if use_direct else "r1o-guarded-apply"),
    }

    if args.dry_run:
        mp = str(asmi_body.get("model_path") or "")
        expanded = Path.home() / mp[2:] if mp.startswith("~/") else Path(mp).expanduser()
        result["model_path_exists"] = expanded.exists()
        result["model_path_resolved"] = str(expanded)
        result["guarded_apply_url"] = f"{R1O_URL}/api/hermes/serve-configs/{recipe_name}/apply"
        result["ok"] = True
        result["note"] = "dry-run only — no load call"
        print(json.dumps(result, indent=2, default=str))
        return 0

    # Live load: prefer r1o apply (guardedServeLoad). Only --direct → raw asmi.
    if use_direct:
        # asmi selects slot via ?port= query, not body.port (daemon.rs).
        port = int(asmi_body.get("port") or 19080)
        path = f"/serve/load?port={port}"
        result["asmi_url"] = f"{ASMI_URL}{path}"
        code, data = _asmi_json("POST", path, asmi_body, timeout=float(args.timeout))
        result["http_status"] = code
        result["asmi"] = data
        result["ok"] = code in (200, 201)
    else:
        url = f"{R1O_URL.rstrip('/')}/api/hermes/serve-configs/{recipe_name}/apply"
        result["apply_url"] = url
        # Optional port override still goes through guardedServeLoad.
        apply_payload: dict = {}
        if args.port is not None:
            apply_payload["port"] = int(args.port)
        req = urllib.request.Request(
            url,
            data=json.dumps(apply_payload).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=float(args.timeout)) as resp:
                raw = resp.read().decode()
                code = resp.status
                try:
                    data = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    data = raw
        except urllib.error.HTTPError as e:
            code = e.code
            raw = e.read().decode() if e.fp else str(e)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = raw
        except Exception as e:  # noqa: BLE001
            code, data = 0, str(e)
        result["http_status"] = code
        result["apply"] = data
        result["ok"] = code in (200, 201) and isinstance(data, dict) and data.get("ok") is True

    sc, status = _asmi_json("GET", "/serve/status", timeout=5)
    result["serve_status_http"] = sc
    if isinstance(status, dict):
        port = asmi_body.get("port")
        match = [s for s in (status.get("servers") or []) if s.get("port") == port]
        result["port_status"] = match[0] if match else None
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


def main() -> int:
    ap = argparse.ArgumentParser(prog="model_kb")
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_s = sub.add_parser("search", help="BM25 search recipes")
    p_s.add_argument("query")
    p_s.add_argument("--top-k", type=int, default=8)
    p_s.add_argument("--json", action="store_true")
    p_s.set_defaults(func=cmd_search)

    p_g = sub.add_parser("get", help="Get recipe + ServeRequest body")
    p_g.add_argument("recipe_id")
    p_g.add_argument("--hostfile", default=None)
    p_g.add_argument("--port", type=int, default=None)
    p_g.set_defaults(func=cmd_get)

    p_l = sub.add_parser(
        "load",
        help="Load recipe via r1o guarded apply (default) or asmi --direct",
    )
    p_l.add_argument("recipe_id")
    p_l.add_argument("--dry-run", action="store_true")
    p_l.add_argument(
        "--direct",
        action="store_true",
        help="POST asmi /serve/load directly (bypasses r1o guard)",
    )
    p_l.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override port on asmi body (dry-run always; live only with --direct)",
    )
    p_l.add_argument("--hostfile", default=None)
    p_l.add_argument("--timeout", type=float, default=180.0)
    p_l.set_defaults(func=cmd_load)

    p_v = sub.add_parser("serving", help="Live asmi /serve/status")
    p_v.set_defaults(func=cmd_serving)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
