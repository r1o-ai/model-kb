#!/usr/bin/env python3
"""model_kb.py — search / get / load serve recipes (Phase C).

End-to-end path:
  search "4bit dflash qwen 35b"
  get recipe:qwen35-35b-a3b-dflash-1node
  load <recipe_id> --dry-run
  load <recipe_id> --node <machine>   # POST r1o /api/models/kb/recipes/<id>/serve (lib/r1o_serve.py)

SoT: records.jsonl + bm25_index (serve recipes from the recipe store). Live serving always asmi.
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
from lib.r1o_serve import request_problems, serve_body, serve_recipe, serve_url  # noqa: E402
from serve_recipes import asmi_body_of, materialize_load  # noqa: E402

DEFAULT_RECORDS = ROOT / "records.jsonl"
DEFAULT_INDEX = ROOT / "bm25_index" / "index.pkl"
ASMI_URL = os.getenv("ASMI_URL", "http://127.0.0.1:9090")
# r1o web: the recipe serve route (launch consent + fit + load guards) — the load path.
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
        print("missing index/records — run: python3 pipeline.py", file=sys.stderr)
        return 1
    hits = None
    if getattr(args, "hybrid", False):
        try:
            from hybrid_search import HybridModelKb
            hits = HybridModelKb(args.index).search(args.query, top_k=args.top_k)
        except Exception as e:  # fail open to BM25 — search must never die on the fancy path
            print(f"[hybrid] unavailable ({e}) — BM25 only", file=sys.stderr)
            hits = None
    if hits is None:
        idx = load_index(args.index)
        hits = bm25_search(idx, args.query, top_k=args.top_k)
    if args.json:
        print(json.dumps(hits, indent=2))
        return 0
    if not hits:
        print("no hits")
        return 1
    for rank_i, h in enumerate(hits, 1):
        acc = ",".join(h.get("accels") or []) or "-"
        rid = h.get("recipe_id") or h.get("id") or "?"
        kind = h.get("kind") or "serve_recipe"
        # Display position is the rank truth — hybrid hits can carry a stale
        # BM25 rank from the fused dict; never show it out of order.
        rank = rank_i
        if kind != "serve_recipe":
            print(
                f"#{rank:<2} {h['score']:7.3f}  {str(rid)[:50]:<50} "
                f"kind={kind}  {(h.get('title') or h.get('name') or h.get('search_text') or '')[:60]}"
            )
            continue
        print(
            f"#{rank:<2} {h['score']:7.3f}  {str(rid):<42} "
            f"q={h.get('quant') or '?':8} {h.get('engine')}/{h.get('backend')} "
            f"accel=[{acc}]  {h.get('name')}"
        )
    return 0


DEFAULT_HOSTFILE = Path.home() / ".r1o" / "hostfiles" / "default.json"


def _plan(rec: dict, args: argparse.Namespace) -> dict:
    """materialize_load + the CLI overrides. A recipe's own serve.hostfile wins
    over the default; an explicit --hostfile / --port wins over both."""
    plan = materialize_load(rec, hostfile=str(DEFAULT_HOSTFILE))
    body = plan["asmi_body"]
    if getattr(args, "hostfile", None) and body.get("backend") == "jaccl":
        body["hostfile"] = args.hostfile
    if getattr(args, "port", None):
        plan["port"] = int(args.port)
    return plan


def cmd_get(args: argparse.Namespace) -> int:
    records = load_records(args.records)
    rec = _find_record(records, args.recipe_id)
    if not rec:
        print(f"not found: {args.recipe_id}", file=sys.stderr)
        return 1
    plan = _plan(rec, args)

    out = {
        "recipe_id": rec.get("recipe_id"),
        "kind": rec.get("kind"),
        "artifact": rec.get("artifact"),
        "quant": rec.get("quant"),
        "serve": rec.get("serve"),
        "accelerators": rec.get("accelerators"),
        "hardware": rec.get("hardware"),
        "benchmarks": rec.get("benchmarks"),
        # asmi POST /serve/load?port=<port> with exactly asmi_body.
        "asmi_body": plan["asmi_body"],
        "port": plan["port"],
        "loadable": plan["loadable"],
        "problems": plan["problems"],
        "warnings": plan["warnings"],
        "serve_keys_not_sent": plan["not_sent"],
        "source": "model-kb",
    }
    print(json.dumps(out, indent=2, default=str))
    if plan["problems"]:
        print("not loadable: " + "; ".join(plan["problems"]), file=sys.stderr)
        return 1
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

    plan = _plan(rec, args)
    if plan["problems"]:
        print("materialize error: not loadable: " + "; ".join(plan["problems"]), file=sys.stderr)
        return 1
    body = plan["asmi_body"]
    hostfile = body.get("hostfile")
    if body.get("backend") == "jaccl" and hostfile and not Path(hostfile).expanduser().exists() and not args.dry_run:
        print(f"jaccl hostfile missing: {hostfile}", file=sys.stderr)
        return 1

    # Hard bans
    if body.get("backend") == "auto":
        print("REFUSE: backend=auto", file=sys.stderr)
        return 1
    if "model_path" not in body:
        print("REFUSE: missing model_path", file=sys.stderr)
        return 1

    asmi_body = asmi_body_of(body)
    port = plan["port"]
    recipe_name = rec.get("recipe_id") or (rec.get("source") or {}).get("serve_config_name")
    # Only explicit --direct skips r1o. Without it the load is r1o's recipe
    # serve route, sent once to the machine --node names (never a guessed one).
    use_direct = bool(getattr(args, "direct", False))
    node = getattr(args, "node", None)

    result = {
        "recipe_id": recipe_name,
        "asmi_body": asmi_body,
        "port": port,
        "warnings": plan["warnings"],
        "serve_keys_not_sent": plan["not_sent"],
        "dry_run": bool(args.dry_run),
        "via": "dry-run" if args.dry_run else ("asmi-direct" if use_direct else "r1o-recipe-serve"),
    }
    if not use_direct:
        result["serve_url"] = serve_url(R1O_URL, recipe_name)
        result["request"] = serve_body(node, args.port)
        would_refuse = request_problems(rec, node)
        if would_refuse:
            result["request_problems"] = would_refuse

    if args.dry_run:
        mp = str(asmi_body.get("model_path") or "")
        expanded = Path.home() / mp[2:] if mp.startswith("~/") else Path(mp).expanduser()
        result["model_path_exists"] = expanded.exists()
        result["model_path_resolved"] = str(expanded)
        result["ok"] = True
        result["note"] = "dry-run only — no load call"
        print(json.dumps(result, indent=2, default=str))
        return 0

    # Live load: r1o's recipe serve route. Only --direct → raw asmi on this node.
    if use_direct:
        # asmi selects the slot via ?port= (daemon.rs); without one it uses the
        # engine record's default port — model-kb never guesses a number.
        path = f"/serve/load?port={int(port)}" if port else "/serve/load"
        result["asmi_url"] = f"{ASMI_URL}{path}"
        code, data = _asmi_json("POST", path, asmi_body, timeout=float(args.timeout))
        result["http_status"] = code
        result["asmi"] = data
        result["ok"] = code in (200, 201)
        sc, status = _asmi_json("GET", "/serve/status", timeout=5)
        result["serve_status_http"] = sc
        if isinstance(status, dict):
            match = [s for s in (status.get("servers") or []) if s.get("port") == port]
            result["port_status"] = match[0] if match else None
    else:
        # r1o's answer, whole: a refusal keeps its error / problems / fit, and a
        # launch-approval refusal says where the operator can approve it.
        result.update(serve_recipe(R1O_URL, recipe_name, node=node, port=args.port, timeout=float(args.timeout)))
        if result.get("error"):
            print(result["error"], file=sys.stderr)
            for line in result.get("approve") or []:
                print(f"  - {line}", file=sys.stderr)

    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


def main() -> int:
    ap = argparse.ArgumentParser(prog="model_kb")
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_s = sub.add_parser("search", help="Search recipes (BM25; --hybrid adds RRF fusion with Qwen3-VL embeddings)")
    p_s.add_argument("query")
    p_s.add_argument("--top-k", type=int, default=8)
    p_s.add_argument("--json", action="store_true")
    p_s.add_argument("--hybrid", action="store_true", help="RRF-fuse BM25 with embedding ranks (needs bm25_index/vectors.npy; run embed_model_kb.py)")
    p_s.set_defaults(func=cmd_search)

    p_g = sub.add_parser("get", help="Get recipe + ServeRequest body")
    p_g.add_argument("recipe_id")
    p_g.add_argument("--hostfile", default=None)
    p_g.add_argument("--port", type=int, default=None)
    p_g.set_defaults(func=cmd_get)

    p_l = sub.add_parser(
        "load",
        help="Serve a recipe on a machine via r1o's recipe serve route (default) or asmi --direct",
    )
    p_l.add_argument("recipe_id")
    p_l.add_argument("--dry-run", action="store_true")
    p_l.add_argument(
        "--direct",
        action="store_true",
        help="POST this node's asmi /serve/load directly (skips r1o: no launch consent, no fit check)",
    )
    p_l.add_argument(
        "--node",
        default=None,
        help="Machine to serve on. Required for a single-machine recipe (r1o never picks one); "
             "refused for a multi-machine recipe, which loads on the r1o coordinator",
    )
    p_l.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override the recipe's serve.port (sent to r1o as port; with --direct it is asmi's ?port=)",
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
