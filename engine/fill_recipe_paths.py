#!/usr/bin/env python3
"""fill_recipe_paths.py — set serve-config model.path from live cluster inventory.

SoT: graph-indexer entities (entity_type in model, model_instance).
Never invent a path. Exact name / HF id tail, or same family + overlapping
quant token. Quant-only folders (4bit, 6bit, …) are rejected.

Usage:
  python3 fill_recipe_paths.py --dry-run
  python3 fill_recipe_paths.py            # atomic write to serve-configs.json
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

HOME = Path.home()
DEFAULT_CONFIG = HOME / ".r1o" / "serve-configs.json"
DEFAULT_DB = HOME / "infra" / "graph-indexer" / "index.db"

QUANT_FOLDER = {
    "2bit",
    "3bit",
    "4bit",
    "6bit",
    "8bit",
    "bf16",
    "fp8",
    "fp16",
    "mxfp4",
    "nvfp4",
}

QUANT_STRIP = re.compile(
    r"(?:-?(?:mlx-?)?(?:\d+(?:\.\d+)?bit|\d+(?:\.\d+)?bpw|mxfp4|nvfp4|bf16|"
    r"fp16|fp8|mixed-?\d*-?\d*|iq\d+\w*|q\d+_k\w*|abliterated|dynamic).*)$",
    re.I,
)

QUANT_TOKENS = (
    "mxfp4",
    "nvfp4",
    "4bit",
    "6bit",
    "8bit",
    "3bit",
    "2bit",
    "bf16",
    "fp8",
    "fp16",
    "mixed",
    "3.6bit",
    "3.5bpw",
    "2.7bpw",
    "iq2",
    "q4",
    "q2",
)


def norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def strip_quant(s: str) -> str:
    prev = None
    cur = (s or "").strip()
    while prev != cur:
        prev = cur
        cur = QUANT_STRIP.sub("", cur)
    return cur.strip("-_ ")


def quant_tokens(*parts: str) -> set[str]:
    blob = " ".join(parts).lower()
    return {t.replace(".", "") for t in QUANT_TOKENS if t in blob}


def portable_path(path: str) -> str:
    models = str(HOME / "Models")
    if path.startswith(models + "/") or path == models:
        return "~/" + str(Path(path).relative_to(HOME))
    return path


def load_inventory(db: Path) -> list[dict]:
    if not db.exists():
        raise FileNotFoundError(f"graph-indexer db missing: {db}")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = []
    for r in con.execute(
        "select name, path, node, entity_type from entities "
        "where entity_type in ('model', 'model_instance')"
    ):
        name = r["name"] or ""
        path = r["path"] or ""
        if not path:
            continue
        if name.lower() in QUANT_FOLDER or len(name) <= 5:
            continue
        if "@" in name:
            continue
        rows.append({"name": name, "path": path, "node": r["node"], "kind": r["entity_type"]})
    return rows


def match(model: dict, inventory: list[dict]) -> list[dict]:
    name = model.get("name") or ""
    hf_tail = (model.get("hf_id") or "").split("/")[-1]
    quant = model.get("quant") or ""
    n_name = norm(name)
    n_hf = norm(hf_tail)
    n_fam = norm(strip_quant(name)) or n_name
    q_cfg = quant_tokens(name, quant, hf_tail)
    hits: list[tuple[int, dict]] = []
    for m in inventory:
        n_inv = norm(m["name"])
        n_base = norm(Path(m["path"]).name)
        n_inv_fam = norm(strip_quant(m["name"]))
        score = 0
        if n_name and n_name in (n_inv, n_base):
            score = 100
        elif n_hf and n_hf in (n_inv, n_base):
            score = 98
        elif n_fam and n_inv_fam == n_fam and len(n_fam) >= 8:
            score = 90
        else:
            continue
        q_inv = quant_tokens(m["name"], m["path"])
        if q_cfg:
            if q_cfg & q_inv:
                score += 5
            elif score < 100:
                # Config named a quant; inventory must share it. Fail closed.
                continue
        hits.append((score, m))
    hits.sort(key=lambda x: -x[0])
    by_path: dict[str, tuple[int, dict]] = {}
    for score, m in hits:
        by_path.setdefault(m["path"], (score, m))
    uniq = sorted(by_path.values(), key=lambda x: -x[0])
    if not uniq:
        return []
    top = uniq[0][0]
    cands = [m for s, m in uniq if s == top]
    # Prefer internal SSD over NAS cold-store; more specific (longer) name.
    cands.sort(key=lambda m: ("cold-store" in (m["path"] or ""), -len(m["name"])))
    internal = [m for m in cands if "cold-store" not in (m["path"] or "")]
    pool = internal or cands
    paths = {m["path"] for m in pool}
    if len(paths) == 1:
        return [pool[0]]
    return pool


def apply_fills(cfg: dict, inventory: list[dict], *, write: bool) -> list[dict]:
    report = []
    for c in cfg.get("configs") or []:
        model = c.get("model") or {}
        if model.get("path"):
            continue
        cands = match(model, inventory)
        row = {
            "recipe": c.get("name"),
            "model": model.get("name"),
            "quant": model.get("quant"),
            "hits": [{"node": m["node"], "path": m["path"], "name": m["name"]} for m in cands],
        }
        if len(cands) == 1 or (
            len(cands) > 1 and all(m["path"] == cands[0]["path"] for m in cands)
        ):
            path = portable_path(cands[0]["path"])
            row["fill"] = path
            row["node"] = cands[0]["node"]
            if write:
                model["path"] = path
                c["model"] = model
        elif not cands:
            row["fill"] = None
            row["reason"] = "no live inventory match"
        else:
            row["fill"] = None
            row["reason"] = "ambiguous"
        report.append(row)
    return report


def atomic_write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.config.exists():
        print(f"missing {args.config}")
        return 1
    cfg = json.loads(args.config.read_text())
    inventory = load_inventory(args.db)
    report = apply_fills(cfg, inventory, write=not args.dry_run)
    filled = [r for r in report if r.get("fill")]
    skipped = [r for r in report if not r.get("fill")]
    print(f"inventory={len(inventory)} missing_path={len(report)} fill={len(filled)} skip={len(skipped)}")
    for r in filled:
        print(f"  FILL {r['recipe']} -> {r['fill']}  ({r.get('node')})")
    for r in skipped:
        print(f"  SKIP {r['recipe']} name={r['model']} q={r['quant']} ({r.get('reason')})")
        if r.get("hits") and r.get("reason") == "ambiguous":
            for h in r["hits"][:4]:
                print(f"       {h['node']} {h['path']}")
    if args.dry_run:
        print("dry-run — no write")
        return 0
    atomic_write(args.config, cfg)
    print(f"wrote {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
