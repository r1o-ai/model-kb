#!/usr/bin/env python3
"""matrix_report.py — multi-dimensional views over experiments.jsonl.

Views:
  coverage  — which (accel_set × context) cells exist for a campaign/artifact
  compound  — best tok/s by accelerator set (fixed workload if --context)
  workload  — same X across contexts
  pareto    — tok/s vs peak_ram (when present)

Usage:
  python3 matrix_report.py --campaign c --view coverage
  python3 matrix_report.py --artifact Qwen3.6 --view compound --context 55000
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_LOG = ROOT / "experiments.jsonl"


def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def accel_key(axes: dict[str, Any]) -> str:
    acc = list(axes.get("accelerators") or [])
    if not acc:
        return "∅ baseline"
    return "+".join(sorted(acc))


def filter_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    out = rows
    if args.campaign:
        out = [r for r in out if r.get("campaign") == args.campaign]
    if args.artifact:
        out = [
            r
            for r in out
            if args.artifact.lower()
            in str((r.get("axes") or {}).get("artifact") or "").lower()
        ]
    if args.status:
        out = [r for r in out if r.get("status") == args.status]
    return out


def view_coverage(rows: list[dict]) -> None:
    # cell = (accel_key, context) -> statuses
    cells: dict[tuple[str, Any], list[str]] = defaultdict(list)
    for r in rows:
        ax = r.get("axes") or {}
        ctx = (ax.get("workload") or {}).get("context_len")
        cells[(accel_key(ax), ctx)].append(str(r.get("status")))
    print(f"{'X accelerators':<36} {'ctx':>8}  statuses")
    print("-" * 70)
    for (xk, ctx), st in sorted(cells.items(), key=lambda x: (str(x[0][1]), x[0][0])):
        print(f"{xk:<36} {str(ctx):>8}  {','.join(st)}")
    print(f"\n# cells={len(cells)} rows={len(rows)}")


def view_compound(rows: list[dict], context: int | None) -> None:
    best: dict[str, dict] = {}
    for r in rows:
        if r.get("status") != "done":
            continue
        ax = r.get("axes") or {}
        w = ax.get("workload") or {}
        if context is not None and w.get("context_len") != context:
            continue
        m = r.get("metrics") or {}
        tps = m.get("agg_tps")
        if tps is None:
            continue
        k = accel_key(ax)
        prev = best.get(k)
        if not prev or float(tps) > float((prev.get("metrics") or {}).get("agg_tps") or 0):
            best[k] = r
    print(f"{'X accelerators':<36} {'tps':>8} {'base':>8} {'accept':>8}  artifact")
    print("-" * 90)
    for k, r in sorted(best.items(), key=lambda kv: -float((kv[1].get("metrics") or {}).get("agg_tps") or 0)):
        m = r.get("metrics") or {}
        ax = r.get("axes") or {}
        print(
            f"{k:<36} {m.get('agg_tps'):>8} {m.get('baseline_tps') or '-':>8} "
            f"{m.get('acceptance_pct') or '-':>8}  {ax.get('artifact')}"
        )


def view_workload(rows: list[dict]) -> None:
    # X -> ctx -> best tps
    grid: dict[str, dict[Any, float]] = defaultdict(dict)
    for r in rows:
        if r.get("status") != "done":
            continue
        ax = r.get("axes") or {}
        m = r.get("metrics") or {}
        tps = m.get("agg_tps")
        if tps is None:
            continue
        xk = accel_key(ax)
        ctx = (ax.get("workload") or {}).get("context_len")
        prev = grid[xk].get(ctx)
        if prev is None or float(tps) > prev:
            grid[xk][ctx] = float(tps)
    ctxs = sorted({c for g in grid.values() for c in g.keys()}, key=lambda x: (x is None, x or 0))
    header = f"{'X':<32}" + "".join(f"{str(c):>10}" for c in ctxs)
    print(header)
    print("-" * len(header))
    for xk in sorted(grid.keys()):
        line = f"{xk:<32}"
        for c in ctxs:
            v = grid[xk].get(c)
            line += f"{v:>10.1f}" if v is not None else f"{'—':>10}"
        print(line)


def view_pareto(rows: list[dict]) -> None:
    pts = []
    for r in rows:
        if r.get("status") != "done":
            continue
        m = r.get("metrics") or {}
        tps = m.get("agg_tps")
        ram = m.get("peak_ram_gb")
        if tps is None:
            continue
        pts.append((float(tps), float(ram) if ram is not None else None, r))
    pts.sort(key=lambda x: -x[0])
    print(f"{'tps':>8} {'ram_gb':>8}  X  artifact")
    for tps, ram, r in pts:
        ax = r.get("axes") or {}
        print(
            f"{tps:>8.1f} {ram if ram is not None else '—':>8}  "
            f"{accel_key(ax)}  {ax.get('artifact')}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(prog="matrix_report")
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--campaign", default=None)
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--status", default=None)
    ap.add_argument(
        "--view",
        choices=["coverage", "compound", "workload", "pareto"],
        default="coverage",
    )
    ap.add_argument("--context", type=int, default=None, help="for compound view")
    args = ap.parse_args()
    rows = filter_rows(load(Path(args.log)), args)
    if not rows:
        print("no experiments match filters")
        return 0
    if args.view == "coverage":
        view_coverage(rows)
    elif args.view == "compound":
        view_compound(rows, args.context)
    elif args.view == "workload":
        view_workload(rows)
    else:
        view_pareto(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
