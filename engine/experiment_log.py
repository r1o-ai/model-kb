#!/usr/bin/env python3
"""experiment_log.py — append-only multi-dimensional model optimization log.

Each row is a point in (artifact × quant × engine × backend × accelerators ×
hardware × workload) with optional metrics. Never overwrites done metrics.

Usage:
  python3 experiment_log.py add --campaign c --artifact Qwen3.6-35B-A3B-4bit \\
      --engine mlx_lm --backend single --accels dflash,turboquant \\
      --context 55000 --concurrency 1 --agg-tps 22.9 --baseline-tps 29.1 \\
      --status done --hypothesis "TQ+DFlash at 55k"

  python3 experiment_log.py list --campaign c
  python3 experiment_log.py plan-wave --campaign c --artifact ... --accels dflash,turboquant,ddtree
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_LOG = ROOT / "experiments.jsonl"

# Known compoundable accelerators (extend as discovered)
DEFAULT_ACCEL_CATALOG = [
    "dflash",
    "turboquant",
    "ddtree",
    "mtp",
    "mtplx",
    "jaccl",
    "prompt_cache",
    "continuous_batching",
    "triattention",
    "msa",
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]


def load_experiments(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def append_experiment(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def make_id(campaign: str, axes: dict[str, Any], ts: str) -> str:
    art = _slug(str(axes.get("artifact") or "model"))
    acc = "-".join(axes.get("accelerators") or ["baseline"]) or "baseline"
    w = axes.get("workload") or {}
    ctx = w.get("context_len") or "x"
    day = ts[:10]
    camp = _slug(campaign or "x")[:16]
    return f"exp:{day}-{camp}-{art}-{_slug(acc)}-{ctx}-{uuid.uuid4().hex[:6]}"


def cmd_add(args: argparse.Namespace) -> int:
    accels = [a.strip() for a in (args.accels or "").split(",") if a.strip()]
    axes = {
        "artifact": args.artifact,
        "hf_id": args.hf_id,
        "quant": args.quant,
        "engine": args.engine,
        "backend": args.backend,
        "accelerators": accels,
        "hardware": {
            "nodes": [n.strip() for n in (args.nodes or "").split(",") if n.strip()],
            "chip": args.chip,
            "ram_gb": args.ram_gb,
        },
        "workload": {
            "context_len": args.context,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "task": args.task,
        },
    }
    metrics = {
        "agg_tps": args.agg_tps,
        "per_req_tps": args.per_req_tps,
        "baseline_tps": args.baseline_tps,
        "ttft_ms": args.ttft_ms,
        "acceptance_pct": args.acceptance,
        "peak_ram_gb": args.peak_ram,
        "prefill_s": args.prefill_s,
        "quality": args.quality,
        "notes": args.notes,
        "error": args.error,
    }
    # drop nulls in metrics for cleanliness
    metrics = {k: v for k, v in metrics.items() if v is not None}

    ts = _now()
    row: dict[str, Any] = {
        "experiment_id": args.experiment_id or make_id(args.campaign, axes, ts),
        "ts": ts,
        "status": args.status,
        "hypothesis": args.hypothesis,
        "parent_experiment_id": args.parent,
        "campaign": args.campaign,
        "axes": axes,
        "metrics": metrics,
        "baseline_ref": args.baseline_ref,
        "recipe_id": args.recipe_id,
        "source": {
            "type": args.source_type,
            "path": args.source_path,
            "span": args.source_span,
        },
        "confidence": args.confidence,
        "author": args.author,
        "supersedes": args.supersedes,
    }
    append_experiment(Path(args.log), row)
    print(json.dumps({"ok": True, "experiment_id": row["experiment_id"]}, indent=2))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    rows = load_experiments(Path(args.log))
    if args.campaign:
        rows = [r for r in rows if r.get("campaign") == args.campaign]
    if args.status:
        rows = [r for r in rows if r.get("status") == args.status]
    if args.artifact:
        rows = [
            r
            for r in rows
            if args.artifact.lower() in str((r.get("axes") or {}).get("artifact") or "").lower()
        ]
    for r in rows[-args.limit :]:
        ax = r.get("axes") or {}
        m = r.get("metrics") or {}
        print(
            f"{r.get('experiment_id')}\t{r.get('status')}\t"
            f"X={','.join(ax.get('accelerators') or []) or '∅'}\t"
            f"W={ax.get('workload', {}).get('context_len')}@c{ax.get('workload', {}).get('concurrency')}\t"
            f"tps={m.get('agg_tps')}\t{ax.get('artifact')}"
        )
    print(f"# {len(rows)} rows")
    return 0


def cmd_plan_wave(args: argparse.Namespace) -> int:
    """Emit planned rows for baseline + singles + pairs (not full power set)."""
    base_accels = [a.strip() for a in args.accels.split(",") if a.strip()]
    combos: list[list[str]] = [[]]  # baseline
    for a in base_accels:
        combos.append([a])
    if not args.singles_only:
        for a, b in itertools.combinations(base_accels, 2):
            combos.append([a, b])
    if args.triples:
        for trip in itertools.combinations(base_accels, 3):
            combos.append(list(trip))

    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    planned = 0
    for ctx in contexts:
        for acc in combos:
            # re-use add path
            ns = argparse.Namespace(
                log=args.log,
                campaign=args.campaign,
                artifact=args.artifact,
                hf_id=args.hf_id,
                quant=args.quant,
                engine=args.engine,
                backend=args.backend,
                accels=",".join(acc),
                nodes=args.nodes,
                chip=args.chip,
                ram_gb=args.ram_gb,
                context=ctx,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                task=args.task,
                agg_tps=None,
                per_req_tps=None,
                baseline_tps=None,
                ttft_ms=None,
                acceptance=None,
                peak_ram=None,
                prefill_s=None,
                quality=None,
                notes=None,
                error=None,
                status="planned",
                hypothesis=args.hypothesis
                or f"wave: X={acc or ['baseline']} @ ctx={ctx}",
                parent=None,
                baseline_ref=None,
                recipe_id=None,
                source_type="plan",
                source_path=None,
                source_span=None,
                confidence="planned",
                author=args.author,
                supersedes=None,
                experiment_id=None,
            )
            cmd_add(ns)
            planned += 1
    print(f"planned {planned} experiments → {args.log}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="experiment_log")
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", help="Append one experiment row")
    p.add_argument("--campaign", required=True)
    p.add_argument("--artifact", required=True)
    p.add_argument("--hf-id", default=None)
    p.add_argument("--quant", default="unknown")
    p.add_argument("--engine", default="mlx_lm")
    p.add_argument("--backend", default="single")
    p.add_argument("--accels", default="", help="comma-separated accelerator types")
    p.add_argument("--nodes", default="")
    p.add_argument("--chip", default=None)
    p.add_argument("--ram-gb", type=float, default=None)
    p.add_argument("--context", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--task", default="chat")
    p.add_argument("--agg-tps", type=float, default=None)
    p.add_argument("--per-req-tps", type=float, default=None)
    p.add_argument("--baseline-tps", type=float, default=None)
    p.add_argument("--ttft-ms", type=float, default=None)
    p.add_argument("--acceptance", type=float, default=None)
    p.add_argument("--peak-ram", type=float, default=None)
    p.add_argument("--prefill-s", type=float, default=None)
    p.add_argument("--quality", type=float, default=None)
    p.add_argument("--notes", default=None)
    p.add_argument("--error", default=None)
    p.add_argument("--status", default="planned", choices=["planned", "running", "done", "failed", "aborted"])
    p.add_argument("--hypothesis", default=None)
    p.add_argument("--parent", default=None)
    p.add_argument("--baseline-ref", default=None)
    p.add_argument("--recipe-id", default=None)
    p.add_argument("--source-type", default="manual")
    p.add_argument("--source-path", default=None)
    p.add_argument("--source-span", default=None)
    p.add_argument("--confidence", default="planned")
    p.add_argument("--author", default="ma")
    p.add_argument("--supersedes", default=None)
    p.add_argument("--experiment-id", default=None)
    p.set_defaults(func=cmd_add)

    pl = sub.add_parser("list", help="List experiments")
    pl.add_argument("--campaign", default=None)
    pl.add_argument("--status", default=None)
    pl.add_argument("--artifact", default=None)
    pl.add_argument("--limit", type=int, default=50)
    pl.set_defaults(func=cmd_list)

    pw = sub.add_parser("plan-wave", help="Plan baseline+singles+pairs for compound wave")
    pw.add_argument("--campaign", required=True)
    pw.add_argument("--artifact", required=True)
    pw.add_argument("--accels", required=True, help="pool of accels to compound")
    pw.add_argument("--contexts", default="512,8192,55000")
    pw.add_argument("--singles-only", action="store_true")
    pw.add_argument("--triples", action="store_true")
    pw.add_argument("--hf-id", default=None)
    pw.add_argument("--quant", default="4bit")
    pw.add_argument("--engine", default="mlx_lm")
    pw.add_argument("--backend", default="single")
    pw.add_argument("--nodes", default="")
    pw.add_argument("--chip", default=None)
    pw.add_argument("--ram-gb", type=float, default=None)
    pw.add_argument("--concurrency", type=int, default=1)
    pw.add_argument("--max-tokens", type=int, default=256)
    pw.add_argument("--task", default="chat")
    pw.add_argument("--hypothesis", default=None)
    pw.add_argument("--author", default="ma")
    pw.set_defaults(func=cmd_plan_wave)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
