#!/usr/bin/env python3
"""kb-guard — refuse destructive model-kb ingests that would silently delete records.

THE ORDERING TRAP
-----------------
`ingest_serve_configs.py` opens records.jsonl with mode "w" — it writes ONLY
serve_recipe rows. Every cloud_context, research_note, hf_family and hf_card record
is destroyed. There is no warning, no diff, and no non-zero exit: the run looks like
a success and the corpus silently loses most of its content.

Every other ingest MERGES (read → merge → write), so the trap fires exactly once, on
the one script people run first because it is listed first.

This guard makes the destruction visible BEFORE it happens:
  * snapshots the corpus,
  * predicts what a clobbering run would remove,
  * refuses unless the loss is acknowledged.

Usage:
    kb-guard.py check   <records.jsonl>            # what's in there now
    kb-guard.py plan    <records.jsonl> recipes    # what a run would destroy
    kb-guard.py backup  <records.jsonl>            # timestamped snapshot
    kb-guard.py verify  <records.jsonl> [--min N]  # post-ingest sanity gate

Exit codes:
    0  safe / healthy
    2  usage error
    3  corpus missing or unreadable
    4  DESTRUCTIVE — the planned run would delete non-recipe records
    5  corpus regressed (verify found fewer records than the floor)
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import time
from pathlib import Path

# Only this script targets records.jsonl with open("w"). The hf_* scripts also use
# "w" but write to their OWN enrichments_*.jsonl, so they are not corpus-destructive.
CLOBBERS_CORPUS = {"recipes": "ingest_serve_configs.py"}
CLOBBER_KEEPS = {"serve_recipe"}


def load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"corpus not found: {path}", file=sys.stderr)
        sys.exit(3)
    out = []
    for n, line in enumerate(path.open(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  warn: line {n} is not valid JSON — skipped", file=sys.stderr)
    return out


def counts(records: list[dict]) -> collections.Counter:
    return collections.Counter(r.get("kind", "?") for r in records)


def show(records: list[dict]) -> None:
    c = counts(records)
    for k, v in c.most_common():
        print(f"  {k:<16} {v}")
    print(f"  {'TOTAL':<16} {sum(c.values())}")


def cmd_check(a) -> int:
    recs = load(a.records)
    print(f"=== {a.records} ===")
    show(recs)
    return 0


def cmd_plan(a) -> int:
    recs = load(a.records)
    c = counts(recs)
    script = CLOBBERS_CORPUS.get(a.phase)
    if not script:
        print(f"phase '{a.phase}' MERGES into the corpus — safe, no records lost.")
        return 0

    doomed = {k: v for k, v in c.items() if k not in CLOBBER_KEEPS}
    lost = sum(doomed.values())
    print(f"=== plan: {a.phase} ({script}) ===")
    print(f"This script opens {a.records.name} with mode 'w' and writes ONLY "
          f"{'/'.join(sorted(CLOBBER_KEEPS))} rows.\n")
    if not lost:
        print("✓ corpus holds only recipe rows — nothing else to lose. Safe.")
        return 0
    print("⛔ DESTRUCTIVE — these records would be DELETED:")
    for k, v in sorted(doomed.items(), key=lambda kv: -kv[1]):
        print(f"     {k:<16} {v}")
    print(f"     {'TOTAL LOST':<16} {lost}  of {sum(c.values())}")
    print("\n   Safe rebuild order (recipes FIRST, then the merging phases re-add):")
    print("     1. recipes      (clobbers — must be first)")
    print("     2. cloud        (merge)")
    print("     3. research     (merge)")
    print("     4. vendor       (merge)")
    print("     5. join + index")
    print("\n   Or snapshot first:  kb-guard.py backup " + str(a.records))
    return 4


def cmd_backup(a) -> int:
    if not a.records.exists():
        print(f"corpus not found: {a.records}", file=sys.stderr)
        return 3
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = a.records.with_suffix(a.records.suffix + f".bak-{stamp}")
    shutil.copy2(a.records, dest)
    recs = load(a.records)
    print(f"✓ snapshot: {dest}  ({len(recs)} records)")
    return 0


def cmd_verify(a) -> int:
    recs = load(a.records)
    c = counts(recs)
    total = sum(c.values())
    print(f"=== verify {a.records} ===")
    show(recs)
    problems = []
    if a.min is not None and total < a.min:
        problems.append(f"total {total} is below floor {a.min}")
    # A corpus that is 100% recipes almost always means the trap fired.
    if total and set(c) <= CLOBBER_KEEPS:
        problems.append("corpus contains ONLY recipe rows — the clobber trap likely fired")
    ids = [r.get("id") for r in recs if r.get("id")]
    dupes = [i for i, n in collections.Counter(ids).items() if n > 1]
    if dupes:
        problems.append(f"{len(dupes)} duplicate id(s), e.g. {dupes[:3]}")
    if problems:
        print("\n⛔ PROBLEMS:")
        for p in problems:
            print(f"   - {p}")
        return 5
    print("\n✓ corpus looks healthy")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="kb-guard")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check"); p.add_argument("records", type=Path); p.set_defaults(fn=cmd_check)
    p = sub.add_parser("plan")
    p.add_argument("records", type=Path)
    p.add_argument("phase", choices=["recipes", "cloud", "research", "vendor", "hf", "family"])
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("backup"); p.add_argument("records", type=Path); p.set_defaults(fn=cmd_backup)
    p = sub.add_parser("verify")
    p.add_argument("records", type=Path)
    p.add_argument("--min", type=int, default=None, help="minimum expected record count")
    p.set_defaults(fn=cmd_verify)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
