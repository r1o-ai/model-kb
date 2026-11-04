#!/usr/bin/env python3
"""export_seed.py — anonymize live records.jsonl for the shipped seed.

Never copy ~/.r1o/model-kb/records.jsonl into the app or the public repo. The seed
is a derived view: no cluster hostnames, no tailnet names, no private IPs, no home
paths, no email addresses, no node inventories, no NAS paths.

Node names are read at export time from the live cluster file (`--cluster`,
env R1O_CLUSTER_JSON, default ~/.r1o/cluster.json) plus this machine's hostname.
No pinned host list: a node added to the cluster is scrubbed without a code change.
An unreadable or empty cluster file fails closed — nothing is written.

Usage:
  python3 export_seed.py
  python3 export_seed.py --out ~/Projects/model-kb-kit/data/records.seed.jsonl
  python3 export_seed.py --check-only   # leak-scan an existing seed
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = Path.home() / "Projects" / "model-kb-kit" / "data" / "records.seed.jsonl"
DEFAULT_CLUSTER = Path(os.environ.get("R1O_CLUSTER_JSON", Path.home() / ".r1o" / "cluster.json"))

EXIT_NO_INVENTORY = 3

# Shape-based patterns: these never need a list.
TAILNET_RE = re.compile(r"\b[\w-]+(?:\.[\w-]+)*\.ts\.net\b", re.I)
PRIVATE_IP_RE = re.compile(
    r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))"
    r"\.\d{1,3}\.\d{1,3}\b"
)
HOME_RE = re.compile(r"/(?:Users|home)/[^/\s\"']+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")


class InventoryUnavailable(RuntimeError):
    """The live node list could not be read; refuse to export."""


def load_node_names(cluster: Path) -> set[str]:
    """Hostnames from the live cluster file (+ this machine). Empty → InventoryUnavailable."""
    names: set[str] = set()
    try:
        data = json.loads(cluster.read_text())
    except (OSError, ValueError) as e:
        raise InventoryUnavailable(f"cannot read node inventory {cluster}: {e}") from e
    nodes = data.get("nodes", data) if isinstance(data, dict) else data
    rows = nodes.values() if isinstance(nodes, dict) else nodes
    keys = nodes.keys() if isinstance(nodes, dict) else []
    for row in rows:
        if isinstance(row, dict):
            for k in ("hostname", "name", "tailscale_hostname", "short_name"):
                if isinstance(row.get(k), str) and row[k].strip():
                    names.add(row[k].strip().split(".")[0].lower())
            for alias in row.get("aliases") or []:
                if isinstance(alias, str) and alias.strip():
                    names.add(alias.strip().split(".")[0].lower())
    names.update(str(k).split(".")[0].lower() for k in keys)
    if not names:
        raise InventoryUnavailable(f"no node names in {cluster}")
    me = socket.gethostname().split(".")[0].lower()
    if me:
        names.add(me)
    return names


class Anonymizer:
    def __init__(self, node_names: set[str]):
        if not node_names:
            raise InventoryUnavailable("empty node list")
        alt = "|".join(sorted((re.escape(n) for n in node_names), key=len, reverse=True))
        # Hostnames are lowercase; case-sensitive so prose like "Hugging Face Hub" survives.
        self.host_re = re.compile(rf"\b(?:{alt})\b")
        self.host_key_re = re.compile(rf"^(?:{alt})$", re.I)

    def portable_path(self, raw: str | None) -> str | None:
        if not raw:
            return None
        s = HOME_RE.sub("~", str(raw)).replace("\\", "/")
        if s.startswith("~/Models/") or s == "~/Models":
            return s
        marker = "/Models/"
        if marker in s:
            return "~/Models/" + s.split(marker, 1)[1]
        return None

    def scrub_str(self, s: str) -> str:
        s = TAILNET_RE.sub("a-node.tailnet", s)
        s = PRIVATE_IP_RE.sub("a-private-ip", s)
        s = HOME_RE.sub("~", s)
        s = EMAIL_RE.sub("redacted", s)
        s = self.host_re.sub("a-node", s)
        return s

    def scrub(self, obj: object, *, key: str | None = None) -> object:
        if isinstance(obj, str):
            if key == "path":
                return self.portable_path(obj)
            return self.scrub_str(obj)
        if isinstance(obj, list):
            if key == "nodes":
                return []
            return [self.scrub(x) for x in obj]
        if isinstance(obj, dict):
            out: dict = {}
            for k, v in obj.items():
                dk = "a-node" if self.host_key_re.match(str(k)) else k
                if k in {"asmi_path", "inventory_model_id"}:
                    out[dk] = None
                elif k == "nodes":
                    out[dk] = []
                elif k == "path":
                    out[dk] = self.portable_path(v) if isinstance(v, str) else None
                else:
                    out[dk] = self.scrub(v, key=k)
            return out
        return obj

    def leak_scan(self, blob: str) -> list[str]:
        hits: list[str] = []
        for label, rx in (("cluster hostname", self.host_re), ("tailnet name", TAILNET_RE),
                          ("private IP", PRIVATE_IP_RE), ("home path", HOME_RE), ("email", EMAIL_RE)):
            m = rx.search(blob)
            if m:
                hits.append(f"{label} ({m.group(0)!r})")
        return hits


def export(src: Path, dest: Path, anon: Anonymizer) -> int:
    rows: list[str] = []
    for line in src.open():
        line = line.strip()
        if not line:
            continue
        rows.append(json.dumps(anon.scrub(json.loads(line)), ensure_ascii=False, separators=(",", ":")))
    blob = "\n".join(rows) + ("\n" if rows else "")
    leaks = anon.leak_scan(blob)
    if leaks:
        print(f"FAIL: seed still contains {leaks} — refusing to write {dest}", file=sys.stderr)
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(blob)
    tmp.replace(dest)
    print(f"wrote {len(rows)} anonymized records → {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=ROOT / "records.jsonl")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cluster", type=Path, default=DEFAULT_CLUSTER)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    try:
        anon = Anonymizer(load_node_names(args.cluster))
    except InventoryUnavailable as e:
        print(f"FAIL: {e} — refusing to export or certify a seed", file=sys.stderr)
        return EXIT_NO_INVENTORY
    if args.check_only:
        target = args.out if args.out.exists() else args.records
        leaks = anon.leak_scan(target.read_text())
        if leaks:
            print(f"FAIL: {target} leaks {leaks}")
            return 1
        print(f"OK: {target} has no hostname/tailnet/private-IP/home/email leaks")
        return 0
    if not args.records.exists():
        print(f"FAIL: records missing at {args.records}")
        return 2
    return export(args.records, args.out, anon)


if __name__ == "__main__":
    raise SystemExit(main())
