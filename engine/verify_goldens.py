#!/usr/bin/env python3
"""verify_goldens.py — run golden-queries.yaml end-to-end.

Persistent gate for model-kb ARDD Phase B. GOLDEN_PASS means every non-optional
query in golden-queries.yaml succeeded — including unit checks like
q-quant-decimal-bit (3.6bit ≠ 6bit).

Usage:
  python3 verify_goldens.py
  python3 verify_goldens.py --yaml golden-queries.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from build_bm25 import load_index, load_records, search  # noqa: E402
from ingest_serve_configs import normalize_quant, recipe_to_serve_request  # noqa: E402


def _load_yaml(path: Path) -> dict:
    """Minimal YAML loader for our flat golden schema (no PyYAML required)."""
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text())
    except ImportError:
        pass

    # Fallback: only supports the structures we write.
    text = path.read_text()
    queries: list[dict] = []
    cur: dict | None = None
    list_key: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("version:"):
            continue
        if line.startswith("queries:"):
            continue
        if re.match(r"^\s+-\s+id:", line):
            if cur:
                queries.append(cur)
            cur = {"id": line.split("id:", 1)[1].strip().strip("\"'")}
            list_key = None
            continue
        if cur is None:
            continue
        m = re.match(r"^\s+(\w+):\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            items = []
            if inner:
                for part in re.findall(r"\"([^\"]*)\"|'([^']*)'|([^,\s]+)", inner):
                    items.append(next(x for x in part if x))
            cur[key] = items
            list_key = None
        elif val == "" or val == "|":
            list_key = key
            cur[key] = []
        else:
            # strip quotes / trailing comments
            val = re.sub(r"\s+#.*$", "", val).strip().strip("\"'")
            if val in ("true", "false"):
                cur[key] = val == "true"
            elif re.fullmatch(r"-?\d+", val):
                cur[key] = int(val)
            else:
                cur[key] = val
            list_key = None
    if cur:
        queries.append(cur)
    return {"version": 1, "queries": queries}


def run_bm25_query(q: dict, idx, records: list[dict]) -> tuple[bool, str]:
    hits = search(idx, q["q"], top_k=int(q.get("top_k") or 5))
    notes: list[str] = []

    for pat in q.get("require_any_name_regex") or []:
        if not any(
            re.search(pat, (h.get("name") or "") + " " + (h.get("recipe_id") or ""), re.I)
            for h in hits
        ):
            return False, f"no hit matching name regex {pat!r}"
        notes.append(f"name~{pat}")

    for accel in q.get("require_accel") or []:
        if not any(accel in (h.get("accels") or []) for h in hits):
            return False, f"no hit with accel={accel}"
        notes.append(f"accel={accel}")

    for quant in q.get("require_quant_in") or []:
        ok = any((h.get("quant") == quant) for h in hits)
        if not ok:
            # also accept search_text containing quant token
            ok = any(
                quant in ((h.get("search_text") or "") + " " + (h.get("quant") or ""))
                for h in hits
            )
        if not ok:
            return False, f"no hit with quant in {q.get('require_quant_in')}"
        notes.append(f"quant={quant}")

    for backend in q.get("require_backend_in") or []:
        if not any((h.get("backend") == backend) for h in hits):
            return False, f"no hit with backend={backend}"
        notes.append(f"backend={backend}")

    kinds = q.get("require_kind_in")
    if kinds:
        if not any((h.get("kind") in kinds) for h in hits):
            return False, f"no hit with kind in {kinds}"

    for role in q.get("forbid_roles") or []:
        for h in hits:
            rec = next((r for r in records if r.get("id") == h.get("id")), None)
            if rec and role in (rec.get("roles_fit") or []):
                return False, f"hit {h.get('recipe_id')} has forbidden role {role}"

    top = hits[0]["recipe_id"] if hits else "(none)"
    return True, f"top={top}; " + ",".join(notes)


def run_unit(q: dict, records: list[dict]) -> tuple[bool, str]:
    qid = q.get("id") or ""
    assertion = (q.get("assert") or "").strip()

    # Named units we implement explicitly (stable, not eval).
    if qid == "q-recipe-count" or "len(records" in assertion:
        n = sum(1 for r in records if r.get("kind") == "serve_recipe")
        ok = n >= 20
        return ok, f"serve_recipe_count={n}"

    if qid == "q-serve-request-no-auto" or "recipe_to_serve_request" in assertion:
        for r in records:
            host = (
                "~/.r1o/hostfiles/default.json"
                if r.get("serve", {}).get("backend") == "jaccl"
                else None
            )
            body = recipe_to_serve_request(r, hostfile=host)
            if body.get("backend") == "auto":
                return False, f"{r.get('recipe_id')} backend=auto"
            if "model_path" not in body:
                return False, f"{r.get('recipe_id')} missing model_path"
            if body.get("backend") not in ("single", "jaccl"):
                return False, f"{r.get('recipe_id')} backend={body.get('backend')}"
        return True, f"materialized={len(records)} all model_path+backend ok"

    if qid in ("q-quant-decimal-bit", "q-decimal-quant") or "normalize_quant" in assertion:
        a = normalize_quant("3.6bit")["label"]
        b = normalize_quant("6bit")["label"]
        c = normalize_quant("4bit")["label"]
        ok = a == "3.6bit" and b == "6bit" and c == "4bit" and a != b
        # also scan live records for regression
        bad = []
        for r in records:
            raw = (r.get("quant") or {}).get("raw") or ""
            lab = (r.get("quant") or {}).get("label")
            if "3.6" in raw and lab == "6bit":
                bad.append(r.get("recipe_id"))
        if bad:
            return False, f"records still mislabeled 6bit: {bad}"
        return ok, f"3.6bit→{a}, 6bit→{b}, 4bit→{c}"

    return False, f"unimplemented unit assert for {qid}: {assertion!r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=Path, default=ROOT / "golden-queries.yaml")
    ap.add_argument("--index", type=Path, default=ROOT / "bm25_index" / "index.pkl")
    ap.add_argument("--records", type=Path, default=ROOT / "records.jsonl")
    args = ap.parse_args()

    if not args.yaml.exists():
        print(f"FAIL missing {args.yaml}")
        return 1
    if not args.records.exists():
        print(f"FAIL missing {args.records}; run ingest_serve_configs.py")
        return 1
    if not args.index.exists():
        print(f"FAIL missing {args.index}; run build_bm25.py --rebuild")
        return 1

    spec = _load_yaml(args.yaml)
    queries = spec.get("queries") or []
    records = load_records(args.records)
    idx = load_index(args.index)

    results: dict[str, bool] = {}
    for q in queries:
        qid = q.get("id") or "?"
        optional = bool(q.get("optional"))
        try:
            if q.get("type") == "unit":
                ok, detail = run_unit(q, records)
            else:
                ok, detail = run_bm25_query(q, idx, records)
        except Exception as e:  # noqa: BLE001 — surface as fail
            ok, detail = False, f"exception: {e}"

        status = "PASS" if ok else ("SKIP" if optional and not ok else "FAIL")
        if optional and not ok:
            results[qid] = True  # optional failure does not gate
            print(f"  {status} {qid}: {detail}")
        else:
            results[qid] = ok
            print(f"  {status} {qid}: {detail}")

    failed = [k for k, v in results.items() if not v]
    print("RESULTS", results)
    if failed:
        print("GOLDEN_FAIL", failed)
        return 1
    # Hard require decimal quant id present in suite
    quant_ids = {"q-quant-decimal-bit", "q-decimal-quant"}
    if not quant_ids.intersection(results):
        print("GOLDEN_FAIL: suite missing decimal-quant unit (q-quant-decimal-bit)")
        return 1
    print("GOLDEN_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
