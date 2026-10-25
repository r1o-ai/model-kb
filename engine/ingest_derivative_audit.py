#!/usr/bin/env python3
"""ingest_derivative_audit.py — per-derivative facts the family tree does not store.

Input: enrichments_hf_family.jsonl (live ids from the Hub walk).
Output: enrichments_hf_derivative.jsonl, one hf_derivative row per root.

For every member id the family file actually lists this script fetches, from
the Hub, only:

  - repo metadata (gated flag, tags, siblings, library_name, cardData)
  - config.json when that sibling exists (quantization_config)
  - *.safetensors.index.json when that sibling exists (weight_map key counts)
  - README only for non-quantized relations (modification metrics)

It does not download weight shards. It does not invent a serve recipe.

Fail closed:
  - family file missing, unreadable, or measurements.truncated → exit 1, no out file
  - official config/index unreachable (status 0 or 5xx) → exit 1
  - a full result is never "0 keys" or "matches official" when the fetch failed
  - 401/403 is access=gated_unauthenticated (token source name only, never the token)
  - --strict-gated (pipeline --release) exits 1 if a recipe root itself is gated unread

Usage:
  python3 ingest_derivative_audit.py
  python3 ingest_derivative_audit.py --strict-gated
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lib.hf_hub import (  # noqa: E402
    HF_ROOT,
    get_json,
    get_text,
    token_source,
)

QUANT_FIELD = re.compile(r"quant|block|group_size|ignore|modules_to_not|fmt|scheme|bits|dtype", re.I)
FRACTION = re.compile(r"(\d+)\s*/\s*(\d+)")
ENGINE_TS_DEFAULT = Path.home() / "Projects" / "r1o" / "web" / "src" / "lib" / "engine-config.ts"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def content_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def coarse_group(key: str) -> str:
    """Group weight names without a pinned module list.

    Drop numeric layer indexes, then keep the first three remaining segments.
    model.layers.17.mlp and model.layers.18.self_attn stay different groups;
    model.layers.17.mlp and model.layers.18.mlp share one. The segments come
    from the official index's own keys.
    """
    parts = [p for p in key.split(".") if p and not p.isdigit()]
    if not parts:
        return key
    return ".".join(parts[:3])


def count_groups(weight_map: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in weight_map:
        if not isinstance(key, str):
            continue
        group = coarse_group(key)
        counts[group] = counts.get(group, 0) + 1
    return counts


def declared_towers(config: dict[str, Any]) -> list[str]:
    """Config keys that name a nested module. Unmapped tokens are reported, not guessed."""
    towers: list[str] = []

    def walk(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        for key, val in obj.items():
            if not isinstance(key, str):
                continue
            if isinstance(val, dict) and key.endswith("_config"):
                token = key[: -len("_config")]
                if len(token) >= 3 and token not in {"text", "model"}:
                    towers.append(token)
                walk(val)
            elif isinstance(val, int) and key.startswith("num_") and val > 0 and key != "num_hidden_layers":
                towers.append(key)

    walk(config)
    # preserve order, drop dupes
    return list(dict.fromkeys(towers))


def map_towers(towers: list[str], groups: dict[str, int]) -> tuple[dict[str, list[str]], list[str]]:
    mapped: dict[str, list[str]] = {}
    unmapped: list[str] = []
    for tower in towers:
        token = tower.lower()
        hits = [g for g in groups if token in g.lower()]
        if hits:
            mapped[tower] = hits
        else:
            unmapped.append(tower)
    return mapped, unmapped


def quant_view(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {"status": "unreadable"}
    block: dict[str, Any] | None = None
    source_key = None
    raw = config.get("quantization_config")
    if isinstance(raw, dict):
        block, source_key = raw, "quantization_config"
    else:
        for key, val in config.items():
            if isinstance(key, str) and "quant" in key.lower() and isinstance(val, dict):
                block, source_key = val, key
                break
    if block is None:
        return {"status": "no_quantization_config"}
    fields = {k: v for k, v in block.items() if isinstance(k, str) and QUANT_FIELD.search(k)}
    return {
        "status": "ok",
        "source_key": source_key,
        "quant_method": block.get("quant_method"),
        "fields": fields,
    }


def _norm(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(sorted(value, key=lambda x: json.dumps(x, sort_keys=True, default=str)), default=str)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return json.dumps(value, default=str)


def compare_quant(official: dict[str, Any], derivative: dict[str, Any]) -> dict[str, Any]:
    if official.get("status") != "ok":
        return {"parity": "no_official", "diverge_fields": []}
    if derivative.get("status") == "unreadable":
        return {"parity": "unreadable", "diverge_fields": []}
    if derivative.get("status") != "ok":
        return {"parity": "no_quantization_config", "diverge_fields": []}
    off = official.get("fields") or {}
    der = derivative.get("fields") or {}
    keys = sorted(set(off) | set(der) | {"quant_method"})
    diverge = []
    off_method = official.get("quant_method")
    der_method = derivative.get("quant_method")
    if off_method != der_method:
        diverge.append("quant_method")
    for key in keys:
        if key == "quant_method":
            continue
        if _norm(off.get(key)) != _norm(der.get(key)):
            diverge.append(key)
    return {"parity": "diverge" if diverge else "match", "diverge_fields": diverge}


def sibling_names(api: dict[str, Any]) -> list[str]:
    names = []
    for sib in api.get("siblings") or []:
        if isinstance(sib, dict) and sib.get("rfilename"):
            names.append(str(sib["rfilename"]))
    return names


def sibling_bytes(api: dict[str, Any]) -> int | None:
    total = 0
    saw = False
    for sib in api.get("siblings") or []:
        if not isinstance(sib, dict):
            continue
        size = sib.get("size")
        if isinstance(size, int):
            total += size
            saw = True
        elif sib.get("rfilename"):
            # A listed file with no size must not become a fake 0-byte total.
            return None
    return total if saw else None


def formats_of(names: list[str], library_name: str | None) -> list[str]:
    found: set[str] = set()
    for name in names:
        lower = name.lower()
        if lower.endswith(".gguf"):
            found.add("gguf")
        if lower.endswith(".safetensors") or lower.endswith(".safetensors.index.json"):
            found.add("safetensors")
    if library_name and "mlx" in library_name.lower():
        found.add("mlx")
    return sorted(found)


def index_names(names: list[str]) -> list[str]:
    return [n for n in names if n.endswith(".safetensors.index.json")]


# JSON is read whole: a byte cap truncates it into an unparseable document, and index size grows
# with tensor count (the official Qwen3.8-Flash-Next-FP8 index is ~17 MB). Unparseable → malformed.
def fetch_config(repo: str) -> tuple[str, dict[str, Any] | None]:
    kind, _status, body = get_json(f"{HF_ROOT}/{repo}/raw/main/config.json")
    if kind != "ok":
        return kind, None
    return ("ok", body) if isinstance(body, dict) else ("malformed", None)


def fetch_index(repo: str, filename: str) -> tuple[str, dict[str, int] | None]:
    kind, _status, body = get_json(f"{HF_ROOT}/{repo}/resolve/main/{filename}", timeout=120.0)
    if kind != "ok":
        return kind, None
    weight_map = body.get("weight_map") if isinstance(body, dict) else None
    if not isinstance(weight_map, dict):
        return "malformed", None
    return "ok", count_groups(weight_map)


def parse_engine_path_kinds(engine_ts: Path) -> dict[str, str | None]:
    """modelPathKind per EngineId from the live registry. Missing file → {}."""
    if not engine_ts.is_file():
        return {}
    text = engine_ts.read_text()
    # Engine object keys are `name: {` at the start of ENGINES. Path kind is the
    # only format fact the registry stores today.
    blocks = re.split(r"\n  ([a-z][a-z0-9_]*): \{", text)
    # blocks[0] is preamble, then name, body, name, body...
    kinds: dict[str, str | None] = {}
    if len(blocks) < 3:
        return {}
    it = iter(blocks[1:])
    for name, body in zip(it, it):
        match = re.search(r"modelPathKind:\s*'([^']+)'", body)
        kinds[name] = match.group(1) if match else None
    return kinds


def servability(
    fmts: list[str],
    quant_method: str | None,
    path_kinds: dict[str, str | None],
) -> dict[str, Any]:
    """Engines we can prove from the registry. No proof → flags_status undeclared.

    gguf-file engines are the only format binding engine-config.ts stores.
    safetensors does not imply vLLM, SGLang, or MLX — the recipe's serve.engine
    stays the serve SoT until the registry grows an explicit weightFormats field.
    """
    if not path_kinds:
        return {
            "formats": fmts,
            "quant_method": quant_method,
            "gguf_engines": [],
            "flags_status": "registry_missing",
        }
    gguf_engines = sorted(eid for eid, kind in path_kinds.items() if kind == "gguf-file")
    incompatible: list[str] = []
    only_gguf = fmts == ["gguf"]
    has_gguf = "gguf" in fmts
    has_weights = "safetensors" in fmts or "mlx" in fmts
    if only_gguf:
        # Directory engines are not proven compatible; they are simply not gguf-file.
        pass
    if has_weights and not has_gguf:
        incompatible = gguf_engines  # a gguf-file engine cannot load this tree
    return {
        "formats": fmts,
        "quant_method": quant_method,
        "gguf_engines": gguf_engines if has_gguf else [],
        "incompatible_engines": incompatible,
        "flags_status": "undeclared",
        "note": "quant_method is recorded; extra CLI flags wait for engine-config quantArg",
    }


def _cell_text(cell: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", cell, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _tables(readme: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    chunks = re.findall(r"<table[\s\S]*?</table>", readme, flags=re.I)
    if not chunks:
        # GFM tables: a header line, a |---| separator, then rows.
        lines = readme.splitlines()
        i = 0
        while i < len(lines) - 1:
            if "|" in lines[i] and re.search(r"\|\s*-{2,}", lines[i + 1]):
                block = []
                while i < len(lines) and "|" in lines[i]:
                    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                    block.append(cells)
                    i += 1
                if len(block) >= 2:
                    # drop the separator row
                    tables.append([block[0]] + [r for r in block[2:]])
                continue
            i += 1
        return tables
    for table in chunks:
        rows: list[list[str]] = []
        for tr in re.findall(r"<tr[\s\S]*?</tr>", table, flags=re.I):
            cells = [_cell_text(c) for c in re.findall(r"<t[dh][^>]*>([\s\S]*?)</t[dh]>", tr, flags=re.I)]
            if any(cells):
                rows.append(cells)
        if len(rows) >= 2:
            tables.append(rows)
    return tables


def _number(cell: str) -> float | None:
    cell = cell.strip().rstrip("%")
    mult = re.fullmatch(r"(-?\d+(?:\.\d+)?)x", cell, flags=re.I)
    if mult:
        return float(mult.group(1))
    if re.fullmatch(r"-?\d+(?:\.\d+)?", cell):
        return float(cell)
    return None


def modification_report(card: dict[str, Any] | None, readme: str, repo: str) -> dict[str, Any]:
    """Structured comparisons only. Prose without a table is not 'measured'."""
    metrics: list[dict[str, Any]] = []
    if isinstance(card, dict):
        index = card.get("model-index") or card.get("model_index") or []
        if isinstance(index, list):
            for entry in index:
                if not isinstance(entry, dict):
                    continue
                for result in entry.get("results") or []:
                    if not isinstance(result, dict):
                        continue
                    dataset = result.get("dataset")
                    dataset_name = dataset.get("name") if isinstance(dataset, dict) else dataset
                    for metric in result.get("metrics") or []:
                        if not isinstance(metric, dict):
                            continue
                        if not isinstance(metric.get("value"), (int, float)):
                            continue
                        metrics.append(
                            {
                                "name": str(metric.get("name") or metric.get("type") or ""),
                                "value": metric.get("value"),
                                "n": None,
                                "dataset": dataset_name,
                                "source": "model-index",
                            }
                        )
    rigor = "model-index" if metrics else None
    for rows in _tables(readme):
        header = rows[0]
        for row in rows[1:]:
            if not row:
                continue
            name = row[0]
            if not name:
                continue
            for idx, cell in enumerate(row[1:], start=1):
                frac = FRACTION.search(cell)
                value = _number(cell)
                if frac:
                    metrics.append(
                        {
                            "name": name,
                            "value": int(frac.group(1)),
                            "n": int(frac.group(2)),
                            "column": header[idx] if idx < len(header) else None,
                            "source": "readme_table",
                        }
                    )
                    rigor = rigor or "table_with_n"
                elif value is not None:
                    metrics.append(
                        {
                            "name": name,
                            "value": value,
                            "n": None,
                            "column": header[idx] if idx < len(header) else None,
                            "source": "readme_table",
                        }
                    )
                    rigor = rigor or "table_no_n"
    kl = next((m for m in metrics if re.search(r"\bkl\b|kullback|divergence", m["name"], re.I)), None)
    costs = [
        m
        for m in metrics
        if re.search(r"token|latency|throughput", m["name"], re.I)
    ]
    if not metrics:
        return {"status": "unreported", "rigor": "none", "metrics": [], "repo": repo}
    return {
        "status": "ok",
        "rigor": rigor or "table_no_n",
        "metrics": metrics[:80],
        "kl_divergence": kl,
        "serving_cost": costs[:12],
        "repo": repo,
    }


def repo_api(repo: str) -> tuple[str, dict[str, Any] | None]:
    kind, _status, body = get_json(f"https://huggingface.co/api/models/{repo}")
    if kind != "ok" or not isinstance(body, dict):
        return kind, None
    return "ok", body


# Bump when census or comparison logic changes, so cached rows computed by old rules are redone.
AUDIT_VERSION = 1

# Statuses that describe the repo's content at a commit. Anything else (gated, unreachable,
# http_error) depends on access or the network and is re-checked every run.
_SETTLED = {"ok", "missing", "malformed", "layout_differs", "no_safetensors_index", "no_official_index"}


def _reusable(prev: dict[str, Any]) -> bool:
    if prev.get("audit_version") != AUDIT_VERSION:
        return False
    if prev.get("access") != "ok" or prev.get("config_status", "ok") not in {"ok", "missing", "malformed"}:
        return False
    if (prev.get("components") or {}).get("status") not in _SETTLED:
        return False
    return (prev.get("modification") or {}).get("status") not in {"gated_unauthenticated", "unreachable", "http_error"}


def audit_member(
    member: dict[str, Any],
    *,
    root: str,
    official_groups: dict[str, int] | None,
    official_quant: dict[str, Any],
    official_towers: dict[str, list[str]],
    path_kinds: dict[str, str | None],
    sleep_s: float,
    official_sha: str | None = None,
    cached: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    repo = str(member.get("id") or "")
    relation = member.get("relation") or "unknown"
    row: dict[str, Any] = {
        "id": repo,
        "relation": relation,
        "downloads": member.get("downloads") or 0,
        "access": "ok",
    }
    kind, api = repo_api(repo)
    time.sleep(sleep_s)
    if kind == "gated":
        row["access"] = "gated_unauthenticated"
        row["token_source"] = token_source()
        return row
    if kind != "ok" or api is None:
        row["access"] = kind
        return row
    gated_flag = api.get("gated")
    if gated_flag not in (False, None, "false", "auto"):
        # "auto" still needs acceptance; metadata may 200 for logged-in nodes.
        # A true/manual gate that nevertheless returned 200 was readable.
        row["gated_flag"] = gated_flag
    # Same commit, audited against the same official commit: the census cannot have changed.
    # Downloads and servability are not content, so they are refreshed.
    sha = api.get("sha")
    prev = (cached or {}).get(repo)
    if prev and sha and prev.get("sha") == sha and prev.get("official_sha") == official_sha and _reusable(prev):
        row = {**prev, "relation": relation, "downloads": member.get("downloads") or 0, "cache": "sha_match"}
        row["servability"] = servability(row.get("formats") or [], (row.get("quant") or {}).get("quant_method"), path_kinds)
        return row
    row["sha"] = sha
    row["official_sha"] = official_sha
    row["audit_version"] = AUDIT_VERSION
    names = sibling_names(api)
    row["formats"] = formats_of(names, api.get("library_name"))
    row["total_bytes"] = sibling_bytes(api)
    row["library_name"] = api.get("library_name")

    cfg_kind, config = fetch_config(repo)
    time.sleep(sleep_s)
    if cfg_kind == "gated":
        row["config_status"] = "gated_unauthenticated"
        row["token_source"] = token_source()
    elif cfg_kind != "ok":
        row["config_status"] = cfg_kind
    else:
        row["config_status"] = "ok"
        qv = quant_view(config)
        row["quant"] = qv
        row["quant_parity"] = compare_quant(official_quant, qv)
        towers = declared_towers(config or {})
        _mapped, unmapped = map_towers(towers, official_groups or {})
        if unmapped:
            row["unmapped_towers"] = unmapped

    indexes = index_names(names)
    if not indexes:
        row["components"] = {"status": "no_safetensors_index"}
    else:
        # One index is the census. Multiple indexes are each counted; the
        # first successful one is the comparison against official.
        idx_kind, groups = fetch_index(repo, indexes[0])
        time.sleep(sleep_s)
        if idx_kind != "ok" or groups is None:
            row["components"] = {"status": idx_kind, "index": indexes[0]}
        elif not official_groups:
            row["components"] = {
                "status": "no_official_index",
                "index": indexes[0],
                "group_count": len(groups),
                "key_count": sum(groups.values()),
            }
        elif groups.get(trunk := max(official_groups, key=official_groups.get), 0) == 0:
            # No derivative lacks its own transformer trunk. Missing the official's largest group
            # means the tensors are renamed (MLX, EXL3, some compressed-tensors exports), so
            # absences say nothing about dropped components.
            row["components"] = {
                "status": "layout_differs",
                "index": indexes[0],
                "key_count": sum(groups.values()),
                "official_key_count": sum(official_groups.values()),
                "missing_trunk": trunk,
            }
        else:
            dropped = sorted(g for g, n in official_groups.items() if n > 0 and groups.get(g, 0) == 0)
            shrunk = sorted(
                g for g, n in official_groups.items() if n > 0 and 0 < groups.get(g, 0) < n
            )
            row["components"] = {
                "status": "ok",
                "index": indexes[0],
                "key_count": sum(groups.values()),
                "official_key_count": sum(official_groups.values()),
                "dropped_groups": dropped,
                "shrunk_groups": shrunk[:40],
            }
            if official_towers:
                tower_drops = {
                    tower: [g for g in groups_for if g in dropped]
                    for tower, groups_for in official_towers.items()
                }
                tower_drops = {k: v for k, v in tower_drops.items() if v}
                if tower_drops:
                    row["components"]["tower_drops"] = tower_drops

    row["servability"] = servability(
        row.get("formats") or [],
        (row.get("quant") or {}).get("quant_method"),
        path_kinds,
    )

    if relation == "quantized":
        row["modification"] = {"status": "skipped_quantized_relation"}
    elif row["access"] != "ok":
        row["modification"] = {"status": row["access"]}
    else:
        readme_kind, _st, readme = get_text(f"{HF_ROOT}/{repo}/raw/main/README.md", max_bytes=400_000)
        time.sleep(sleep_s)
        card = api.get("cardData") if isinstance(api.get("cardData"), dict) else {}
        if readme_kind == "gated":
            row["modification"] = {"status": "gated_unauthenticated"}
        elif readme_kind not in ("ok", "missing"):
            row["modification"] = {"status": readme_kind}
        else:
            row["modification"] = modification_report(card, readme if readme_kind == "ok" else "", repo)
    return row


def official_census(
    root: str, sleep_s: float
) -> tuple[str, dict[str, int] | None, dict[str, Any], dict[str, list[str]], str | None]:
    """Status, weight groups, quant view, tower map, and the root's commit sha (the cache key)."""
    kind, api = repo_api(root)
    if kind != "ok" or api is None:
        return kind, None, {"status": kind}, {}, None
    sha = api.get("sha")
    cfg_kind, config = fetch_config(root)
    time.sleep(sleep_s)
    quant = quant_view(config) if cfg_kind == "ok" else {"status": cfg_kind}
    towers = declared_towers(config or {}) if isinstance(config, dict) else []
    indexes = index_names(sibling_names(api))
    if not indexes:
        return "ok", None, quant, {}, sha
    idx_kind, groups = fetch_index(root, indexes[0])
    time.sleep(sleep_s)
    if idx_kind != "ok" or groups is None:
        return idx_kind, None, quant, {}, sha
    mapped, _unmapped = map_towers(towers, groups)
    return "ok", groups, quant, mapped, sha


def build_row(
    family: dict[str, Any],
    audited: list[dict[str, Any]],
    *,
    official_status: str,
) -> dict[str, Any]:
    meas = family.get("measurements") or {}
    root = str(meas.get("root_base") or "")
    relations: dict[str, int] = {}
    dropped_ids: list[str] = []
    gated_ids: list[str] = []
    for row in audited:
        rel = str(row.get("relation") or "unknown")
        relations[rel] = relations.get(rel, 0) + 1
        comps = row.get("components") or {}
        if comps.get("dropped_groups"):
            dropped_ids.append(row["id"])
        if row.get("access") == "gated_unauthenticated" or row.get("config_status") == "gated_unauthenticated":
            gated_ids.append(row["id"])
    eid = f"hfderiv:{root.replace('/', '__')}"
    body = {
        "id": eid,
        "kind": "hf_derivative",
        "enrichment_id": eid,
        "title": f"HF derivative audit · {root}",
        "applies_to": {
            "model_globs": [root, root.split("/")[-1], f"*{root.split('/')[-1]}*"],
            "hf_ids": [root],
            "base_models": [root],
            "accel_types": [],
            "engines": [],
        },
        "claims": [
            f"{len(audited)} derivatives audited under {root}",
            "relations: " + ", ".join(f"{k}={v}" for k, v in sorted(relations.items())),
        ],
        "measurements": {
            "root_base": root,
            "official_status": official_status,
            "discovered": meas.get("member_count"),
            "truncated": bool(meas.get("truncated")),
            "audited": len(audited),
            "relation_counts": relations,
            "component_drop_ids": dropped_ids,
            "gated_unauthenticated": gated_ids,
            "token_source": token_source(),
            "derivatives": audited,
        },
        "confidence": "measured" if official_status == "ok" else "pointer",
        "search_text": " ".join(
            [
                "hf-derivative",
                root.lower(),
                " ".join(sorted(relations)),
                " ".join(dropped_ids[:20]),
            ]
        ).lower(),
        "search_narrative": (
            f"Derivative audit for {root}: {len(audited)} members, "
            f"{len(dropped_ids)} with a weight-group the official index has and the derivative lacks, "
            f"{len(gated_ids)} gated unread."
        ),
        "source": {"type": "huggingface", "root_base": root, "url": f"https://huggingface.co/{root}"},
        "sources_used": ["huggingface"],
        "created": _now(),
    }
    body["content_hash"] = content_hash(
        {"root": root, "audited": len(audited), "drops": dropped_ids, "gated": gated_ids}
    )
    return body


def flatten_members(family: dict[str, Any]) -> list[dict[str, Any]]:
    levels = (family.get("measurements") or {}).get("family_levels") or {}
    out: list[dict[str, Any]] = []
    if isinstance(levels, dict):
        for rows in levels.values():
            if isinstance(rows, list):
                out.extend(r for r in rows if isinstance(r, dict) and r.get("id"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="ingest_derivative_audit")
    ap.add_argument("--family", type=Path, default=ROOT / "enrichments_hf_family.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "enrichments_hf_derivative.jsonl")
    ap.add_argument("--engine-ts", type=Path, default=ENGINE_TS_DEFAULT)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--limit-roots", type=int, default=0)
    ap.add_argument("--limit-members", type=int, default=0, help="Test cap. Exceeding it exits 1 and writes nothing.")
    ap.add_argument("--strict-gated", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="Re-fetch every derivative even when its commit is unchanged")
    args = ap.parse_args()

    # The previous output is the cache: a member is re-fetched only when its commit or the
    # official root's commit moved. An unreadable previous file means a full audit, not a failure.
    cache_by_root: dict[str, dict[str, dict[str, Any]]] = {}
    if not args.no_cache and args.out.is_file():
        try:
            for prev_row in load_jsonl(args.out):
                pm = prev_row.get("measurements") or {}
                cache_by_root[str(pm.get("root_base") or "")] = {
                    d["id"]: d for d in pm.get("derivatives") or [] if isinstance(d, dict) and d.get("id")
                }
        except (OSError, json.JSONDecodeError):
            cache_by_root = {}

    if not args.family.is_file():
        print(f"FAIL: family enrichments missing at {args.family}", file=sys.stderr)
        return 1
    try:
        families = [r for r in load_jsonl(args.family) if r.get("kind") == "hf_family"]
    except (OSError, json.JSONDecodeError) as e:
        print(f"FAIL: cannot read {args.family}: {type(e).__name__}", file=sys.stderr)
        return 1
    if not families:
        print("FAIL: no hf_family rows — refusing to write an empty derivative file", file=sys.stderr)
        return 1
    if args.limit_roots:
        families = families[: args.limit_roots]

    path_kinds = parse_engine_path_kinds(args.engine_ts)
    if not path_kinds:
        print(f"FAIL: engine registry unreadable at {args.engine_ts}", file=sys.stderr)
        return 1

    exit_code = 0
    rows: list[dict[str, Any]] = []
    for family in families:
        meas = family.get("measurements") or {}
        root = str(meas.get("root_base") or "")
        if not root or "/" not in root:
            print(f"FAIL: hf_family row without root_base ({family.get('id')})", file=sys.stderr)
            return 1
        if meas.get("truncated"):
            print(f"FAIL: family walk truncated for {root} — not auditing a partial tree", file=sys.stderr)
            return 1
        members = flatten_members(family)
        if args.limit_members and len(members) > args.limit_members:
            print(f"FAIL: --limit-members truncated {root}", file=sys.stderr)
            return 1

        off_status, off_groups, off_quant, off_towers, off_sha = official_census(root, args.sleep)
        if off_status in {"unreachable", "http_error"} or off_quant.get("status") in {"unreachable", "http_error"}:
            print(f"FAIL: official metadata unreachable for {root} ({off_status} config={off_quant.get('status')})", file=sys.stderr)
            return 1
        if off_status == "gated":
            print(f"gated unread: {root} token_source={token_source()}")
            if args.strict_gated:
                exit_code = 1
            rows.append(
                build_row(
                    family,
                    [
                        {
                            "id": root,
                            "relation": "base",
                            "access": "gated_unauthenticated",
                            "token_source": token_source(),
                        }
                    ],
                    official_status="gated",
                )
            )
            continue

        audited: list[dict[str, Any]] = []
        for member in members:
            if member.get("id") == root:
                continue
            audited.append(
                audit_member(
                    member,
                    root=root,
                    official_groups=off_groups,
                    official_quant=off_quant,
                    official_towers=off_towers,
                    path_kinds=path_kinds,
                    sleep_s=args.sleep,
                    official_sha=off_sha,
                    cached=cache_by_root.get(root),
                )
            )
        # A derivative whose fetch died (unreachable) must not look like a clean census.
        transient = {"unreachable", "http_error"}
        bad = [
            f"{r['id']}={r.get('access') if r.get('access') in transient else (r.get('components') or {}).get('status')}"
            for r in audited
            if r.get("access") in transient or (r.get("components") or {}).get("status") in transient
        ]
        if bad:
            print(f"FAIL: derivative metadata fetch failed under {root}: {bad[:8]}", file=sys.stderr)
            return 1
        # --strict-gated applies to the official root above, the source every comparison needs.
        # A gated third-party derivative is not skipped silently: it is listed by id in
        # gated_unauthenticated, so failing a release build over one would punish normal Hub state.
        official_status = off_status if off_groups is not None or off_status != "ok" else "no_index"
        rows.append(build_row(family, audited, official_status=official_status))
        hits = sum(1 for r in audited if r.get("cache") == "sha_match")
        print(f"  + {root}: audited={len(audited)} unchanged={hits} official={off_status}")

    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(args.out)
    print(f"wrote {len(rows)} → {args.out}")
    if exit_code != 0:
        print("FAIL: --strict-gated and at least one repo was gated unread", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
