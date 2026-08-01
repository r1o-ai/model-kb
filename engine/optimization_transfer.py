#!/usr/bin/env python3
"""optimization_transfer.py — codify when an accel proven on A applies to B.

Example: Qwen native MTP heads measured on base → finetune/quant variant of the
same family may work with *lower* acceptance (status=degraded, mult≈0.55).
DFlash draft heads do NOT transfer across arch_class.

Used by:
  - enrich_optimization_transfer.py (batch onto records)
  - model-kb MCP model_optimization_transfer tool
  - r1o-control model_switch hints (optional)

Rules file: optimization_transfer_rules.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RULES_PATH = ROOT / "optimization_transfer_rules.json"

_RULES: dict[str, Any] | None = None


def load_rules(path: Path = RULES_PATH) -> dict[str, Any]:
    global _RULES
    if _RULES is None:
        _RULES = json.loads(path.read_text())
    assert _RULES is not None
    return _RULES


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _markers_hit(text: str, markers: list[str]) -> bool:
    t = (text or "").lower()
    return any(m in t for m in markers)


def detect_relation(
    proven: dict[str, Any],
    target: dict[str, Any],
    rules: dict[str, Any] | None = None,
) -> str:
    """Return relation_type key between proven recipe and target recipe/artifact."""
    rules = rules or load_rules()
    det = rules.get("relation_detection") or {}
    finetune_m = det.get("finetune_name_markers") or []
    quant_m = det.get("quant_name_markers") or []

    pa = proven.get("artifact") or {}
    ta = target.get("artifact") or {}
    p_hf = str(pa.get("hf_id") or "")
    t_hf = str(ta.get("hf_id") or "")
    p_name = str(pa.get("name") or "")
    t_name = str(ta.get("name") or "")
    p_arch = str(pa.get("architecture") or pa.get("arch_class") or "")
    t_arch = str(ta.get("architecture") or ta.get("arch_class") or "")

    if p_hf and t_hf and p_hf == t_hf:
        return "same_artifact"
    if p_name and t_name and _norm(p_name) == _norm(t_name):
        return "same_artifact"

    # family roots from enrichments
    p_root = (proven.get("hf_family") or {}).get("root_base") or ""
    t_root = (target.get("hf_family") or {}).get("root_base") or ""
    same_family = bool(p_root and t_root and p_root == t_root)
    if not same_family and p_hf and t_hf:
        # weak: shared basename stem
        ps = p_hf.split("/")[-1].split("-")[:3]
        ts = t_hf.split("/")[-1].split("-")[:3]
        if ps and ps == ts:
            same_family = True
        # Qwen/X vs mlx-community/X-4bit
        if _norm(p_hf.split("/")[-1]) in _norm(t_hf) or _norm(t_hf.split("/")[-1]) in _norm(p_hf):
            same_family = True

    t_blob = f"{t_hf} {t_name}"
    p_blob = f"{p_hf} {p_name}"
    target_is_finetune = _markers_hit(t_blob, finetune_m)
    target_is_quant = _markers_hit(t_blob, quant_m) or (
        (target.get("quant") or {}).get("label") not in (None, "", "unknown")
        and (proven.get("quant") or {}).get("label") != (target.get("quant") or {}).get("label")
    )

    if same_family:
        if target_is_finetune and not _markers_hit(p_blob, finetune_m):
            return "same_base_finetune"
        if target_is_quant:
            return "same_base_quant"
        return "same_family_sibling"

    # arch class — only with a shared family token (never bare "MoE"↔"MoE")
    family_tokens = (
        "qwen3.6", "qwen3.5", "qwen36", "qwen35", "qwen3",
        "minimax", "kimi", "gemma-4", "gemma4", "gemma",
        "glm-5", "glm5", "deepseek", "nex-n2", "nex_n2",
    )
    shared_family = any(tok in p_blob.lower() and tok in t_blob.lower() for tok in family_tokens)
    if shared_family and p_arch and t_arch and _norm(p_arch) == _norm(t_arch):
        return "same_arch_class"
    if shared_family and p_arch and t_arch and "moe" in p_arch.lower() and "moe" in t_arch.lower():
        return "same_arch_class"

    return "cross_family"


def proven_accels(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for a in recipe.get("accelerators") or []:
        if not isinstance(a, dict) or not a.get("enabled", True):
            continue
        t = a.get("type")
        if not t:
            continue
        params = a.get("params") or {}
        note = a.get("note") or ""
        # UNVERIFIED in note → lower confidence seed
        conf = "measured"
        if re.search(r"unverified|experimental|expect", note, re.I):
            conf = "inferred"
        if re.search(r"do not|never|broken", note, re.I):
            conf = "pointer"
        out.append(
            {
                "type": t,
                "draft_model": a.get("draft_model"),
                "params": params,
                "note": note,
                "avg_accepted_tokens": params.get("avg_accepted_tokens"),
                "expected_speedup": params.get("expected_speedup"),
                "confidence": conf,
                "source_recipe_id": recipe.get("recipe_id"),
            }
        )
    return out


def transfer_one(
    accel_type: str,
    relation: str,
    proven: dict[str, Any],
    target: dict[str, Any],
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rules = rules or load_rules()
    acc = (rules.get("accelerators") or {}).get(accel_type) or {}
    tr = (acc.get("transfer") or {}).get(relation) or {
        "status": "deny",
        "acceptance_mult": 0.0,
        "confidence": "pointer",
        "caveat": f"No rule for {accel_type} × {relation}",
    }
    base_acc = None
    for a in proven_accels(proven):
        if a["type"] == accel_type:
            base_acc = a.get("avg_accepted_tokens")
            break
    expected_acc = None
    if base_acc is not None and tr.get("acceptance_mult") is not None:
        try:
            expected_acc = round(float(base_acc) * float(tr["acceptance_mult"]), 3)
        except (TypeError, ValueError):
            expected_acc = None

    return {
        "accel": accel_type,
        "relation": relation,
        "status": tr.get("status"),
        "acceptance_mult": tr.get("acceptance_mult"),
        "expected_avg_accepted_tokens": expected_acc,
        "confidence": tr.get("confidence"),
        "caveat": tr.get("caveat"),
        "kind": acc.get("kind"),
        "proven_on": {
            "recipe_id": proven.get("recipe_id"),
            "hf_id": (proven.get("artifact") or {}).get("hf_id"),
            "name": (proven.get("artifact") or {}).get("name"),
        },
        "target": {
            "recipe_id": target.get("recipe_id"),
            "hf_id": (target.get("artifact") or {}).get("hf_id"),
            "name": (target.get("artifact") or {}).get("name"),
        },
        "actionable": tr.get("status") in ("proven", "likely", "degraded", "experimental"),
        "serve_recommendation": _serve_rec(tr.get("status")),
    }


def _serve_rec(status: str | None) -> str:
    return {
        "proven": "use_recipe_as_is",
        "likely": "use_with_smoke_bench",
        "degraded": "use_but_expect_lower_acceptance_rebench",
        "experimental": "try_only_with_explicit_verify",
        "deny": "do_not_transfer",
    }.get(status or "", "unknown")


def transfers_for_target(
    target: dict[str, Any],
    catalog: list[dict[str, Any]],
    rules: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """For a target recipe, find all proven accels on related recipes and score transfer."""
    rules = rules or load_rules()
    results: list[dict[str, Any]] = []
    for proven in catalog:
        if proven.get("kind") and proven.get("kind") != "serve_recipe":
            continue
        if proven.get("recipe_id") == target.get("recipe_id"):
            # self — mark proven accels as same_artifact
            for a in proven_accels(proven):
                results.append(
                    transfer_one(a["type"], "same_artifact", proven, target, rules)
                )
            continue
        rel = detect_relation(proven, target, rules)
        if rel == "cross_family":
            continue
        for a in proven_accels(proven):
            if a["type"] not in (rules.get("accelerators") or {}):
                continue
            results.append(transfer_one(a["type"], rel, proven, target, rules))

    # dedupe by (accel, proven_recipe) keep best status
    rank = {"proven": 0, "likely": 1, "degraded": 2, "experimental": 3, "deny": 4}
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for r in results:
        key = (r["accel"], (r.get("proven_on") or {}).get("recipe_id") or "")
        prev = best.get(key)
        if not prev or rank.get(r.get("status") or "", 9) < rank.get(prev.get("status") or "", 9):
            best[key] = r
    out = list(best.values())
    out.sort(key=lambda x: (rank.get(x.get("status") or "", 9), x.get("accel") or ""))
    return out


def summarize_for_recipe(
    target: dict[str, Any],
    catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = transfers_for_target(target, catalog)
    by_status: dict[str, list] = {}
    for r in rows:
        if r.get("relation") == "same_artifact":
            continue  # skip self noise in summary
        by_status.setdefault(r.get("status") or "?", []).append(r)
    return {
        "recipe_id": target.get("recipe_id"),
        "artifact": target.get("artifact"),
        "transfers": rows,
        "by_status": {k: len(v) for k, v in by_status.items()},
        "actionable_transfers": [
            r
            for r in rows
            if r.get("relation") != "same_artifact"
            and r.get("status") in ("likely", "degraded", "experimental", "proven")
        ],
    }
