"""asmi `POST /serve/load` — the contract model-kb adapts to.

Rule (MA, 2026-10-31): model-kb adapts its tools to asmi, not the other way
around. A serve config's `serve` section uses asmi's request field names, and
every one of them reaches the load unchanged. model-kb never renames an asmi
field, never adds one, never invents a parallel name.

`LOAD_REQUEST_FIELDS` is the field set of asmi's `LoadRequest` struct
(apple-smi `crates/cluster-monitor/src/types.rs`), with each field's JSON type.
It is the whole struct, not a pick: a field asmi adds must be added here, and
`tests/test_asmi_passthrough.py` compares this table with the struct whenever
the apple-smi source is on disk (`ASMI_TYPES_RS`), so drift fails a test.

The port is not a LoadRequest field: asmi takes it as `?port=N`.

Which flag a field becomes is the engine record's business (`GET /serve/engines`
→ `engines[].fields[].request_field`); asmi ignores a field the engine has no
rule for. `extra_args`, `params` and `env` are asmi's generic passthroughs for
anything an engine record does not model.
"""
from __future__ import annotations

from typing import Any

# name → JSON type, as asmi deserializes it (Option<T> = may be absent or null).
LOAD_REQUEST_FIELDS: dict[str, str] = {
    "model_path": "string",
    "backend": "string",
    "hostfile": "string",
    "engine": "string",
    "draft_model": "string",
    "num_draft_tokens": "u32",
    "decode_concurrency": "u32",
    "prompt_concurrency": "u32",
    "prefill_step_size": "u32",
    "pipeline": "bool",
    "prompt_cache_size": "u32",
    "prompt_cache_bytes": "u64",
    "use_mtp": "bool",
    "cache_type": "string",
    "max_tokens": "u32",
    "kv_bits": "f64",
    "kv_quant_scheme": "string",
    "vision_cache_size": "u32",
    "ctx_size": "u64",
    "max_num_seqs": "u32",
    "draft_kind": "string",
    "draft_block_size": "u32",
    "drafter": "string",
    "env": "string_map",
    "dry_run": "bool",
    "extra_args": "string_list",
    "params": "string_map",
}

# LoadRequest fields a recipe's `serve` section does not supply, and why.
NOT_FROM_SERVE: dict[str, str] = {
    "model_path": "the model path comes from the config's model.path",
    "dry_run": "dry_run is a per-request preview switch the caller sets",
}

_UINT_MAX = {"u32": 2**32 - 1, "u64": 2**64 - 1}
_TYPE_WORDS = {
    "string": "a string",
    "u32": "a whole number from 0 to 4294967295",
    "u64": "a whole number of 0 or more",
    "f64": "a number",
    "bool": "true or false",
    "string_list": 'a list of strings (pre-split argv, e.g. ["--flag", "value"])',
    "string_map": "an object of string values",
}


def describe_type(kind: str) -> str:
    return _TYPE_WORDS.get(kind, kind)


def type_ok(kind: str, value: Any) -> bool:
    """True when asmi would deserialize `value` as `kind`. No coercion."""
    if kind == "string":
        return isinstance(value, str)
    if kind in _UINT_MAX:
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _UINT_MAX[kind]
    if kind == "f64":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "string_list":
        return isinstance(value, list) and all(isinstance(t, str) for t in value)
    if kind == "string_map":
        return isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items())
    return False


def type_problem(field: str, value: Any) -> str | None:
    """A named reason why asmi would refuse `value` for `field`, or None."""
    kind = LOAD_REQUEST_FIELDS.get(field)
    if kind is None or type_ok(kind, value):
        return None
    return (
        f"serve.{field} is {value!r} ({type(value).__name__}); asmi LoadRequest.{field} "
        f"takes {describe_type(kind)}"
    )


def serve_load_fields(serve: dict[str, Any]) -> dict[str, Any]:
    """Split a `serve` section into what goes to asmi and what does not.

    Returns:
      fields    — every asmi LoadRequest field the section sets, unchanged
                  (null/absent = unset = not sent).
      problems  — values asmi would refuse, each named; such a field is not in
                  `fields` and the load is blocked.
      warnings  — asmi fields the section sets but a recipe does not own.
      not_sent  — keys that are not asmi LoadRequest fields (config metadata,
                  the port, which asmi takes as ?port=). Listed, never sent.
    """
    fields: dict[str, Any] = {}
    problems: list[str] = []
    warnings: list[str] = []
    not_sent: list[str] = []
    for key, value in serve.items():
        if key not in LOAD_REQUEST_FIELDS:
            not_sent.append(key)
            continue
        if value is None:
            continue
        if key in NOT_FROM_SERVE:
            warnings.append(f"serve.{key} is not sent: {NOT_FROM_SERVE[key]}")
            continue
        why = type_problem(key, value)
        if why:
            problems.append(why)
            continue
        fields[key] = value
    return {"fields": fields, "problems": problems, "warnings": warnings, "not_sent": sorted(not_sent)}


def asmi_body_of(request: dict[str, Any]) -> dict[str, Any]:
    """Only the LoadRequest fields of a model-kb serve request (drops port, recipe_id, accelerators)."""
    return {k: v for k, v in request.items() if k in LOAD_REQUEST_FIELDS}
