#!/usr/bin/env python3
"""Live Hugging Face Hub client for model-kb ingest.

Not a model catalog. Callers pass repo ids and filter expressions they
already resolved from the corpus or from a previous Hub response.

Token resolution (value is never logged, returned to stdout, or written
into a record):
  1. env HF_TOKEN, else HUGGING_FACE_HUB_TOKEN
  2. $HF_HOME/token if that file exists
  3. ~/.cache/huggingface/token if that file exists
Missing token is not an error until a call returns 401/403.

Fail closed: HTTP failures are a status, never a synthetic empty list
that callers can mistake for "no derivatives".
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote

HF_API = "https://huggingface.co/api/models"
HF_ROOT = "https://huggingface.co"

# Hub list-filter relation slots on base_model. This is the API vocabulary,
# not a list of models. Tags seen at runtime with any other slot are queried
# too (see relations_to_query).
SEED_RELATIONS = ("finetune", "quantized", "merge")

_NEXT_LINK = re.compile(r'<([^>]+)>;\s*rel="?next"?', re.I)


def token_source() -> str | None:
    """Where a token would come from. Never the token itself."""
    if os.environ.get("HF_TOKEN"):
        return "env:HF_TOKEN"
    if os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return "env:HUGGING_FACE_HUB_TOKEN"
    home = os.environ.get("HF_HOME")
    if home and (Path(home) / "token").is_file():
        return "file:HF_HOME/token"
    if (Path.home() / ".cache" / "huggingface" / "token").is_file():
        return "file:~/.cache/huggingface/token"
    return None


def _read_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val:
            return val.strip()
    candidates: list[Path] = []
    home = os.environ.get("HF_HOME")
    if home:
        candidates.append(Path(home) / "token")
    candidates.append(Path.home() / ".cache" / "huggingface" / "token")
    for path in candidates:
        try:
            if path.is_file():
                text = path.read_text().strip()
                if text:
                    return text
        except OSError:
            continue
    return None


def _request(url: str, timeout: float) -> tuple[int, bytes, dict[str, str], str | None]:
    headers = {"User-Agent": "r1o-model-kb/1.0 (hf-hub)"}
    token = _read_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            hdrs = {k: v for k, v in resp.headers.items()}
            return resp.status, raw, hdrs, None
    except urllib.error.HTTPError as e:
        raw = e.read() if e.fp else b""
        hdrs = {k: v for k, v in (e.headers.items() if e.headers else [])}
        return e.code, raw, hdrs, None
    except Exception as e:  # noqa: BLE001 — network failure must surface as status 0
        return 0, b"", {}, type(e).__name__


def _decode_json(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None


def next_url(headers: dict[str, str]) -> str | None:
    link = headers.get("Link") or headers.get("link") or ""
    match = _NEXT_LINK.search(link)
    return match.group(1) if match else None


def classify_http(status: int) -> str:
    if status == 200:
        return "ok"
    if status in (401, 403):
        return "gated"
    if status == 0 or status >= 500:
        return "unreachable"
    if status == 404:
        return "missing"
    return "http_error"


def get_json(url: str, timeout: float = 40.0) -> tuple[str, int, Any]:
    """Return (class, http_status, body). class never pretends failure is []."""
    status, raw, _headers, _err = _request(url, timeout)
    kind = classify_http(status)
    if kind != "ok":
        return kind, status, None
    return kind, status, _decode_json(raw)


def get_text(url: str, timeout: float = 40.0, max_bytes: int = 2_000_000) -> tuple[str, int, str]:
    status, raw, _headers, _err = _request(url, timeout)
    kind = classify_http(status)
    if kind != "ok":
        return kind, status, ""
    return kind, status, raw[:max_bytes].decode("utf-8", "replace")


def list_filter(
    filter_expr: str,
    *,
    page_limit: int = 100,
    hard_limit: int = 0,
    sleep_s: float = 0.0,
    timeout: float = 40.0,
) -> tuple[str, int, list[dict[str, Any]], bool]:
    """Paginate GET /api/models?filter=…

    Returns (class, http_status, models, truncated).

    truncated is True when a full page came back with no Link rel=next,
    or when hard_limit stopped the walk. A full page without a next link
    is NOT a complete result — callers must fail closed instead of storing
    it as the whole tree.

    On gated / unreachable / http_error, models is [] and class says why.
    """
    url = (
        f"{HF_API}?filter={quote(filter_expr, safe='')}"
        f"&limit={page_limit}&sort=downloads&direction=-1&full=true"
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    last_status = 0
    while url:
        status, raw, headers, _err = _request(url, timeout)
        last_status = status
        kind = classify_http(status)
        if kind != "ok":
            return kind, status, [], False
        data = _decode_json(raw)
        if not isinstance(data, list):
            return "http_error", status, [], False
        for row in data:
            if not isinstance(row, dict):
                continue
            mid = row.get("id")
            if not mid or mid in seen:
                continue
            seen.add(str(mid))
            out.append(row)
        nxt = next_url(headers)
        if hard_limit and len(out) >= hard_limit:
            return "ok", status, out[:hard_limit], True
        if not nxt:
            truncated = len(data) >= page_limit
            return "ok", status, out, truncated
        url = nxt
        if sleep_s:
            time.sleep(sleep_s)
    return "ok", last_status, out, False


def relations_to_query(extra_slots: set[str] | None = None) -> list[str]:
    slots = list(SEED_RELATIONS)
    for slot in sorted(extra_slots or []):
        if slot and slot not in slots and slot != "untyped":
            slots.append(slot)
    return slots


def relation_slots_from_tags(tags: list[Any]) -> set[str]:
    found: set[str] = set()
    for tag in tags:
        if not isinstance(tag, str) or not tag.startswith("base_model:"):
            continue
        rest = tag[len("base_model:") :]
        if ":" in rest:
            slot = rest.split(":", 1)[0].strip()
            if slot:
                found.add(slot)
    return found


def relation_for(tags: list[Any], root: str) -> str:
    """Relation slot this child declared toward root. 'unknown' if none.

    Does not guess from the repo name.
    """
    untyped = False
    for tag in tags:
        if not isinstance(tag, str) or not tag.startswith("base_model:"):
            continue
        rest = tag[len("base_model:") :]
        if rest == root:
            untyped = True
            continue
        if ":" in rest:
            slot, target = rest.split(":", 1)
            if target == root and slot:
                return slot
    return "untyped" if untyped else "unknown"
