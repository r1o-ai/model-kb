"""Live engine ids: the union of asmi /serve/engines over every online node.

Engine sets differ per node, so no single node's asmi is the truth. Nodes come
from asmi's own cluster listing (ASMI_URL); each node is reached on the same
asmi port through its tailnet address (LAN names from /etc/hosts are not
reachable from cron), then the resolver. No hostnames or engine ids live here.

Fails closed: if no node answers, EngineCatalogUnavailable — callers keep the
previous state instead of marking every recipe non-servable.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from .embed_discovery import _resolver_addresses, _url, live_nodes, tailnet_addresses

TIMEOUT_S = float(os.getenv("MODEL_KB_ASMI_TIMEOUT_S", "3"))


class EngineCatalogUnavailable(RuntimeError):
    """No online node's asmi answered /serve/engines."""


def _asmi_url() -> str:
    return os.getenv("ASMI_URL", "http://localhost:9090").rstrip("/")


def _engine_ids(base: str) -> list[str] | None:
    """Engine ids one asmi declares, or None when it does not answer."""
    try:
        with urllib.request.urlopen(f"{base}/serve/engines", timeout=TIMEOUT_S) as r:
            body = json.loads(r.read() or b"null")
    except (OSError, ValueError, urllib.error.URLError):
        return None
    rows = body.get("engines") if isinstance(body, dict) else body
    if not isinstance(rows, list):
        return None
    return sorted({str(e["id"]) for e in rows if isinstance(e, dict) and e.get("id")})


def live_engine_ids(trace: list[str] | None = None) -> dict[str, list[str]]:
    """{engine_id: [online nodes declaring it]} — union over the cluster, sorted."""
    trace = trace if trace is not None else []
    parts = urlsplit(_asmi_url())
    port = parts.port or (443 if parts.scheme == "https" else 80)
    nodes = live_nodes()
    if not nodes:
        raise EngineCatalogUnavailable(f"asmi at {_asmi_url()} listed no online nodes")
    tailnet = tailnet_addresses()

    union: dict[str, set[str]] = {}
    answered = 0
    for name in nodes:
        ids = None
        for addr in tailnet.get(name.lower(), []):
            ids = _engine_ids(_url(addr, port))
            trace.append(f"{name} {addr}: {'ok' if ids is not None else 'no answer'}")
            if ids is not None:
                break
        if ids is None:
            for addr in _resolver_addresses(name, port):
                ids = _engine_ids(_url(addr, port))
                trace.append(f"{name} {addr} (resolver): {'ok' if ids is not None else 'no answer'}")
                if ids is not None:
                    break
        if ids is None:
            continue
        answered += 1
        for eid in ids:
            union.setdefault(eid, set()).add(name)

    if not answered or not union:
        raise EngineCatalogUnavailable(
            f"no /serve/engines answer from {len(nodes)} listed node(s): " + "; ".join(trace[-12:])
        )
    return {eid: sorted(ns) for eid, ns in sorted(union.items())}
