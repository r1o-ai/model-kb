"""Find a live Qwen3-VL embedding server, or fail closed.

The engine needs one embedding server for `embed_model_kb.py` (corpus vectors)
and `hybrid_search.py` (query vectors). No host lives in this file. Resolution:

1. Scalar override: QWEN_EMBED_SERVER or EMBED_URL env, else `server_url` in the
   user-owned config (~/.config/qwen-embed/config.toml, `[embed]`). The override
   is still probed: a URL that does not prove an embedding server is an error,
   never a silent fall-through to some other host.
2. Live discovery: the online nodes from asmi (`ASMI_URL`, default the local
   daemon). Each node is tried at every address that reaches it — its tailnet
   addresses from `tailscale status --json`, then whatever the system resolver
   returns for its name — plus any `hosts` the user lists in the config.
3. Nothing proves itself: raise EmbedServerUnavailable. There is no
   `localhost:<port>` guess; a guess 404s and lets a caller write stale vectors.

Why tailnet addresses: a job started by cron or launchd outside the GUI login is
subject to macOS Local Network privacy. Its connects to LAN addresses fail with
EHOSTUNREACH ("No route to host"), while the same connect from an ssh session
works. Node names on a node may resolve to LAN addresses (/etc/hosts pins), so
discovery by resolver alone found nothing from cron. Tailnet addresses route
over the Tailscale interface and are reachable in both contexts.

A candidate counts only when GET /health carries an `embedding` section with
dimensions > 0 (a chat server or another app on the port does not). If
R1O_EMBED_MODEL is set, the served model's basename must match it.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

try:
    import tomllib
except ImportError:  # pragma: no cover — Python < 3.11
    tomllib = None  # type: ignore[assignment]

# Service contract of com.r1o.embed-server; the user config's [embed].port wins.
DEFAULT_EMBED_PORT = 19091
PROBE_TIMEOUT_S = float(os.getenv("MODEL_KB_EMBED_PROBE_TIMEOUT", "1.5"))

_cached: str | None = None


class EmbedServerUnavailable(RuntimeError):
    """No embedding server proved itself. Callers must not write vectors."""


def _config_path() -> Path:
    return Path(os.getenv("QWEN_EMBED_CONFIG", str(Path.home() / ".config" / "qwen-embed" / "config.toml")))


def _load_config() -> dict[str, Any]:
    path = _config_path()
    if tomllib is None or not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f).get("embed", {}) or {}
    except (OSError, ValueError):
        return {}


def _url(host: str, port: int) -> str:
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


def probe(url: str, timeout: float = PROBE_TIMEOUT_S) -> tuple[bool, str]:
    """(proven, reason). Proven = /health names an embedding model with dims > 0."""
    parts = urlsplit(url)
    host, port = parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)
    if not host:
        return False, "not a URL"
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as e:
        return False, f"connect: {e.strerror or e}"
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=timeout) as r:
            body = json.loads(r.read() or b"{}") if r.status == 200 else {}
    except urllib.error.HTTPError as e:
        return False, f"/health HTTP {e.code}"
    except (OSError, ValueError) as e:
        return False, f"/health: {e}"
    emb = body.get("embedding") if isinstance(body, dict) else None
    try:
        dims = int((emb or {}).get("dimensions") or 0)
    except (TypeError, ValueError):
        dims = 0
    if not isinstance(emb, dict) or dims <= 0:
        return False, "/health has no embedding section"
    want = os.getenv("R1O_EMBED_MODEL", "").strip()
    if want and Path(str(emb.get("model") or "")).name != want:
        return False, f"serves {emb.get('model')!r}, R1O_EMBED_MODEL wants {want!r}"
    return True, f"{emb.get('model')} dims={dims}"


def _asmi_get(path: str) -> Any:
    base = os.getenv("ASMI_URL", "http://localhost:9090").rstrip("/")
    with urllib.request.urlopen(f"{base}{path}", timeout=max(PROBE_TIMEOUT_S, 3.0)) as r:
        return json.loads(r.read() or b"null")


def live_nodes() -> list[str]:
    """Online node names from asmi /cluster; /nodes (no online flag) if that fails."""
    try:
        rows = _asmi_get("/cluster")
        if isinstance(rows, dict):
            rows = rows.get("nodes") or []
        names = [str(n["hostname"]) for n in rows if isinstance(n, dict) and n.get("hostname") and n.get("online") is not False]
        if names:
            return names
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        return [str(n) for n in (_asmi_get("/nodes") or {}).get("nodes") or []]
    except (OSError, ValueError, AttributeError, TypeError):
        return []


def _tailscale_candidates() -> list[str]:
    """Absolute tailscale CLI paths, PATH-independent (cron PATH is minimal).

    Each is canonicalized: the App binary must run from its real path (through a
    symlink it crashes with an unknown bundle identifier).
    """
    search = os.pathsep.join(
        [os.environ.get("PATH", ""), "/opt/homebrew/bin", "/usr/local/bin", "/Applications/Tailscale.app/Contents/MacOS"]
    )
    found = [shutil.which("tailscale", path=search), shutil.which("Tailscale", path=search)]
    out: list[str] = []
    for p in found:
        if p:
            real = os.path.realpath(p)
            if real not in out and os.access(real, os.X_OK):
                out.append(real)
    return out


def tailnet_addresses() -> dict[str, list[str]]:
    """Lower-cased node name -> tailnet addresses (IPv4 first), online peers + self.

    Only a parsed status with BackendState == Running counts; the App binary
    under launchd prints a GUI error and exits 0, so exit status is not state.
    """
    env = dict(os.environ, TAILSCALE_BE_CLI="1")
    for exe in _tailscale_candidates():
        try:
            r = subprocess.run([exe, "status", "--json"], capture_output=True, text=True, timeout=8, env=env)
            status = json.loads(r.stdout)
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
        if not isinstance(status, dict) or status.get("BackendState") != "Running":
            continue
        out: dict[str, list[str]] = {}
        peers = list((status.get("Peer") or {}).values())
        if isinstance(status.get("Self"), dict):
            peers.append(status["Self"])
        for p in peers:
            if p.get("Online") is False:
                continue
            ips = sorted(p.get("TailscaleIPs") or [], key=lambda a: ":" in a)
            dns_label = str(p.get("DNSName") or "").split(".", 1)[0].lower()
            for key in {dns_label, str(p.get("HostName") or "").lower()} - {""}:
                out.setdefault(key, [])
                out[key] += [a for a in ips if a not in out[key]]
        return out
    return {}


def _resolver_addresses(name: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(name, port, type=socket.SOCK_STREAM)
    except OSError:
        return []
    seen: list[str] = []
    for info in infos:
        addr = info[4][0]
        if addr not in seen:
            seen.append(addr)
    return sorted(seen, key=lambda a: ":" in a)


def discover_embed_server(force_refresh: bool = False, trace: list[str] | None = None) -> str:
    """Return the base URL of a proven embedding server or raise EmbedServerUnavailable."""
    global _cached
    if _cached and not force_refresh:
        return _cached
    trace = trace if trace is not None else []
    cfg = _load_config()

    explicit = os.getenv("QWEN_EMBED_SERVER") or os.getenv("EMBED_URL") or cfg.get("server_url")
    if explicit:
        url = str(explicit).rstrip("/")
        ok, why = probe(url)
        trace.append(f"override {url}: {why}")
        if not ok:
            raise EmbedServerUnavailable(f"embed server override {url} is not an embedding server ({why})")
        _cached = url
        return url

    port = int(cfg.get("port") or DEFAULT_EMBED_PORT)
    nodes = live_nodes()
    if not nodes and not cfg.get("hosts"):
        raise EmbedServerUnavailable("asmi returned no live nodes and no [embed].hosts is configured")
    tailnet = tailnet_addresses()
    trace.append(f"asmi nodes={nodes} tailnet_names={len(tailnet)}")

    candidates: list[tuple[str, str]] = []
    for host in cfg.get("hosts") or []:
        candidates += [(str(host), a) for a in _resolver_addresses(str(host), port)]
    for name in nodes:
        addrs = tailnet.get(name.lower(), []) + _resolver_addresses(name, port)
        candidates += [(name, a) for a in addrs]

    tried: set[str] = set()
    for name, addr in candidates:
        if addr in tried:
            continue
        tried.add(addr)
        url = _url(addr, port)
        ok, why = probe(url)
        trace.append(f"{name} {url}: {why}")
        if ok:
            _cached = url
            return url
    raise EmbedServerUnavailable(
        f"no embedding server answered on port {port} across {len(nodes)} live node(s): " + "; ".join(trace[-12:])
    )


if __name__ == "__main__":
    import sys

    t: list[str] = []
    try:
        print(discover_embed_server(trace=t))
        rc = 0
    except EmbedServerUnavailable as e:
        print(f"FAIL: {e}", file=sys.stderr)
        rc = 2
    for line in t:
        print("  " + line, file=sys.stderr)
    raise SystemExit(rc)
