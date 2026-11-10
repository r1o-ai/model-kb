"""lib/modelkb_client.py — talk to a model-kb HTTP server (stdlib urllib only).

The HTTP server (`engine/model-kb-http.py`) is model-kb's front door to the one
recipe writer, `recipe_store.put`. Anything that is not the model-kb process
itself — the r1o web app, a second Home, a shell script — writes a recipe by
calling this client instead of reaching into `recipes.jsonl` or shelling into
`model-kb-mcp.py`. The server keeps every model-kb guard intact (writer-node
check, `.ingest.lock`, validation, snapshot re-export) and reads the record back
so a caller can prove the write landed.

    from lib.modelkb_client import get_recipe, health, put_recipe
    if health()["writer"]:
        put_recipe("Qwen/Qwen3-30B-A3B", "mlx_lm", "user", {"sampling": {"temperature": 0.2}})

Configuration is read at call time, so a test or a second install can point it
elsewhere without touching this file:

    MODEL_KB_ROOT          the model-kb install dir (default: this engine's dir)
    MODEL_KB_HTTP_URL      full base URL (default http://127.0.0.1:<DEFAULT_PORT>)
    MODEL_KB_HTTP_PORT     server port (read by the server, not by this client)
    MODEL_KB_HTTP_TOKEN    path to the bearer-token file (default <root>/http-token)
    MODEL_KB_HTTP_TIMEOUT_S  request timeout in seconds (default 15)

Every failure raises a `ModelKbHttpError` subclass carrying the server's
`code`, its one-sentence `reason` and the HTTP `status`, so a caller can tell
"retry after the ingest" (409) from "this node may not write" (403) from "the
patch is wrong" (422).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote

# The one place model-kb's HTTP port default is written down. Chosen outside
# r1o's Shared/ports.json and outside the 19080-19999 block asmi hands out for
# serve processes; the server and every client read it from here.
DEFAULT_PORT = 59710
TOKEN_NAME = "http-token"
DEFAULT_TIMEOUT_S = 15.0

__all__ = [
    "DEFAULT_PORT", "DEFAULT_TIMEOUT_S", "ModelKbHttpError", "ModelKbInvalid",
    "ModelKbNotWriter", "ModelKbRetry", "ModelKbUnavailable", "TOKEN_NAME",
    "base_url", "get_recipe", "health", "put_recipe", "read_token", "resolve_root", "token_path",
]


# -- configuration -------------------------------------------------------------

def resolve_root() -> Path:
    """The model-kb install directory: MODEL_KB_ROOT, else this engine's own dir.

    Every script in this repo resolves its root as `Path(__file__).parent`, so
    the default is the same directory they use — on the live install that is
    `~/.r1o/model-kb` (or the legacy `~/infra/model-kb`) because that is where
    the engine is installed. No home path is written down here.
    """
    env = os.environ.get("MODEL_KB_ROOT", "").strip()
    return Path(env).expanduser() if env else Path(__file__).resolve().parents[1]


def token_path(root: Path | None = None) -> Path:
    """Where the bearer token lives: MODEL_KB_HTTP_TOKEN (a path), else <root>/http-token."""
    env = os.environ.get("MODEL_KB_HTTP_TOKEN", "").strip()
    return Path(env).expanduser() if env else (root or resolve_root()) / TOKEN_NAME


def base_url() -> str:
    """The server's base URL: MODEL_KB_HTTP_URL, else loopback on DEFAULT_PORT."""
    env = os.environ.get("MODEL_KB_HTTP_URL", "").strip()
    return (env or f"http://127.0.0.1:{DEFAULT_PORT}").rstrip("/")


def _timeout() -> float:
    raw = os.environ.get("MODEL_KB_HTTP_TIMEOUT_S", "").strip()
    try:
        return float(raw) if raw else DEFAULT_TIMEOUT_S
    except ValueError:
        return DEFAULT_TIMEOUT_S


# -- errors --------------------------------------------------------------------

class ModelKbHttpError(RuntimeError):
    """A model-kb HTTP call failed. `status` 0 means it never reached the server."""

    def __init__(self, status: int, code: str, reason: str, payload: Any = None) -> None:
        super().__init__(reason)
        self.status = status
        self.code = code
        self.reason = reason
        self.payload = payload

    @property
    def retry(self) -> bool:
        """True only when nothing was written and trying again later is the fix."""
        return self.status == 409


class ModelKbUnavailable(ModelKbHttpError):
    """No answer from the server (down, wrong port, wrong host)."""


class ModelKbRetry(ModelKbHttpError):
    """409 — an ingest holds the store lock; nothing was written."""


class ModelKbNotWriter(ModelKbHttpError):
    """403 — this node's crontab does not run the pipeline, so it may not write."""


class ModelKbInvalid(ModelKbHttpError):
    """422 — the server refused the input; fix the arguments and resend."""


_SUBCLASSES: dict[int, type[ModelKbHttpError]] = {
    403: ModelKbNotWriter,
    409: ModelKbRetry,
    422: ModelKbInvalid,
}


# -- transport -----------------------------------------------------------------

def _request(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    token: str | None = None,
    auth: bool = True,
    url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    base = (url or base_url()).rstrip("/")
    target = f"{base}{path}"
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if auth:
        headers["Authorization"] = f"Bearer {token or read_token()}"
    req = urllib.request.Request(target, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout or _timeout()) as resp:
            raw = resp.read()
            return json.loads(raw or b"{}")
    except urllib.error.HTTPError as e:
        with e:  # closes the socket; without this urllib leaks the response
            raw = e.read() if e.fp else b""
        try:
            payload = json.loads(raw or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"code": "unparseable", "reason": raw.decode(errors="replace")[:300]}
        if not isinstance(payload, dict):
            payload = {"code": "unparseable", "reason": str(payload)[:300]}
        code = str(payload.get("code") or f"http-{e.code}")
        reason = str(payload.get("reason") or f"{method} {path} failed with HTTP {e.code}")
        raise _SUBCLASSES.get(e.code, ModelKbHttpError)(e.code, code, reason, payload) from e
    except (urllib.error.URLError, OSError) as e:
        raise ModelKbUnavailable(
            0, "unreachable",
            f"model-kb HTTP server at {base} did not answer ({e}) — start it with "
            f"`python3 {resolve_root() / 'model-kb-http.py'}`",
        ) from e


def read_token(root: Path | None = None) -> str:
    """The bearer token, or a clear error that says which file is missing."""
    path = token_path(root)
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise ModelKbUnavailable(
            0, "no-token",
            f"cannot read the model-kb HTTP token at {path} ({e}) — create it with "
            f"`python3 {resolve_root() / 'model-kb-http.py'} --init-token` or set MODEL_KB_HTTP_TOKEN",
        ) from e
    if not token:
        raise ModelKbUnavailable(0, "no-token", f"the model-kb HTTP token file {path} is empty")
    return token


def _recipe_path(model: str, engine: str, source: str) -> str:
    # Percent-encode each segment: HF model ids contain "/" (Qwen/Qwen3-30B-A3B),
    # and an encoded model stays one path segment instead of looking like four.
    return f"/v1/recipes/{quote(model, safe='')}/{quote(engine, safe='')}/{quote(source, safe='')}"


# -- operations ----------------------------------------------------------------

def health(*, url: str | None = None, timeout: float | None = None) -> dict[str, Any]:
    """{ok, model_kb_dir, writer, writer_why, ...} — open, needs no token."""
    return _request("GET", "/health", auth=False, url=url, timeout=timeout)


def get_recipe(
    model: str,
    engine: str,
    source: str,
    *,
    url: str | None = None,
    token: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any] | None:
    """The (model, engine, source) record, or None when the store holds none."""
    try:
        payload = _request("GET", _recipe_path(model, engine, source), url=url, token=token, timeout=timeout)
    except ModelKbHttpError as e:
        if e.status == 404:
            return None
        raise
    record = payload.get("record")
    return record if isinstance(record, dict) else None


def put_recipe(
    model: str,
    engine: str,
    source: str,
    patch: dict[str, Any],
    *,
    url: str | None = None,
    token: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Merge `patch` into the record through `recipe_store.put` and return the write proof.

    Returns the server's success payload: `record` (what the writer wrote),
    `read_back` (what `recipe_store.get` found afterwards) and `snapshot`
    (the re-exported recipes.snapshot.json). A null value in `patch` deletes
    that key. Raises ModelKbRetry (409), ModelKbNotWriter (403) or
    ModelKbInvalid (422) — each carries the server's reason.
    """
    return _request("PUT", _recipe_path(model, engine, source), body={"patch": patch},
                    url=url, token=token, timeout=timeout)
