#!/usr/bin/env python3
"""model-kb-http.py — model-kb's recipe HTTP API (stdlib only).

Why this exists: `recipe_store.put` is the ONE writer of `recipes.jsonl`, but the
only way to reach it from outside model-kb was the stdio MCP tool
`model_recipe_put` — so every other consumer (r1o's web app, a shell script, a
second Home) had to shell into a short-lived `python3` that imported
`model-kb-mcp.py` in-process. This server is the front door instead: model-kb
owns the write API, and its clients are clients.

Routes (base http://127.0.0.1:<port>):

  GET  /health                              open, no token
       → {ok, model_kb_dir, writer, writer_why, token_configured, store, store_present}
  GET  /v1/recipes/{model}/{engine}/{source}
       → 200 {ok, record} · 404 {ok:false, code:"not-found", reason}
  PUT  /v1/recipes/{model}/{engine}/{source}   body {"patch": {...}}
       → 200 {ok, record, read_back, snapshot} — the record is read back out of
         the store after the write, so a caller can prove it landed.
         A null value in `patch` deletes that key (recipe_store semantics).

There is deliberately only one write shape: all three keys live in the path, so
`source` can never disagree with the URL and no second shape can drift. POST is
refused with 405 and a reason that names the route to use.

Errors are always `{"ok": false, "code": "...", "reason": "<one sentence>"}`:
  400 bad-json / bad-request   401 unauthorized   403 not-writer-node
  404 not-found                405 method-not-allowed   409 retry (lock held)
  413 too-large                422 invalid              500 internal

`{model}` may be percent-encoded (HF ids contain "/") or left raw — the path is
read right-to-left, so `/v1/recipes/Qwen/Qwen3-30B-A3B/mlx_lm/user` and the
%2F form both name the same record.

Auth: a bearer token from a 0600 file (default `<root>/http-token`, override
with `MODEL_KB_HTTP_TOKEN`), created by `--init-token` or on first start.
Everything except `/health` requires it. No token readable → every non-health
request is refused (fail closed), never allowed.

Concurrency: the store's own `.ingest.lock` serialises writers. This server
holds no cache and no state across requests — every request re-reads the store,
so it is safe to run several of them against one root.

Run:
  python3 model-kb-http.py --init-token        # create the token file
  python3 model-kb-http.py                     # 127.0.0.1:59710
  python3 model-kb-http.py --port 0            # any free port, printed on start

Port: `MODEL_KB_HTTP_PORT`, default 59710 (`lib/modelkb_client.DEFAULT_PORT` —
the one place it is written down). 59710 is outside r1o's `Shared/ports.json`
(59408 web, 59409 pair, 59410 edge, 9090 asmi, 8317 cliproxy) and outside the
19080-19999 block asmi assigns to serve processes.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR))

import recipe_store  # noqa: E402  — the one writer, imported as every other entry point does
from lib.modelkb_client import (  # noqa: E402
    DEFAULT_PORT, TOKEN_NAME, resolve_root, token_path,
)

HOST = "127.0.0.1"
MAX_BODY_BYTES = 1 << 20  # 1 MiB — a recipe patch is a small JSON object
API_PREFIX = "/v1/recipes/"
SERVER_NAME = "model-kb-http"


# -- token ---------------------------------------------------------------------

def ensure_token(path: Path) -> tuple[str, bool]:
    """(token, created). Creates the file 0600 if absent; never rewrites one."""
    path = Path(path)
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing, False
    except FileNotFoundError:
        pass
    except OSError as e:
        raise SystemExit(f"{SERVER_NAME}: cannot read the token file {path}: {e}")
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # Another start won the race: its token is the one clients will read.
        return path.read_text(encoding="utf-8").strip(), False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token + "\n")
        os.chmod(path, 0o600)
    except OSError:
        path.unlink(missing_ok=True)
        raise
    return token, True


def read_token_file(path: Path) -> str | None:
    """The current token, or None when the file is missing/empty (→ fail closed)."""
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


# -- responses -----------------------------------------------------------------

def _err(code: str, reason: str) -> dict[str, Any]:
    return {"ok": False, "code": code, "reason": reason}


def _routes() -> list[str]:
    return [
        "GET  /health",
        "GET  /v1/recipes/{model}/{engine}/{source}",
        "PUT  /v1/recipes/{model}/{engine}/{source}   body {\"patch\": {...}}",
    ]


class RecipeHandler(BaseHTTPRequestHandler):
    """One request in, one JSON body out. No state is kept between requests."""

    server_version = SERVER_NAME + "/1"
    # HTTP/1.0: every response closes its connection, so an unread request body
    # can never be mistaken for the next request line. Local API, no keep-alive.
    protocol_version = "HTTP/1.0"

    # -- plumbing ---------------------------------------------------------

    @property
    def root(self) -> Path:
        return self.server.model_kb_root  # type: ignore[attr-defined]

    @property
    def token_file(self) -> Path:
        return self.server.token_file  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        self.server.log(f"{self.command} {self.path} -> {getattr(self, '_status', '?')}")  # type: ignore[attr-defined]

    def _json(self, status: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode()
        self._status = status
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _fail(self, status: int, code: str, reason: str, headers: dict[str, str] | None = None) -> None:
        self._json(status, _err(code, reason), headers)

    # -- auth -------------------------------------------------------------

    def _authorized(self) -> bool:
        """Check the bearer token against the file, read fresh. Refuses when absent."""
        token = read_token_file(self.token_file)
        if token is None:
            self._fail(
                401, "unauthorized",
                f"no token is configured at {self.token_file}, so no write can be trusted — "
                f"create it with `python3 {ENGINE_DIR / 'model-kb-http.py'} --init-token`",
                {"WWW-Authenticate": 'Bearer realm="model-kb"'},
            )
            return False
        header = self.headers.get("Authorization") or ""
        got = header[7:].strip() if header[:7].lower() == "bearer " else ""
        if not got:
            self._fail(
                401, "unauthorized",
                f"send `Authorization: Bearer <token>` using the token in {self.token_file}",
                {"WWW-Authenticate": 'Bearer realm="model-kb"'},
            )
            return False
        if not secrets.compare_digest(got.encode(), token.encode()):
            self._fail(
                401, "unauthorized",
                f"the bearer token does not match the one in {self.token_file}",
                {"WWW-Authenticate": 'Bearer realm="model-kb"'},
            )
            return False
        return True

    def _protect(self) -> bool:
        """Auth for every non-health route, including ones that answer 405/404.

        Reads the method's body with the same cap as PUT, so an oversized body is
        refused with 413 (and its bytes never reach a handler) whatever the verb.
        A body left unread is harmless: the response closes the connection.
        """
        self._raw_body = b""
        if not self._authorized():
            return False
        if self.command in ("PUT", "POST", "PATCH", "DELETE"):
            raw = self._body()
            if raw is None:
                return False
            self._raw_body = raw
        return True

    # -- request body ------------------------------------------------------

    def _drain(self, n: int) -> None:
        """Consume an oversized body before answering, so the caller reads the 413
        instead of a broken pipe while it is still writing."""
        remaining = min(n, self.server.max_body_bytes * 32)  # type: ignore[attr-defined]
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                return
            remaining -= len(chunk)

    def _body(self) -> bytes | None:
        """The request body, or None after answering with the reason it is unusable."""
        if (self.headers.get("Transfer-Encoding") or "").lower().find("chunked") >= 0:
            self._fail(400, "bad-request",
                       "chunked bodies are not supported — send Content-Length with a small JSON object")
            return None
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            return b""
        try:
            n = int(raw_len)
        except ValueError:
            self._fail(400, "bad-request", f"Content-Length {raw_len!r} is not an integer")
            return None
        limit = self.server.max_body_bytes  # type: ignore[attr-defined]
        if n > limit:
            self._drain(n)
            self._fail(413, "too-large", f"the request body is {n} bytes; the limit is {limit} bytes")
            return None
        data = self.rfile.read(n)
        if len(data) != n:
            self._fail(400, "bad-request", f"the request body ended after {len(data)} of {n} bytes")
            return None
        return data

    # -- dispatch ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path.rstrip("/") == "/health":
            return self._health()
        if not path.startswith(API_PREFIX):
            if not self._protect():
                return
            return self._fail(404, "not-found", f"no route for GET {path}; routes are: " + "; ".join(_routes()))
        if not self._protect():
            return
        return self._get_recipe(path)

    def do_PUT(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if not self._protect():
            return
        if not path.startswith(API_PREFIX):
            return self._fail(
                404, "not-found",
                f"no route for PUT {path}; write with PUT /v1/recipes/{{model}}/{{engine}}/{{source}}",
            )
        keys = self._keys(path)
        if keys is None:
            return
        model, engine, source = keys
        try:
            body = json.loads(self._raw_body or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return self._fail(400, "bad-json", f"the request body is not JSON ({e}) — send {{\"patch\": {{...}}}}")
        if not isinstance(body, dict):
            return self._fail(422, "invalid",
                              f"the request body must be a JSON object with a `patch` key (got {type(body).__name__})")
        if "patch" not in body:
            return self._fail(
                422, "invalid",
                'the request body must carry a `patch` object — e.g. {"patch": {"sampling": {"temperature": 0.2}}}',
            )
        self._put(model, engine, source, body["patch"])

    def do_POST(self) -> None:  # noqa: N802
        self._wrong_method()

    do_PATCH = do_DELETE = do_POST

    def _wrong_method(self) -> None:
        if not self._protect():
            return
        path = urlsplit(self.path).path
        if path.startswith(API_PREFIX):
            self._fail(
                405, "method-not-allowed",
                f"{self.command} is not supported on {path} — write with PUT and all three keys in the path "
                f"(PUT {path.rstrip('/')}/{{source}} with body {{\"patch\": {{...}}}})",
                {"Allow": "GET, PUT"},
            )
        elif path.rstrip("/") == "/health":
            self._fail(405, "method-not-allowed", "use GET /health", {"Allow": "GET"})
        else:
            self._fail(405, "method-not-allowed",
                       f"{self.command} is not supported; routes are: " + "; ".join(_routes()),
                       {"Allow": "GET, PUT"})

    # -- handlers ----------------------------------------------------------

    def _health(self) -> None:
        root = self.root
        try:
            writer, why = recipe_store.is_writer_node(root)
        except Exception as e:  # noqa: BLE001 — health must answer even when the check cannot run
            writer, why = False, f"the writer-node check failed: {e}"
        store = root / recipe_store.STORE
        self._json(200, {
            "ok": True,
            "model_kb_dir": str(root),
            "writer": bool(writer),
            "writer_why": why,
            "token_configured": read_token_file(self.token_file) is not None,
            "store": str(store),
            "store_present": store.exists(),
            "routes": _routes(),
        })

    def _keys(self, path: str) -> tuple[str, str, str] | None:
        """(model, engine, source) from the path, or None after answering why not.

        Read right-to-left: engine and source never contain "/", so the model may
        arrive either percent-encoded (`Qwen%2FQwen3-30B-A3B`) or raw.
        """
        parts = [unquote(p) for p in path[len(API_PREFIX):].split("/")]
        if len(parts) < 3 or any(p == "" for p in parts):
            self._fail(
                404, "not-found",
                f"{path} does not name a record — the form is "
                f"/v1/recipes/{{model}}/{{engine}}/{{source}} (all three, none empty)",
            )
            return None
        model = "/".join(parts[:-2])
        engine, source = parts[-2], parts[-1]
        if source not in recipe_store.SOURCES:
            self._fail(
                422, "invalid",
                f"source must be one of {', '.join(recipe_store.SOURCES)} (got {source!r})",
            )
            return None
        return model, engine, source

    def _get_recipe(self, path: str) -> None:
        keys = self._keys(path)
        if keys is None:
            return
        model, engine, source = keys
        try:
            rec = recipe_store.get(self.root, model=model, engine=engine, source=source)
        except (recipe_store.RecipeError, OSError) as e:
            return self._fail(500, "internal", f"the recipe store cannot be read: {e}")
        if rec is None:
            return self._fail(
                404, "not-found",
                f"no {source} recipe for {model} on {engine} in {self.root / recipe_store.STORE}",
            )
        self._json(200, {"ok": True, "record": rec})

    def _put(self, model: str, engine: str, source: str, patch: Any) -> None:
        # LockHeld → 409 (retry), NotWriterNode → 403 (this node may not write),
        # RecipeError → 422 (the caller must fix the input). Each carries the
        # reason recipe_store itself gives, so there is one explanation, not two.
        try:
            written = recipe_store.put(self.root, model=model, engine=engine, source=source, patch=patch,
                                       lock_timeout_s=self.server.lock_timeout_s)  # type: ignore[attr-defined]
        except recipe_store.LockHeld as e:
            return self._fail(409, "retry", str(e))
        except recipe_store.NotWriterNode as e:
            return self._fail(403, "not-writer-node", str(e))
        except recipe_store.RecipeError as e:
            return self._fail(422, "invalid", str(e))
        except OSError as e:
            return self._fail(500, "internal", f"the recipe store could not be written: {e}")
        except Exception as e:  # noqa: BLE001 — never answer a write with a bare 500
            self.server.log(traceback.format_exc())  # type: ignore[attr-defined]
            return self._fail(500, "internal", f"the recipe write failed: {e}")
        try:
            read_back = recipe_store.get(self.root, model=model, engine=engine, source=source)
        except (recipe_store.RecipeError, OSError) as e:
            return self._fail(
                500, "read-back-failed",
                f"the record was written but cannot be read back ({e}) — fix the store before trusting it",
            )
        if read_back is None:
            return self._fail(
                500, "read-back-failed",
                f"the write reported success but {source} {model} on {engine} is not in "
                f"{self.root / recipe_store.STORE} — the store may be corrupt",
            )
        self._json(200, {
            "ok": True,
            "record": written,
            "read_back": read_back,
            "snapshot": str(self.root / recipe_store.SNAPSHOT),
        })


class RecipeHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer carrying the three things a request needs: root, token, log."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], *, root: Path, token_file: Path,
                 max_body_bytes: int = MAX_BODY_BYTES, lock_timeout_s: float | None = None,
                 log=None) -> None:
        self.model_kb_root = Path(root)
        self.token_file = Path(token_file)
        self.max_body_bytes = int(max_body_bytes)
        # None → recipe_store's own MODEL_KB_PUT_LOCK_TIMEOUT_S. Tests set 0 so a
        # held lock answers 409 immediately instead of waiting.
        self.lock_timeout_s = lock_timeout_s
        self.log = log or (lambda line: print(line, file=sys.stderr, flush=True))
        super().__init__(address, RecipeHandler)


def make_server(port: int, *, root: Path, token_file: Path, host: str = HOST,
                max_body_bytes: int = MAX_BODY_BYTES, lock_timeout_s: float | None = None,
                log=None) -> RecipeHTTPServer:
    """Build (not start) a server. `port=0` binds a free port; read it back from
    `server.server_address[1]`. Tests use this and call `serve_forever` in a thread."""
    return RecipeHTTPServer((host, port), root=root, token_file=token_file,
                            max_body_bytes=max_body_bytes, lock_timeout_s=lock_timeout_s, log=log)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=SERVER_NAME, description="model-kb's recipe HTTP API")
    ap.add_argument("--port", type=int, default=int(os.environ.get("MODEL_KB_HTTP_PORT") or DEFAULT_PORT),
                    help=f"port to bind on loopback (default MODEL_KB_HTTP_PORT, else {DEFAULT_PORT};"
                         " 0 = any free port)")
    ap.add_argument("--host", default=HOST, help=f"bind address (default {HOST}; this is a local API)")
    ap.add_argument("--root", type=Path, default=None,
                    help="model-kb install dir (default MODEL_KB_ROOT, else this file's directory)")
    ap.add_argument("--token-file", type=Path, default=None,
                    help=f"bearer-token file (default MODEL_KB_HTTP_TOKEN, else <root>/{TOKEN_NAME})")
    ap.add_argument("--max-body-bytes", type=int, default=MAX_BODY_BYTES,
                    help=f"largest accepted request body (default {MAX_BODY_BYTES})")
    ap.add_argument("--init-token", action="store_true",
                    help="create the token file 0600 if absent, print its path, exit")
    ap.add_argument("--quiet", action="store_true", help="do not log each request")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser() if args.root else resolve_root()
    token_file = Path(args.token_file).expanduser() if args.token_file else token_path(root)

    token, created = ensure_token(token_file)
    if args.init_token:
        print(f"{'created' if created else 'exists'} {token_file} (mode 0600)")
        print(token)
        return 0

    log = (lambda line: None) if args.quiet else (lambda line: print(line, file=sys.stderr, flush=True))
    server = make_server(args.port, root=root, token_file=token_file, host=args.host,
                         max_body_bytes=args.max_body_bytes, log=log)
    host, port = server.server_address[0], server.server_address[1]
    try:
        writer, why = recipe_store.is_writer_node(root)
    except Exception as e:  # noqa: BLE001
        writer, why = False, f"the writer-node check failed: {e}"
    print(f"{SERVER_NAME} on http://{host}:{port}  root={root}  token={token_file} "
          f"({'created' if created else 'existing'})", flush=True)
    print(f"{SERVER_NAME} writer={'yes' if writer else 'no'} — {why}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
