# The recipe HTTP API — write a recipe from anything, not just MCP

`recipes.jsonl` has exactly one writer, `engine/recipe_store.py` (`put()` under
`.ingest.lock`, writer node only). Until now the only way to reach it from
outside model-kb was the stdio MCP tool `model_recipe_put`, so every other
consumer — r1o's web app, a shell script, a second Home — had to import
`model-kb-mcp.py` in a short-lived `python3`. `engine/model-kb-http.py` is the
front door instead: model-kb owns the write API, and its clients are clients.

It is stdlib only (`http.server`, `json`, `secrets`, `urllib`), so there is no
dependency to install, and it never writes the store itself — it calls
`recipe_store.put` and then `recipe_store.get`, so the answer carries the record
that was actually read back.

## How to run it

```bash
cd /path/to/model-kb            # the install whose crontab runs pipeline.py
python3 engine/model-kb-http.py --init-token   # creates <root>/http-token (0600), prints it
python3 engine/model-kb-http.py                # http://127.0.0.1:59710
python3 engine/model-kb-http.py --port 0       # any free port; the real one is printed on start
```

The server binds `127.0.0.1` only. `GET /health` is open; everything else needs
`Authorization: Bearer <token>`. The token lives in `<root>/http-token`
(override: `MODEL_KB_HTTP_TOKEN=<path>`), is created 0600 by `--init-token` or
by the first start, and is never rewritten once it holds a value. **No token
file → every non-health request is refused with 401 and a reason** (fail
closed); it never falls back to open.

On macOS, `deploy/com.model-kb.http.plist.in` is a launchd template; the header
of that file is the install recipe.

## Options

| Flag | Env | Default |
|---|---|---|
| `--port` | `MODEL_KB_HTTP_PORT` | `59710` (`lib/modelkb_client.DEFAULT_PORT`) |
| `--host` | — | `127.0.0.1` |
| `--root` | `MODEL_KB_ROOT` | the directory holding `model-kb-http.py` |
| `--token-file` | `MODEL_KB_HTTP_TOKEN` | `<root>/http-token` |
| `--max-body-bytes` | — | `1048576` (1 MiB) |
| `--init-token` | — | create the token file 0600 if absent, print it, exit |

**The port.** 59710 sits outside r1o's `Shared/ports.json` (59408 web, 59409
pair server, 59410 edge, 9090 asmi, 8317 cliproxy) and outside the 19080–19999
block asmi hands out to serve processes. It is written down in exactly one place
— `engine/lib/modelkb_client.py` — and the server, the client and the tests all
read it from there.

## Routes

One write shape: all three keys are in the path, so `source` can never disagree
with the URL. `POST` is refused with 405 and a reason naming the route to use.

### `GET /health` — open, no token

```json
{ "ok": true,
  "model_kb_dir": "/Users/me/.r1o/model-kb",
  "writer": true,
  "writer_why": "crontab runs pipeline.py: 0 */3 * * * /bin/bash /Users/me/bin/kb-ingest.sh",
  "token_configured": true,
  "store": "/Users/me/.r1o/model-kb/recipes.jsonl",
  "store_present": true,
  "routes": ["GET  /health", "GET  /v1/recipes/{model}/{engine}/{source}", "PUT  /v1/recipes/{model}/{engine}/{source}   body {\"patch\": {...}}"] }
```

`writer` is `recipe_store.is_writer_node(root)` — the same crontab check a write
runs, reported up front so a client can skip a write it knows will be refused.
It never includes the token.

### `GET /v1/recipes/{model}/{engine}/{source}`

`{model}` may be percent-encoded (`Qwen%2FQwen3-30B-A3B`) or raw
(`Qwen/Qwen3-30B-A3B`): the path is read right-to-left, and `engine`/`source`
never contain `/`, so both name the same record.

```
GET /v1/recipes/Qwen%2FQwen3-30B-A3B/mlx_lm/user
200 {"ok": true, "record": {"model": "Qwen/Qwen3-30B-A3B", "engine": "mlx_lm", "source": "user",
     "recipe": {"sampling": {"temperature": 0.2}}, "created": "…", "updated": "…",
     "history": [{"at": "…", "patch": {"sampling": {"temperature": 0.2}}}]}}
```

### `PUT /v1/recipes/{model}/{engine}/{source}` — body `{"patch": {...}}`

`patch` deep-merges into the record (`recipe_store` semantics); a `null` value
deletes that key. Each write appends to `history[]`, rewrites `recipes.jsonl`
atomically and re-exports `recipes.snapshot.json` (0444). The response carries
`record` (what the writer returned) **and** `read_back` (what
`recipe_store.get` found afterwards), so a caller proves the write landed
instead of trusting a 200:

```
PUT /v1/recipes/Qwen%2FQwen3-30B-A3B/mlx_lm/user
{"patch": {"sampling": {"temperature": 0.2}}}
200 {"ok": true,
     "record":    {…the record recipe_store.put returned…},
     "read_back": {…the same record, re-read from recipes.jsonl…},
     "snapshot":  "/Users/me/.r1o/model-kb/recipes.snapshot.json"}
```

## Errors

Every error is `{"ok": false, "code": "...", "reason": "<one actionable sentence>"}`
— never a bare code.

| Status | `code` | When | What a caller does |
|---|---|---|---|
| 400 | `bad-json` | body is not JSON | fix the body |
| 400 | `bad-request` | chunked body, non-integer `Content-Length`, short body | send `Content-Length` + small JSON |
| 401 | `unauthorized` | no/wrong bearer token, or no token file configured | read `<root>/http-token`; `--init-token` if absent |
| 403 | `not-writer-node` | this node's crontab does not run `pipeline.py` | write from the writer node |
| 404 | `not-found` | unknown record, incomplete path, unrouted path | fix the path |
| 405 | `method-not-allowed` | any verb other than GET/PUT on a recipe, non-GET on `/health` | use the route named in `reason` |
| 409 | `retry` | an ingest holds `.ingest.lock`; **nothing was written** | retry after the ingest |
| 413 | `too-large` | body over `--max-body-bytes` (default 1 MiB) | shrink the patch |
| 422 | `invalid` | `recipe_store._validate` refused the input (source, model, engine, patch shape) | fix the argument named in `reason` |
| 500 | `internal` / `read-back-failed` | store unreadable, or written-but-not-readable | inspect the store; a read-back failure means do not trust it |

```
409 {"ok": false, "code": "retry",
     "reason": "model-kb ingest running (lock held by 4242 2026-11-10T05:57:33+00:00 pipeline); nothing written — retry after it finishes"}
403 {"ok": false, "code": "not-writer-node",
     "reason": "no crontab entry on this node runs pipeline.py for /Users/me/.r1o/model-kb; write on the node that runs the model-kb pipeline"}
422 {"ok": false, "code": "invalid", "reason": "source must be one of research, benchmark, user (got 'guess')"}
```

## Concurrency

The store's own `.ingest.lock` is the serialisation — this server adds no cache
and holds no state between requests, so several servers may run against one root
and two concurrent PUTs both land (neither patch is lost; the second waits up to
`MODEL_KB_PUT_LOCK_TIMEOUT_S`, default 10 s, then gets 409).

## The client

`engine/lib/modelkb_client.py` is the stdlib client r1o and any future consumer
should copy or import:

```python
from lib.modelkb_client import health, get_recipe, put_recipe

health()                                  # {"ok": True, "writer": True, ...}
get_recipe("Qwen/Qwen3-30B-A3B", "mlx_lm", "user")          # record or None
put_recipe("Qwen/Qwen3-30B-A3B", "mlx_lm", "user", {"sampling": {"temperature": 0.2}})
```

It raises `ModelKbRetry` (409), `ModelKbNotWriter` (403), `ModelKbInvalid`
(422) or `ModelKbUnavailable` (never reached the server), each carrying
`status`, `code` and the server's `reason`. Configuration is read at call time:
`MODEL_KB_HTTP_URL`, `MODEL_KB_HTTP_TOKEN`, `MODEL_KB_HTTP_TIMEOUT_S`,
`MODEL_KB_ROOT`.

### Moving r1o's `recipe-writer.ts` onto it

r1o today spawns `python3` that imports `model-kb-mcp.py`
(`web/src/lib/model-kb/recipe-writer.ts`). The replacement is one fetch — the
route's shapes already line up (`record` + `readBack`, 409 on retry, 422
otherwise):

```ts
const base = process.env.MODEL_KB_HTTP_URL ?? `http://127.0.0.1:${59710}`;
const token = (await readFile(process.env.MODEL_KB_HTTP_TOKEN ?? `${modelKbDir()}/http-token`, 'utf8')).trim();
const res = await fetch(`${base}/v1/recipes/${encodeURIComponent(model)}/${encodeURIComponent(engine)}/${source}`, {
  method: 'PUT',
  headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
  body: JSON.stringify({ patch }),
});
const body = await res.json();
// body.ok, body.record, body.read_back (was readBack), body.reason, body.code === 'retry'
```

`read_back` (snake_case, like the rest of the model-kb JSON) is the one rename
on the r1o side. No `python3`, no `model-kb-mcp.py` import, no venv discovery:
the server is model-kb's, so the guards live with the store.
