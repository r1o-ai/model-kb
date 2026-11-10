"""r1o_serve.py — ask r1o to serve a model-kb recipe on a machine.

One request:

    POST <r1o>/api/models/kb/recipes/<recipe_id>/serve    {"node"?: str, "port"?: int}

(r1o web `app/api/models/kb/recipes/[id]/serve/route.ts` → `serveRecipeOnNode`,
the same path the topology Recipes tab and the r1o Agent's `serve_apply` use.)

  single-machine recipe — `node` is required. r1o never picks a machine, and
                          neither does this module.
  multi-machine recipe  — loads on the r1o coordinator with its hostfile; a
                          `node` is refused.
  port                  — optional override of the recipe's serve.port.

The answer is r1o's. A refusal carries r1o's `error`, plus its `problems`,
`fit`, `engine_required` and `warnings` when it sent them, and the whole reply
under `r1o`. The request is sent once: never retried, never re-aimed at another
machine.

Consent. The route is wrapped in r1o's model-launch approval. An approval is a
single-use token r1o mints only for a request the operator confirmed in the
app's own window, or for an r1o Agent tool call (approved on its card, or under
a standing allow the operator set). r1o's rule for everything else is that
automation must not mint or reuse one. This caller is neither, so it sends no
token and r1o answers 403 MODEL_LAUNCH_APPROVAL_REQUIRED. That comes back as
`consent_required` with where the operator can start the same launch. It is
never worked around.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

# r1o web lib/model-launch-operations.ts MODEL_LAUNCH_APPROVAL_CODE.
APPROVAL_CODE = "MODEL_LAUNCH_APPROVAL_REQUIRED"
_REFUSAL_KEYS = ("problems", "fit", "engine_required", "warnings", "reason")
_SERVED_KEYS = ("node", "engine", "port", "model", "warnings")

APPROVE_WHERE = [
    "r1o app: Topology → Recipes, open this recipe, pick the machine, confirm the launch.",
    "r1o Agent: ask it to serve this recipe; its serve_apply tool asks for approval on its card "
    "(or runs under a standing allow the operator set).",
]


def serve_path(recipe_id: str) -> str:
    return f"/api/models/kb/recipes/{urllib.parse.quote(str(recipe_id), safe='')}/serve"


def serve_url(base: str, recipe_id: str) -> str:
    return base.rstrip("/") + serve_path(recipe_id)


def serve_body(node: str | None = None, port: int | None = None) -> dict[str, Any]:
    """The route's JSON body. Nothing is filled in: no node unless the caller named one."""
    body: dict[str, Any] = {}
    if node and str(node).strip():
        body["node"] = str(node).strip()
    if port is not None:
        body["port"] = int(port)
    return body


def request_problems(rec: dict[str, Any], node: str | None) -> list[str]:
    """What r1o refuses about node vs recipe before it looks at any machine
    (serveRecipeOnNode: a recipe is multi-machine when hardware.min_nodes > 1)."""
    rid = rec.get("recipe_id") or rec.get("id")
    min_nodes = (rec.get("hardware") or {}).get("min_nodes")
    multi = isinstance(min_nodes, (int, float)) and min_nodes > 1
    named = bool(node and str(node).strip())
    if multi and named:
        return [f"recipe {rid} needs {int(min_nodes)} machines: r1o loads it on its coordinator with the "
                f"recipe's hostfile and refuses a node (got {str(node).strip()!r})"]
    if not multi and not named:
        return [f"name the machine to serve {rid} on (node): r1o does not pick one, and neither does model-kb"]
    return []


def serve_recipe(
    base: str,
    recipe_id: str,
    *,
    node: str | None = None,
    port: int | None = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """POST the serve request once and report r1o's answer.

    Returns {ok, serve_url, request, http_status, …}: on success r1o's node /
    engine / port; on a refusal `error` and whatever r1o explained it with; on a
    launch-approval refusal `consent_required` and `approve`. `r1o` is the reply
    as sent whenever it was a JSON object.
    """
    url = serve_url(base, recipe_id)
    body = serve_body(node, port)
    out: dict[str, Any] = {"ok": False, "serve_url": url, "request": body}
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code, raw = resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        with e:
            code, raw = e.code, e.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001 — connection refused, timeout, DNS: all "r1o did not answer"
        out["http_status"] = 0
        out["error"] = f"r1o did not answer at {url}: {e}. Nothing was started by this call."
        return out

    out["http_status"] = code
    try:
        data = json.loads(raw) if raw.strip() else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        if code == 404:
            out["error"] = (
                f"r1o at {base.rstrip('/')} has no recipe serve route (HTTP 404 for POST {serve_path(recipe_id)}, "
                "not JSON): that r1o build predates it. Update r1o, then retry. Nothing was started."
            )
        else:
            out["error"] = f"r1o answered HTTP {code} without a JSON object for POST {serve_path(recipe_id)}"
        out["body"] = raw[:300]
        return out

    out["r1o"] = data
    if code == 403 and data.get("code") == APPROVAL_CODE:
        where = f" on {body['node']}" if body.get("node") else ""
        out["consent_required"] = True
        out["reason"] = data.get("reason")
        out["error"] = (
            f"r1o refused to start {recipe_id}{where}: a model launch needs the operator's explicit approval, "
            "and this tool cannot carry one (r1o mints an approval only for a request confirmed in its own "
            "window or for an r1o Agent tool call). Nothing was started."
        )
        out["approve"] = list(APPROVE_WHERE)
        return out
    if code in (200, 201) and data.get("ok") is True:
        out["ok"] = True
        out.update({k: data[k] for k in _SERVED_KEYS if data.get(k) is not None})
        return out
    out["error"] = str(data.get("error") or f"r1o answered HTTP {code}")
    out.update({k: data[k] for k in _REFUSAL_KEYS if data.get(k) is not None})
    return out
