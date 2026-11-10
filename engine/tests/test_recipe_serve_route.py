"""The load path: model_load / `model_kb.py load` ask r1o's recipe serve route.

    POST <r1o>/api/models/kb/recipes/<recipe_id>/serve   {"node"?, "port"?}

A stub stands in for r1o: it records every request and answers the way the
route does (served, launch approval required, a fit refusal with problems,
an unknown recipe, and an r1o build with no such route). No model is started
and nothing outside the temp dir is read.

    python3 -I -m unittest discover -s engine/tests -v
"""
from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from lib import r1o_serve  # noqa: E402
import serve_recipes  # noqa: E402

# Fixture names: they exist only in this stub.
NODE = "fx-node-a"
FIT = {"level": "blocked", "reasons": [f"{NODE} has 32 GB and the recipe needs 96 GB"]}
ANSWERS = {
    "fx-served": (200, {"ok": True, "status": 200, "recipe_id": "fx-served", "model": "Fx-Model-4bit",
                        "node": NODE, "engine": "fx_engine", "port": 19999, "asmi": {"status": "starting"},
                        "warnings": ["fx warning"]}),
    "fx-consent": (403, {"error": "Explicit approval is required before starting, loading, restarting, or scaling a model.",
                         "code": "MODEL_LAUNCH_APPROVAL_REQUIRED", "reason": "missing"}),
    "fx-blocked": (400, {"ok": False, "status": 400, "recipe_id": "fx-blocked", "node": NODE,
                         "error": f"fx-blocked cannot run on {NODE}: {FIT['reasons'][0]}",
                         "problems": FIT["reasons"], "fit": FIT}),
    "fx-unknown": (404, {"ok": False, "error": "model-kb has no serve recipe fx-unknown"}),
    "fx-old-r1o": (404, "<!DOCTYPE html><html><body>404</body></html>"),
}


class _R1o(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        self.server.seen.append({"path": self.path, "body": json.loads(raw or b"{}"),
                                 "headers": {k.lower(): v for k, v in self.headers.items()}})
        rid = self.path.removeprefix("/api/models/kb/recipes/").removesuffix("/serve")
        code, answer = ANSWERS.get(rid, (500, {"error": f"stub has no answer for {self.path}"}))
        out = answer.encode() if isinstance(answer, str) else json.dumps(answer).encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html" if isinstance(answer, str) else "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def _recipe(name: str, **hardware) -> dict:
    return serve_recipes.recipe_from_config({
        "name": name,
        "model": {"name": "Fx-Model-4bit", "hf_id": "fixture/Fx-Model"},
        "serve": {"engine": "fx_engine", "backend": "single", "port": 19999},
        "hardware": hardware,
    })


class RouteCase(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _R1o)
        self.server.seen = []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.tmp = tempfile.TemporaryDirectory()
        self.records = Path(self.tmp.name) / "records.jsonl"
        self.records.write_text("".join(
            json.dumps(r) + "\n"
            for r in [_recipe(n) for n in ANSWERS] + [_recipe("fx-multi", min_nodes=2)]
        ))

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    @property
    def seen(self) -> list[dict]:
        return self.server.seen


class TestServeRecipe(RouteCase):
    def test_one_post_to_the_recipe_route_with_the_named_node_and_port(self):
        got = r1o_serve.serve_recipe(self.base, "fx-served", node=NODE, port=19998)
        self.assertEqual([(r["path"], r["body"]) for r in self.seen],
                         [("/api/models/kb/recipes/fx-served/serve", {"node": NODE, "port": 19998})])
        self.assertIs(got["ok"], True)
        self.assertEqual((got["node"], got["engine"], got["port"]), (NODE, "fx_engine", 19999))
        self.assertEqual(got["r1o"], ANSWERS["fx-served"][1])
        self.assertEqual(got["warnings"], ["fx warning"])

    def test_no_node_is_ever_filled_in(self):
        r1o_serve.serve_recipe(self.base, "fx-served")
        self.assertEqual(self.seen[0]["body"], {})
        r1o_serve.serve_recipe(self.base, "fx-served", node="  ", port=None)
        self.assertEqual(self.seen[1]["body"], {})

    def test_no_approval_token_is_sent(self):
        r1o_serve.serve_recipe(self.base, "fx-consent", node=NODE)
        headers = self.seen[0]["headers"]
        self.assertNotIn("x-r1o-model-launch-approval", headers)
        self.assertNotIn("origin", headers)
        self.assertNotIn("authorization", headers)
        self.assertNotIn("cookie", headers)

    def test_launch_approval_refusal_says_what_was_refused_and_where_to_approve(self):
        got = r1o_serve.serve_recipe(self.base, "fx-consent", node=NODE)
        self.assertEqual(len(self.seen), 1, "a refusal is never retried")
        self.assertIs(got["ok"], False)
        self.assertIs(got["consent_required"], True)
        self.assertEqual(got["http_status"], 403)
        self.assertEqual(got["reason"], "missing")
        self.assertIn(f"refused to start fx-consent on {NODE}", got["error"])
        self.assertIn("Nothing was started", got["error"])
        where = " ".join(got["approve"])
        self.assertIn("Topology → Recipes", where)
        self.assertIn("serve_apply", where)
        self.assertEqual(got["r1o"], ANSWERS["fx-consent"][1])

    def test_a_fit_refusal_keeps_error_problems_and_fit_and_is_not_retried_elsewhere(self):
        got = r1o_serve.serve_recipe(self.base, "fx-blocked", node=NODE)
        self.assertEqual([r["body"] for r in self.seen], [{"node": NODE}])
        self.assertIs(got["ok"], False)
        self.assertEqual(got["http_status"], 400)
        self.assertEqual(got["error"], ANSWERS["fx-blocked"][1]["error"])
        self.assertEqual(got["problems"], FIT["reasons"])
        self.assertEqual(got["fit"], FIT)
        self.assertNotIn("consent_required", got)

    def test_an_unknown_recipe_is_r1os_own_answer(self):
        got = r1o_serve.serve_recipe(self.base, "fx-unknown", node=NODE)
        self.assertEqual((got["ok"], got["http_status"]), (False, 404))
        self.assertEqual(got["error"], "model-kb has no serve recipe fx-unknown")

    def test_an_r1o_without_the_route_is_named_not_an_opaque_404(self):
        got = r1o_serve.serve_recipe(self.base, "fx-old-r1o", node=NODE)
        self.assertEqual((got["ok"], got["http_status"]), (False, 404))
        self.assertIn("has no recipe serve route", got["error"])
        self.assertIn("Update r1o", got["error"])
        self.assertNotIn("r1o", got, "an HTML page is not r1o's JSON answer")
        self.assertIn("<html>", got["body"])

    def test_r1o_not_answering_is_an_error_with_its_reason(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            dead = f"http://127.0.0.1:{s.getsockname()[1]}"
        got = r1o_serve.serve_recipe(dead, "fx-served", node=NODE, timeout=2)
        self.assertEqual((got["ok"], got["http_status"]), (False, 0))
        self.assertIn("did not answer", got["error"])

    def test_recipe_ids_are_path_escaped(self):
        self.assertEqual(r1o_serve.serve_url("http://fx/", "a b/c"), "http://fx/api/models/kb/recipes/a%20b%2Fc/serve")


class TestRequestProblems(unittest.TestCase):
    def test_single_machine_recipe_needs_a_named_node(self):
        (why,) = r1o_serve.request_problems(_recipe("fx-single"), None)
        self.assertIn("name the machine", why)
        self.assertEqual(r1o_serve.request_problems(_recipe("fx-single"), NODE), [])

    def test_multi_machine_recipe_refuses_a_node(self):
        multi = _recipe("fx-multi", min_nodes=2)
        self.assertEqual(r1o_serve.request_problems(multi, ""), [])
        (why,) = r1o_serve.request_problems(multi, NODE)
        self.assertIn("needs 2 machines", why)
        self.assertIn(NODE, why)


class TestCli(RouteCase):
    def cli(self, *args: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, R1O_URL=self.base, ASMI_URL="http://127.0.0.1:9")
        return subprocess.run([sys.executable, str(ENGINE / "model_kb.py"), "--records", str(self.records), *args],
                              capture_output=True, text=True, timeout=60, env=env)

    def test_dry_run_shows_the_request_and_calls_nothing(self):
        p = self.cli("load", "fx-served", "--dry-run", "--node", NODE, "--port", "19998")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        out = json.loads(p.stdout)
        self.assertEqual(self.seen, [])
        self.assertEqual(out["via"], "dry-run")
        self.assertEqual(out["serve_url"], f"{self.base}/api/models/kb/recipes/fx-served/serve")
        self.assertEqual(out["request"], {"node": NODE, "port": 19998})
        self.assertNotIn("request_problems", out)
        self.assertIn("model_path_exists", out)

    def test_dry_run_without_a_node_says_r1o_would_refuse_it(self):
        out = json.loads(self.cli("load", "fx-served", "--dry-run").stdout)
        self.assertEqual(self.seen, [])
        self.assertEqual(out["request"], {})
        self.assertIn("name the machine", out["request_problems"][0])

    def test_live_load_posts_once_and_exits_zero_when_r1o_served_it(self):
        p = self.cli("load", "fx-served", "--node", NODE)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual([(r["path"], r["body"]) for r in self.seen],
                         [("/api/models/kb/recipes/fx-served/serve", {"node": NODE})])
        out = json.loads(p.stdout)
        self.assertEqual((out["ok"], out["via"], out["node"]), (True, "r1o-recipe-serve", NODE))

    def test_live_load_refused_for_consent_exits_nonzero_and_says_where_to_approve(self):
        p = self.cli("load", "fx-consent", "--node", NODE)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertEqual(len(self.seen), 1)
        out = json.loads(p.stdout)
        self.assertIs(out["consent_required"], True)
        self.assertIn("Nothing was started", p.stderr)
        self.assertIn("Topology → Recipes", p.stderr)


def _load_mcp(records: Path, base: str):
    """Import model-kb-mcp.py (a hyphenated script) with its env pointed at the fixtures."""
    saved = {k: os.environ.get(k) for k in ("MODEL_KB_RECORDS", "R1O_URL", "ASMI_URL")}
    os.environ.update(MODEL_KB_RECORDS=str(records), R1O_URL=base, ASMI_URL="http://127.0.0.1:9")
    # Other test modules import a temp copy of the engine; import this checkout's.
    for mod in [m for m in list(sys.modules)
                if m in ("serve_recipes", "recipe_store", "build_bm25", "lib") or m.startswith("lib.")]:
        del sys.modules[mod]
    if str(ENGINE) in sys.path:
        sys.path.remove(str(ENGINE))
    sys.path.insert(0, str(ENGINE))
    try:
        spec = importlib.util.spec_from_file_location("model_kb_mcp_under_test", ENGINE / "model-kb-mcp.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return mod


@unittest.skipUnless(importlib.util.find_spec("fastmcp"), "fastmcp not installed")
class TestMcpModelLoad(RouteCase):
    def setUp(self) -> None:
        super().setUp()
        tool = _load_mcp(self.records, self.base).model_load
        self.model_load = getattr(tool, "fn", tool)  # the plain function behind the tool

    def test_dry_run_is_the_default_and_calls_nothing(self):
        out = self.model_load("fx-served", node=NODE)
        self.assertEqual(self.seen, [])
        self.assertEqual((out["ok"], out["via"]), (True, "dry-run"))
        self.assertEqual(out["serve_url"], f"{self.base}/api/models/kb/recipes/fx-served/serve")
        self.assertEqual(out["request"], {"node": NODE})
        self.assertEqual(out["asmi_body"]["engine"], "fx_engine")

    def test_dry_run_names_what_r1o_would_refuse(self):
        self.assertIn("name the machine", self.model_load("fx-served")["request_problems"][0])
        self.assertIn("needs 2 machines", self.model_load("fx-multi", node=NODE)["request_problems"][0])
        self.assertNotIn("request_problems", self.model_load("fx-multi"))
        self.assertEqual(self.seen, [])

    def test_live_load_passes_node_and_port_and_returns_r1os_answer(self):
        out = self.model_load("fx-served", dry_run=False, node=NODE, port=19998)
        self.assertEqual([r["body"] for r in self.seen], [{"node": NODE, "port": 19998}])
        self.assertEqual((out["ok"], out["via"], out["node"], out["port"]), (True, "r1o-recipe-serve", NODE, 19999))

    def test_live_load_refused_for_consent_is_actionable(self):
        out = self.model_load("fx-consent", dry_run=False, node=NODE)
        self.assertEqual(len(self.seen), 1)
        self.assertEqual((out["ok"], out["consent_required"], out["http_status"]), (False, True, 403))
        self.assertIn("serve_apply", " ".join(out["approve"]))

    def test_live_load_refusal_carries_problems_and_fit(self):
        out = self.model_load("fx-blocked", dry_run=False, node=NODE)
        self.assertEqual(len(self.seen), 1)
        self.assertEqual((out["ok"], out["problems"], out["fit"]), (False, FIT["reasons"], FIT))


if __name__ == "__main__":
    unittest.main()
