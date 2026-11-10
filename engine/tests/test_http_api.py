"""model-kb's recipe HTTP API: routes, auth, status codes and the write proof.

Every test runs the real server (`engine/model-kb-http.py`) and the real client
(`engine/lib/modelkb_client.py`) against a temp store root, a temp HOME and a
fake `crontab` on PATH — so the writer-node rule is exercised through the same
code the fleet uses, and no test touches the real home, the live corpus or the
network.

    python3 -I -m unittest discover -s engine/tests -v
"""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import shutil
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from lib import modelkb_client as kb  # noqa: E402  — the client r1o and others copy

MODEL = "fixture/Fx-Model"          # fixture data: an HF-shaped id with a "/"
MODEL_PATH = MODEL.replace("/", "%2F")
ENGINE_ID = "fx_engine"


def _load_server_module():
    """Import model-kb-http.py (a hyphenated filename) as a module."""
    spec = importlib.util.spec_from_file_location("model_kb_http", ENGINE / "model-kb-http.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class HttpCase(unittest.TestCase):
    """A temp store root + a running server on an ephemeral port."""

    writer = True
    # None → recipe_store's own MODEL_KB_PUT_LOCK_TIMEOUT_S: a second writer waits
    # for the lock instead of failing. The lock test sets 0 to answer 409 at once.
    lock_timeout: float | None = None

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="model-kb-http-test-"))
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.root = self.tmp / "model-kb"
        self.root.mkdir()
        self.token_file = self.root / kb.TOKEN_NAME
        self.token = self._write_token("test-token-0123456789")

        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self._fake_crontab()
        self._saved_env = dict(os.environ)
        os.environ.update(HOME=str(self.home), PATH=f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}")
        os.environ.pop("MODEL_KB_HTTP_URL", None)
        os.environ.pop("MODEL_KB_HTTP_TOKEN", None)

        self.mod = _load_server_module()
        self.server = self.mod.make_server(
            0, root=self.root, token_file=self.token_file, lock_timeout_s=self.lock_timeout,
            log=lambda line: None,
        )
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        os.environ.clear()
        os.environ.update(self._saved_env)
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- fixtures ---------------------------------------------------------

    def _write_token(self, token: str) -> str:
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.token_file.write_text(token + "\n")
        self.token_file.chmod(0o600)
        return token

    def _fake_crontab(self) -> None:
        """`crontab -l` names this root's pipeline.py (writer) or nothing."""
        script = self.tmp / "cron" / "kb-ingest.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(f'#!/bin/bash\ncd "{self.root}" || exit 1\npython3 pipeline.py\n')
        line = f"0 */3 * * * /bin/bash {script}\n" if self.writer else "# no jobs\n"
        tab = self.bin / "crontab"
        tab.write_text(f"#!/bin/sh\ncat <<'EOF'\n{line}EOF\n")
        tab.chmod(0o755)

    # -- requests ---------------------------------------------------------

    def call(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        token: str | None = "__default__",  # type: ignore[assignment]
    ) -> tuple[int, dict]:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if token is not None:
            headers["Authorization"] = f"Bearer {self.token if token == '__default__' else token}"
        req = Request(self.url + path, data=body, method=method, headers=headers)
        try:
            with urlopen(req, timeout=20) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except HTTPError as e:
            with e:
                return e.code, json.loads(e.read() or b"{}")

    def put(self, patch, *, model=MODEL_PATH, source="user", token="__default__",
            engine=ENGINE_ID) -> tuple[int, dict]:
        body = json.dumps({"patch": patch}).encode()
        return self.call("PUT", f"/v1/recipes/{model}/{engine}/{source}", body, token)

    def store_rows(self) -> list[dict]:
        path = self.root / "recipes.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


class TestHealth(HttpCase):
    def test_health_is_open_and_names_the_writer(self):
        status, body = self.call("GET", "/health", token=None)
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["model_kb_dir"], str(self.root))
        self.assertIs(body["writer"], True, body)
        self.assertIn("pipeline.py", body["writer_why"])
        self.assertIs(body["token_configured"], True)
        self.assertNotIn(self.token, json.dumps(body), "health must never leak the token")

    def test_health_does_not_need_a_token_even_when_none_is_configured(self):
        self.token_file.unlink()
        status, body = self.call("GET", "/health", token=None)
        self.assertEqual(status, 200)
        self.assertIs(body["token_configured"], False)


class TestWriteRead(HttpCase):
    def test_put_then_get_round_trip_with_the_record_read_back(self):
        status, body = self.put({"sampling": {"temperature": 0.2}})
        self.assertEqual(status, 200, body)
        self.assertTrue(body["ok"])
        self.assertEqual(body["read_back"]["recipe"], {"sampling": {"temperature": 0.2}})
        self.assertEqual(body["read_back"]["source"], "user")
        self.assertEqual(body["snapshot"], str(self.root / "recipes.snapshot.json"))
        self.assertTrue((self.root / "recipes.snapshot.json").exists())
        self.assertEqual(stat.S_IMODE((self.root / "recipes.snapshot.json").stat().st_mode), 0o444)

        status, got = self.call("GET", f"/v1/recipes/{MODEL_PATH}/{ENGINE_ID}/user")
        self.assertEqual(status, 200, got)
        self.assertEqual(got["record"], body["read_back"])
        self.assertEqual([h["patch"] for h in got["record"]["history"]], [{"sampling": {"temperature": 0.2}}])

    def test_patch_deep_merges_and_null_deletes(self):
        self.put({"sampling": {"temperature": 0.2, "top_p": 0.9}})
        status, body = self.put({"sampling": {"top_p": None, "seed": 7}})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["read_back"]["recipe"], {"sampling": {"temperature": 0.2, "seed": 7}})

    def test_raw_slash_model_path_names_the_same_record(self):
        self.put({"a": 1})
        status, body = self.call("GET", f"/v1/recipes/{MODEL}/{ENGINE_ID}/user")
        self.assertEqual(status, 200, body)
        self.assertEqual(body["record"]["model"], MODEL)

    def test_the_client_round_trips_through_the_same_routes(self):
        os.environ["MODEL_KB_HTTP_URL"] = self.url
        os.environ["MODEL_KB_HTTP_TOKEN"] = str(self.token_file)
        self.assertTrue(kb.health()["ok"])
        out = kb.put_recipe(MODEL, ENGINE_ID, "user", {"sampling": {"temperature": 0.15}})
        self.assertEqual(out["read_back"]["recipe"], {"sampling": {"temperature": 0.15}})
        self.assertEqual(kb.get_recipe(MODEL, ENGINE_ID, "user")["recipe"],
                         {"sampling": {"temperature": 0.15}})
        self.assertIsNone(kb.get_recipe(MODEL, ENGINE_ID, "benchmark"))

    def test_the_client_maps_retry_to_its_own_error(self):
        self.server.lock_timeout_s = 0  # answer 409 instead of waiting out the timeout
        os.environ["MODEL_KB_HTTP_URL"] = self.url
        os.environ["MODEL_KB_HTTP_TOKEN"] = str(self.token_file)
        fh = open(self.root / ".ingest.lock", "a+")
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with self.assertRaises(kb.ModelKbRetry) as cm:
                kb.put_recipe(MODEL, ENGINE_ID, "user", {"a": 1})
            self.assertEqual(cm.exception.status, 409)
            self.assertTrue(cm.exception.retry)
            self.assertIn("retry", cm.exception.reason.lower())
        finally:
            fh.close()


class TestErrors(HttpCase):
    lock_timeout = 0

    def test_lock_held_is_409_retry_and_writes_nothing(self):
        fh = open(self.root / ".ingest.lock", "a+")
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write("4242 pipeline-under-test\n")
        fh.flush()
        try:
            status, body = self.put({"a": 1})
        finally:
            fh.close()
        self.assertEqual(status, 409, body)
        self.assertEqual(body["code"], "retry")
        self.assertFalse(body["ok"])
        self.assertIn("4242", body["reason"])
        self.assertEqual(self.store_rows(), [], "a locked put must not write")

    def test_unknown_record_is_404_with_a_reason(self):
        status, body = self.call("GET", f"/v1/recipes/{MODEL_PATH}/{ENGINE_ID}/research")
        self.assertEqual(status, 404, body)
        self.assertEqual(body["code"], "not-found")
        self.assertIn("research", body["reason"])

    def test_bad_json_is_400(self):
        status, body = self.call("PUT", f"/v1/recipes/{MODEL_PATH}/{ENGINE_ID}/user", b"{not json")
        self.assertEqual(status, 400, body)
        self.assertEqual(body["code"], "bad-json")
        self.assertTrue(body["reason"])

    def test_bad_patch_is_422_with_the_store_reason(self):
        for patch in ("strings-are-not-objects", [1, 2]):
            with self.subTest(patch=patch):
                status, body = self.put(patch)
                self.assertEqual(status, 422, body)
                self.assertEqual(body["code"], "invalid")
                self.assertIn("patch must be a JSON object", body["reason"])
        status, body = self.put({}, engine="bad%20engine")
        self.assertEqual(status, 422, body)
        self.assertIn("engine", body["reason"])

    def test_unknown_source_is_422(self):
        status, body = self.put({"a": 1}, source="guess")
        self.assertEqual(status, 422, body)
        self.assertIn("source must be one of", body["reason"])

    def test_wrong_method_is_405_with_the_route_to_use(self):
        status, body = self.call("POST", f"/v1/recipes/{MODEL_PATH}/{ENGINE_ID}", b'{"source":"user","patch":{}}')
        self.assertEqual(status, 405, body)
        self.assertEqual(body["code"], "method-not-allowed")
        self.assertIn("/v1/recipes/", body["reason"])

    def test_unrouted_path_is_404_naming_the_routes(self):
        status, body = self.call("GET", "/nope")
        self.assertEqual(status, 404, body)
        self.assertIn("/health", body["reason"])

    def test_incomplete_path_is_404_naming_the_form(self):
        status, body = self.call("PUT", f"/v1/recipes/{MODEL_PATH}/{ENGINE_ID}", b'{"patch":{}}')
        self.assertEqual(status, 404, body)
        self.assertIn("/v1/recipes/{model}/{engine}/{source}", body["reason"])


class TestAuth(HttpCase):
    def test_missing_token_is_401_and_nothing_is_written(self):
        status, body = self.put({"a": 1}, token=None)
        self.assertEqual(status, 401, body)
        self.assertEqual(body["code"], "unauthorized")
        self.assertIn("Bearer", body["reason"])
        self.assertEqual(self.store_rows(), [])

    def test_wrong_token_is_401(self):
        status, body = self.put({"a": 1}, token="not-the-token")
        self.assertEqual(status, 401, body)
        self.assertIn("does not match", body["reason"])
        self.assertEqual(self.store_rows(), [])

    def test_no_token_configured_fails_closed_on_every_non_health_route(self):
        self.token_file.unlink()
        status, body = self.call("GET", f"/v1/recipes/{MODEL_PATH}/{ENGINE_ID}/user", token="anything")
        self.assertEqual(status, 401, body)
        self.assertEqual(body["code"], "unauthorized")
        self.assertIn("--init-token", body["reason"])
        status, body = self.put({"a": 1}, token="anything")
        self.assertEqual(status, 401, body)
        self.assertEqual(self.store_rows(), [])


class TestBodyLimit(HttpCase):
    def test_a_body_over_the_limit_is_413_and_writes_nothing(self):
        self.server.max_body_bytes = 256
        status, body = self.put({"pad": "x" * 1024})
        self.assertEqual(status, 413, body)
        self.assertEqual(body["code"], "too-large")
        self.assertIn("256", body["reason"])
        self.assertEqual(self.store_rows(), [])


class TestNotWriter(HttpCase):
    writer = False

    def test_a_non_writer_node_gets_403_and_writes_nothing(self):
        status, body = self.put({"a": 1})
        self.assertEqual(status, 403, body)
        self.assertEqual(body["code"], "not-writer-node")
        self.assertIn("crontab", body["reason"])
        self.assertEqual(self.store_rows(), [])
        self.assertIs(self.call("GET", "/health", token=None)[1]["writer"], False)


class TestConcurrency(HttpCase):
    def test_two_concurrent_puts_both_land(self):
        """The store's own flock serialises them: neither patch is lost."""
        results: list[tuple[int, dict]] = []
        barrier = threading.Barrier(2)

        def one(patch: dict) -> None:
            barrier.wait()
            results.append(self.put(patch))

        threads = [threading.Thread(target=one, args=({"a": 1},)),
                   threading.Thread(target=one, args=({"b": 2},))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual([r[0] for r in results], [200, 200], results)
        (row,) = self.store_rows()
        self.assertEqual(row["recipe"], {"a": 1, "b": 2})
        self.assertEqual(len(row["history"]), 2)


class TestTokenInit(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="model-kb-http-token-"))
        self.mod = _load_server_module()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_init_token_creates_it_0600_and_never_rewrites_it(self):
        path = self.tmp / "nested" / "http-token"
        first, created = self.mod.ensure_token(path)
        self.assertTrue(created)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(path.read_text().strip(), first)
        second, created_again = self.mod.ensure_token(path)
        self.assertFalse(created_again)
        self.assertEqual(first, second)

    def test_init_token_flag_prints_the_path_and_the_token(self):
        path = self.tmp / "http-token"
        self.assertEqual(self.mod.main(["--init-token", "--token-file", str(path), "--root", str(self.tmp)]), 0)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)


class TestOneWriter(unittest.TestCase):
    """The MCP tool and the CLI still call recipe_store.put directly — the HTTP
    server is an addition, never a dependency they now need."""

    def test_neither_the_mcp_server_nor_the_cli_imports_the_http_api(self):
        for name in ("model-kb-mcp.py", "model_kb.py", "recipe_store.py"):
            with self.subTest(entry_point=name):
                src = (ENGINE / name).read_text()
                self.assertNotIn("model-kb-http", src)
                self.assertNotIn("modelkb_client", src)

    def test_the_http_server_never_writes_the_store_itself(self):
        """It imports recipe_store and calls put/get only — no second writer."""
        import ast

        src = (ENGINE / "model-kb-http.py").read_text()
        self.assertIn("import recipe_store", src)
        self.assertIn("recipe_store.put(", src)
        self.assertIn("recipe_store.get(", src)
        # Code only: the module docstring names recipes.jsonl to explain the store.
        code = ast.parse(src)
        skip: set[int] = set()
        for node in ast.walk(code):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc is not None and node.body and isinstance(node.body[0], ast.Expr):
                    d = node.body[0]
                    skip.update(range(d.lineno, d.end_lineno + 1))
        body = "\n".join(l.split("#")[0] for n, l in enumerate(src.splitlines(), 1) if n not in skip)
        for bypass in ("recipes.jsonl", "write_text(", "os.replace(", "_write_store"):
            with self.subTest(bypass=bypass):
                self.assertNotIn(bypass, body)

    def test_the_mcp_tool_writes_through_recipe_store_put(self):
        src = (ENGINE / "model-kb-mcp.py").read_text()
        start = src.index("def model_recipe_put(")
        body = src[start:src.index("if __name__", start)]
        self.assertIn("recipe_store.put(", body)
        self.assertIn("recipe_store.LockHeld", body)
        self.assertIn("recipe_store.NotWriterNode", body)


class TestDefaultPort(unittest.TestCase):
    def test_the_client_is_the_one_place_the_default_port_lives(self):
        self.assertIsInstance(kb.DEFAULT_PORT, int)
        self.assertGreater(kb.DEFAULT_PORT, 0)
        self.assertIn(str(kb.DEFAULT_PORT), (ENGINE / "model-kb-http.py").read_text())


if __name__ == "__main__":
    unittest.main()
