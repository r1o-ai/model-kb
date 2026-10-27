"""Recipe writer, fold and snapshot: the one writer of hand-made recipe data (E-Q2).

Every test runs against a temp copy of the engine. HOME, ASMI_URL, the embed
server and the tailnet (a fake `tailscale` on PATH) all point at local stubs, so
the live corpus, the live cluster and the network are never touched.

    python3 -I -m unittest discover -s engine/tests -v
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
DIMS = 8

# Fixture cluster: two online nodes with overlapping engine sets, one offline node.
# These names exist only inside the stubs below; they are fixture data, not inventory.
NODE_A, NODE_B, NODE_OFF = "fx-node-a", "fx-node-b", "fx-node-c"
ENGINES_A = ("fx_engine", "fx_a_only")
ENGINES_B = ("fx_engine", "fx_b_only")


def _handler(engines: tuple[str, ...], cluster: list[dict] | None):
    class Stub(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code: int, body) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802
            if self.path == "/serve/engines":
                self._json(200, {"engines": [{"id": e, "source": "builtin"} for e in engines]})
            elif self.path == "/cluster" and cluster is not None:
                self._json(200, cluster)
            elif self.path == "/nodes" and cluster is not None:
                self._json(200, {"nodes": [n["hostname"] for n in cluster], "total": len(cluster)})
            elif self.path == "/health":
                self._json(200, {"embedding": {"model": "stub-embed", "dimensions": DIMS}})
            else:
                self._json(404, {})

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path != "/v1/embeddings":
                return self._json(404, {})
            data = []
            for text in body.get("input") or []:
                h = hashlib.sha256(text.encode()).digest()
                data.append({"embedding": [b / 255.0 + 0.01 for b in h[:DIMS]]})
            self._json(200, {"data": data})

    return Stub


class _V6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def _serve(server: ThreadingHTTPServer) -> ThreadingHTTPServer:
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _start_cluster() -> tuple[list[ThreadingHTTPServer], int]:
    """Node A on 127.0.0.1:P (also the local asmi + embed server), node B on [::1]:P."""
    cluster = [
        {"hostname": NODE_A, "online": True},
        {"hostname": NODE_B, "online": True},
        {"hostname": NODE_OFF, "online": False},
    ]
    for _ in range(20):
        a = ThreadingHTTPServer(("127.0.0.1", 0), _handler(ENGINES_A, cluster))
        port = a.server_address[1]
        try:
            b = _V6Server(("::1", port), _handler(ENGINES_B, None))
        except OSError:
            a.server_close()
            continue
        return [_serve(a), _serve(b)], port
    raise RuntimeError("could not bind the same port on 127.0.0.1 and ::1")


def _dead_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


class KbCase(unittest.TestCase):
    """A temp engine install on the writer node (its crontab runs pipeline.py)."""

    writer = True

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="model-kb-test-"))
        self.home = self.tmp / "home"
        self.kb = self.home / ".r1o" / "model-kb"
        shutil.copytree(
            ENGINE,
            self.kb,
            ignore=shutil.ignore_patterns(
                "tests", "__pycache__", "bm25_index", "*.jsonl", "*.bak*", ".ingest.lock",
                "recipes.snapshot.json", "research-corpus.glob",
            ),
        )
        self.servers, self.port = _start_cluster()
        self.url = f"http://127.0.0.1:{self.port}"
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self._fake_crontab()
        self._fake_tailscale()
        self.env = dict(
            os.environ,
            HOME=str(self.home),
            ASMI_URL=self.url,
            EMBED_URL=self.url,
            R1O_ROOT=str(self.tmp / "no-r1o"),
            PATH=f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
            MODEL_KB_LOCK_TIMEOUT_S="2",
        )
        self._saved_env = dict(os.environ)
        os.environ.update(self.env)
        sys.path.insert(0, str(self.kb))
        for mod in [m for m in list(sys.modules)
                    if m in ("recipe_store", "verify", "ingest_serve_configs") or m == "lib" or m.startswith("lib.")]:
            del sys.modules[mod]
        import recipe_store  # noqa: PLC0415 — the temp copy, after env is set

        self.rs = recipe_store

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._saved_env)
        sys.path.remove(str(self.kb))
        for s in self.servers:
            s.shutdown()
            s.server_close()
        for p in self.tmp.rglob("*"):
            if p.is_file() and not os.access(p, os.W_OK):
                p.chmod(0o644)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fake_crontab(self) -> None:
        """`crontab -l` → one entry whose script runs this KB's pipeline.py (or nothing)."""
        script = self.tmp / "cron" / "kb-ingest.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text('#!/bin/bash\ncd "$HOME/.r1o/model-kb" || exit 1\npython3 pipeline.py --research-only\n')
        line = f"0 */3 * * * /bin/bash {script} # r1o:kb-ingest\n" if self.writer else "# no jobs\n"
        tab = self.bin / "crontab"
        tab.write_text(f"#!/bin/sh\ncat <<'EOF'\n{line}EOF\n")
        tab.chmod(0o755)

    def _fake_tailscale(self) -> None:
        status = {
            "BackendState": "Running",
            "Self": {"HostName": NODE_A, "DNSName": f"{NODE_A}.fixture.", "TailscaleIPs": ["127.0.0.1"], "Online": True},
            "Peer": {"b": {"HostName": NODE_B, "DNSName": f"{NODE_B}.fixture.", "TailscaleIPs": ["::1"], "Online": True}},
        }
        ts = self.bin / "tailscale"
        ts.write_text(f"#!/bin/sh\ncat <<'EOF'\n{json.dumps(status)}\nEOF\n")
        ts.chmod(0o755)

    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(self.kb / "recipe_store.py"), *args],
            cwd=str(self.kb), env=self.env, capture_output=True, text=True, timeout=60,
        )

    def run_pipeline(self, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(self.kb / "pipeline.py"), *args],
            cwd=str(self.kb), env=env or self.env, capture_output=True, text=True, timeout=300,
        )

    # -- corpus fixtures -------------------------------------------------
    def seed_serve_configs(self) -> None:
        (self.home / ".r1o" / "serve-configs.json").write_text(json.dumps({"configs": [{
            "name": "fx-model-1node",
            "model": {"name": "Fx-Model-4bit", "hf_id": "fixture/Fx-Model"},
            "serve": {"engine": "fx_engine", "backend": "single", "decode_concurrency": 2},
            "sampling": {"temperature": 0.6},
        }, {
            "name": "fx-gone-1node",
            "model": {"name": "Fx-Gone-4bit", "hf_id": "fixture/Fx-Gone"},
            "serve": {"engine": "fx_gone", "backend": "single"},
        }]}))

    def seed_research(self) -> None:
        notes = self.tmp / "notes"
        notes.mkdir()
        (notes / "fx-serving.md").write_text(textwrap.dedent("""\
            ---
            tags: [research, inference]
            ---
            # Serving notes for a fixture engine

            Prefill step size matters for long prompts on fx_engine.
            """))
        (self.kb / "research-corpus.glob").write_text(f"{notes}/*.md\n")

    def seed_hf_enrichments(self) -> None:
        root = "fixture/Fx-Model"
        _jsonl(self.kb / "enrichments_hf_family.jsonl", [{
            "id": "hffamily:fx", "enrichment_id": "hffamily:fx", "kind": "hf_family",
            "title": "Fx family", "search_text": "fx family", "measurements": {"root_base": root},
        }])
        _jsonl(self.kb / "enrichments_hf_derivative.jsonl", [{
            "id": "hfderiv:fx", "enrichment_id": "hfderiv:fx", "kind": "hf_derivative",
            "title": "Fx derivatives", "search_text": "fx derivatives",
            "measurements": {"root_base": root, "derivatives": []},
        }])
        _jsonl(self.kb / "enrichments_hf.jsonl", [{
            "id": "hfcard:fx", "enrichment_id": "hfcard:fx", "kind": "hf_card",
            "title": "Fx card", "search_text": "fx card", "measurements": {},
        }])

    def seed_full_corpus(self) -> None:
        """records.jsonl holding every kind verify.py requires, plus recipe rows."""
        self.seed_serve_configs()
        self.seed_research()
        self.seed_hf_enrichments()
        import ingest_serve_configs as isc  # noqa: PLC0415 — the temp copy

        cfgs = json.loads((self.home / ".r1o" / "serve-configs.json").read_text())["configs"]
        cloud = {"id": "cloud:fx", "kind": "cloud_context", "search_text": "fx cloud pack"}
        _jsonl(self.kb / "records.jsonl", [isc.recipe_from_config(c) for c in cfgs] + [cloud])

    def generation_bytes(self) -> dict[str, bytes]:
        out = {}
        for rel in ("records.jsonl", "bm25_index/vectors.npy", "bm25_index/vectors.ids.json"):
            p = self.kb / rel
            out[rel] = p.read_bytes() if p.exists() else b""
        return out

    def aligned(self) -> list[dict]:
        recs = _read_jsonl(self.kb / "records.jsonl")
        ids = json.loads((self.kb / "bm25_index" / "vectors.ids.json").read_text())
        self.assertEqual(ids, [str(r.get("id") or r.get("recipe_id") or i) for i, r in enumerate(recs)])
        return recs


class TestPut(KbCase):
    def test_put_get_roundtrip_keeps_source_and_history(self):
        self.rs.put(self.kb, model="fixture/Fx-Model", engine="fx_engine", source="user",
                    patch={"sampling": {"temperature": 0.3}})
        self.rs.put(self.kb, model="fixture/Fx-Model", engine="fx_engine", source="user",
                    patch={"sampling": {"top_p": 0.9}})
        rec = self.rs.get(self.kb, model="fixture/Fx-Model", engine="fx_engine", source="user")
        self.assertEqual(rec["source"], "user")
        self.assertEqual(rec["recipe"], {"sampling": {"temperature": 0.3, "top_p": 0.9}})
        self.assertEqual([h["patch"] for h in rec["history"]],
                         [{"sampling": {"temperature": 0.3}}, {"sampling": {"top_p": 0.9}}])

    def test_sources_are_separate_records_for_one_model_engine(self):
        for src, t in (("research", 0.7), ("benchmark", 0.5), ("user", 0.2)):
            self.rs.put(self.kb, model="fixture/Fx-Model", engine="fx_engine", source=src,
                        patch={"sampling": {"temperature": t}})
        rows = _read_jsonl(self.kb / "recipes.jsonl")
        self.assertEqual(sorted(r["source"] for r in rows), ["benchmark", "research", "user"])

    def test_none_in_patch_deletes_the_key(self):
        self.rs.put(self.kb, model="m", engine="fx_engine", source="user", patch={"sampling": {"temperature": 0.3}})
        self.rs.put(self.kb, model="m", engine="fx_engine", source="user", patch={"sampling": {"temperature": None}})
        self.assertEqual(self.rs.get(self.kb, model="m", engine="fx_engine", source="user")["recipe"], {"sampling": {}})

    def test_unknown_source_is_rejected(self):
        with self.assertRaises(self.rs.RecipeError):
            self.rs.put(self.kb, model="m", engine="fx_engine", source="guess", patch={})
        self.assertFalse((self.kb / "recipes.jsonl").exists())

    def test_empty_model_or_engine_is_rejected(self):
        for model, engine in (("", "fx_engine"), ("m", ""), ("m", "has space")):
            with self.assertRaises(self.rs.RecipeError):
                self.rs.put(self.kb, model=model, engine=engine, source="user", patch={})

    def test_write_is_atomic_temp_plus_replace(self):
        self.rs.put(self.kb, model="m", engine="fx_engine", source="user", patch={"a": 1})
        leftovers = [p.name for p in self.kb.iterdir() if ".tmp" in p.name or p.name.startswith("recipes.jsonl.")]
        self.assertEqual(leftovers, [])


class TestLock(KbCase):
    def test_put_while_ingest_lock_held_returns_exit_locked(self):
        fh = open(self.kb / ".ingest.lock", "a+")
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write("4242 pipeline-under-test\n")
        fh.flush()
        try:
            with self.assertRaises(self.rs.LockHeld) as cm:
                self.rs.put(self.kb, model="m", engine="fx_engine", source="user", patch={"a": 1}, lock_timeout_s=0)
            self.assertIn("4242", str(cm.exception))
            p = self.run_cli("put", "--model", "m", "--engine", "fx_engine", "--source", "user",
                             "--patch", '{"a": 1}', "--lock-timeout", "0")
            self.assertEqual(p.returncode, self.rs.EXIT_LOCKED, p.stdout + p.stderr)
            self.assertIn("retry", (p.stdout + p.stderr).lower())
        finally:
            fh.close()
        self.assertFalse((self.kb / "recipes.jsonl").exists(), "a locked put must not write")


class TestNotWriter(KbCase):
    writer = False

    def test_put_refused_where_no_crontab_runs_pipeline(self):
        with self.assertRaises(self.rs.NotWriterNode):
            self.rs.put(self.kb, model="m", engine="fx_engine", source="user", patch={"a": 1})
        p = self.run_cli("put", "--model", "m", "--engine", "fx_engine", "--source", "user", "--patch", "{}")
        self.assertEqual(p.returncode, self.rs.EXIT_NOT_WRITER, p.stdout + p.stderr)
        self.assertFalse((self.kb / "recipes.jsonl").exists())


class TestSnapshot(KbCase):
    def test_snapshot_is_0444_generated_and_byte_stable(self):
        self.rs.put(self.kb, model="fixture/Fx-Model", engine="fx_engine", source="research",
                    patch={"sampling": {"temperature": 0.7, "top_p": 0.95}})
        self.rs.put(self.kb, model="fixture/Fx-Model", engine="fx_engine", source="user",
                    patch={"sampling": {"temperature": 0.2}})
        snap = self.kb / "recipes.snapshot.json"
        first = snap.read_bytes()
        self.assertEqual(stat.S_IMODE(snap.stat().st_mode), 0o444)
        self.assertFalse(os.access(snap, os.W_OK))
        self.rs.export_recipe_snapshot(self.kb)
        self.assertEqual(snap.read_bytes(), first, "same input must give the same bytes")
        self.assertEqual(stat.S_IMODE(snap.stat().st_mode), 0o444)

        d = json.loads(first)
        self.assertEqual(next(iter(d)), "GENERATED")
        self.assertIn("do not edit", d["GENERATED"])
        (entry,) = d["recipes"]
        self.assertEqual((entry["model"], entry["engine"]), ("fixture/Fx-Model", "fx_engine"))
        self.assertEqual(entry["effective"]["sampling"], {"temperature": 0.2, "top_p": 0.95})
        self.assertEqual(sorted(entry["sources"]), ["research", "user"])

    def test_export_cli_writes_the_snapshot(self):
        self.rs.put(self.kb, model="m", engine="fx_engine", source="user", patch={"a": 1})
        snap = self.kb / "recipes.snapshot.json"
        snap.chmod(0o644)
        snap.unlink()
        p = self.run_cli("export")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(stat.S_IMODE(snap.stat().st_mode), 0o444)


class TestLiveEngines(KbCase):
    def test_union_over_online_nodes_with_provenance(self):
        from lib.engines import live_engine_ids  # noqa: PLC0415

        got = live_engine_ids()
        self.assertEqual(got, {
            "fx_engine": [NODE_A, NODE_B],
            "fx_a_only": [NODE_A],
            "fx_b_only": [NODE_B],
        })

    def test_asmi_unreachable_fails_closed(self):
        from lib.engines import EngineCatalogUnavailable, live_engine_ids  # noqa: PLC0415

        os.environ["ASMI_URL"] = f"http://127.0.0.1:{_dead_port()}"
        with self.assertRaises(EngineCatalogUnavailable):
            live_engine_ids()


class TestGeneration(KbCase):
    """Real pipeline.py passes in the temp install (no network; stub embed, asmi, tailnet)."""

    def test_user_record_survives_a_full_generation(self):
        self.seed_serve_configs()
        self.seed_research()
        _jsonl(self.kb / "records.jsonl", [])
        first = self.run_pipeline("--skip-network", "--skip-fill-paths", "--skip-verify")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)

        self.rs.put(self.kb, model="fixture/Fx-Model", engine="fx_engine", source="user",
                    patch={"sampling": {"temperature": 0.2}})
        for _ in range(2):  # the recipes clobber runs each time; the fold re-adds, never duplicates
            p = self.run_pipeline("--skip-network", "--skip-fill-paths", "--skip-verify")
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertNotIn("ROLLED BACK", p.stdout)
            recs = self.aligned()
            authored = [r for r in recs if r.get("kind") == "authored_recipe"]
            self.assertEqual(len(authored), 1, [r.get("id") for r in recs])
            self.assertEqual(authored[0]["source"], "user")
            self.assertEqual(authored[0]["recipe"]["sampling"], {"temperature": 0.2})
            self.assertTrue(any(r.get("kind") == "serve_recipe" for r in recs), "the clobber step ran")
        self.assertEqual(self.rs.get(self.kb, model="fixture/Fx-Model", engine="fx_engine",
                                     source="user")["recipe"], {"sampling": {"temperature": 0.2}})

    def test_fold_computes_servable_and_verify_passes(self):
        self.seed_full_corpus()
        self.rs.put(self.kb, model="fixture/Fx-Model", engine="fx_engine", source="user",
                    patch={"sampling": {"temperature": 0.2}})
        self.rs.put(self.kb, model="fixture/Fx-Model", engine="fx_b_only", source="benchmark",
                    patch={"serve": {"decode_concurrency": 4}})

        p = self.run_pipeline("--research-only")  # verify.py runs inside and must pass
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("verify OK", p.stdout)
        recs = self.aligned()
        by_engine = {(r.get("kind"), (r.get("serve") or {}).get("engine") or r.get("engine")): r for r in recs}

        both = by_engine[("serve_recipe", "fx_engine")]
        self.assertIs(both["servable"], True)
        self.assertEqual(both["servable_on"], [NODE_A, NODE_B])

        one = by_engine[("authored_recipe", "fx_b_only")]
        self.assertIs(one["servable"], True)
        self.assertEqual(one["servable_on"], [NODE_B])

        gone = by_engine[("serve_recipe", "fx_gone")]
        self.assertIs(gone["servable"], False)
        self.assertEqual(gone["servable_on"], [])
        self.assertIn("fx_gone", gone["servable_reason"])

        snap = json.loads((self.kb / "recipes.snapshot.json").read_text())
        effective = {(e["model"], e["engine"]): e["effective"] for e in snap["recipes"]}
        self.assertEqual(effective[("fixture/Fx-Model", "fx_engine")]["sampling"], {"temperature": 0.2})

    def test_asmi_unreachable_aborts_the_fold_and_keeps_the_previous_generation(self):
        self.seed_full_corpus()
        ok = self.run_pipeline("--research-only")
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
        before = self.generation_bytes()

        self.rs.put(self.kb, model="fixture/Fx-Model", engine="fx_engine", source="user",
                    patch={"sampling": {"temperature": 0.1}})
        env = dict(self.env, ASMI_URL=f"http://127.0.0.1:{_dead_port()}")
        p = self.run_pipeline("--research-only", env=env)
        self.assertEqual(p.returncode, self.rs.EXIT_NO_ENGINES, p.stdout + p.stderr)
        self.assertIn("engine", (p.stdout + p.stderr).lower())
        self.assertIn("ROLLED BACK", p.stdout)
        self.assertEqual(self.generation_bytes(), before)


class TestVerify(KbCase):
    def test_verify_has_no_engine_registry_file_dependency(self):
        src = (self.kb / "verify.py").read_text()
        self.assertNotIn("engine-config", src)

    def test_recipe_without_servable_fails_verify(self):
        import verify  # noqa: PLC0415 — the temp copy

        recs = [{"id": "recipe:a", "kind": "serve_recipe", "recipe_id": "a", "serve": {"engine": "fx_engine"}}]
        self.assertEqual(verify.unfolded_recipes(recs), ["a"])
        recs[0].update(servable=False, servable_on=[], servable_reason="no online node declares engine fx_engine")
        self.assertEqual(verify.unfolded_recipes(recs), [])


if __name__ == "__main__":
    unittest.main()
