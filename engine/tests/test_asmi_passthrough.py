"""asmi /serve/load is the contract: a serve config's asmi fields reach the
recipe and the load unchanged (MA, 2026-10-31).

Fixture configs only, in a temp recipe store — the live store and records.jsonl
are never read or written.

    python3 -I -m unittest discover -s engine/tests -v
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]

# The apple-smi source, read (never written) to catch drift between asmi's
# LoadRequest struct and lib/asmi_load.py. Skipped when it is not on disk.
TYPES_RS = Path(os.environ.get(
    "ASMI_TYPES_RS",
    Path.home() / "Projects" / "Personal" / "apple-smi" / "crates" / "cluster-monitor" / "src" / "types.rs",
))


def _fresh_imports():
    for mod in [m for m in list(sys.modules)
                if m in ("serve_recipes", "recipe_store", "verify", "lib") or m.startswith("lib.")]:
        del sys.modules[mod]
    if str(ENGINE) in sys.path:
        sys.path.remove(str(ENGINE))
    sys.path.insert(0, str(ENGINE))
    import serve_recipes as sr  # noqa: PLC0415
    import verify  # noqa: PLC0415
    from lib import asmi_load  # noqa: PLC0415

    return sr, verify, asmi_load


sr, verify, asmi_load = _fresh_imports()

# One valid value per asmi field a serve section owns, by JSON type.
_SAMPLE = {
    "string": "fx-value",
    "u32": 7,
    "u64": 34359738368,
    "f64": 3.5,
    "bool": True,
    "string_list": ["--fx-flag", "fx value with spaces"],
    "string_map": {"FX_KEY": "fx"},
}


def _every_asmi_field() -> dict:
    serve = {
        k: _SAMPLE[kind]
        for k, kind in asmi_load.LOAD_REQUEST_FIELDS.items()
        if k not in asmi_load.NOT_FROM_SERVE
    }
    serve.update(engine="fx_engine", backend="jaccl", hostfile="~/.r1o/hostfiles/fx.json")
    return serve


def _cfg(serve: dict, name: str = "fx-recipe") -> dict:
    return {"name": name, "model": {"name": "Fx-Model-4bit", "hf_id": "fixture/Fx-Model"}, "serve": serve}


def _store(root: Path, cfgs: list[dict]) -> None:
    """A recipe store holding `cfgs` the way the store-G migration files them: one
    research record per (model.name, serve.engine), each config under its name."""
    records: dict[tuple[str, str], dict] = {}
    for c in cfgs:
        key = (c["model"]["name"], (c.get("serve") or {}).get("engine") or "fx_engine")
        rec = records.setdefault(key, {"model": key[0], "engine": key[1], "source": "research",
                                       "recipe": {"configs": {}}, "history": []})
        rec["recipe"]["configs"][c["name"]] = {k: v for k, v in c.items() if k != "name"}
    (root / "recipes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records.values()))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestContractMirror(unittest.TestCase):
    @unittest.skipUnless(TYPES_RS.exists(), f"asmi source not on disk ({TYPES_RS})")
    def test_mirror_matches_asmi_load_request_struct(self):
        src = TYPES_RS.read_text()
        m = re.search(r"pub struct LoadRequest \{(.*?)\n\}", src, re.S)
        self.assertIsNotNone(m, "LoadRequest struct not found in types.rs")
        rust = dict(re.findall(r"^\s*pub (\w+):\s*(.+?),\s*$", m.group(1), re.M))

        def kind(t: str) -> str:
            t = t.strip()
            inner = re.fullmatch(r"Option<(.+)>", t)
            t = inner.group(1) if inner else t
            if t == "String":
                return "string"
            if t in ("u32", "u64", "f64", "bool"):
                return t
            if t == "Vec<String>":
                return "string_list"
            if re.fullmatch(r"(std::collections::)?HashMap<String,\s*String>", t):
                return "string_map"
            return f"unmapped:{t}"

        self.assertEqual({k: kind(t) for k, t in rust.items()}, asmi_load.LOAD_REQUEST_FIELDS)


class TestIngestCarriesServe(unittest.TestCase):
    def test_every_asmi_field_reaches_the_recipe_and_the_body_unchanged(self):
        serve = _every_asmi_field()
        rec = sr.recipe_from_config(_cfg(serve))
        self.assertTrue(rec["loadable"], rec["problems"])
        self.assertEqual(rec["problems"], [])
        for key, value in serve.items():
            self.assertEqual(rec["serve"][key], value, key)

        plan = sr.materialize_load(rec)
        body = plan["asmi_body"]
        for key, value in serve.items():
            self.assertEqual(body[key], value, key)
        self.assertEqual(body["model_path"], "~/Models/Fx-Model-4bit")
        self.assertEqual(set(body) - set(asmi_load.LOAD_REQUEST_FIELDS), set())

    def test_extra_args_list_passes_through_as_given(self):
        rec = sr.recipe_from_config(_cfg({"engine": "fx_engine", "backend": "single",
                                           "extra_args": ["--chat-template-args", '{"a": 1}']}))
        self.assertEqual(rec["serve"]["extra_args"], ["--chat-template-args", '{"a": 1}'])
        req = sr.recipe_to_serve_request(rec)
        self.assertEqual(req["extra_args"], ["--chat-template-args", '{"a": 1}'])

    def test_config_metadata_stays_on_the_record_but_out_of_the_asmi_body(self):
        serve = {"engine": "fx_engine", "backend": "single", "port": 19999, "hardware": "fx-chip",
                 "os": "fx-os", "nodes": ["fx-a"], "host": "127.0.0.1", "python": "~/fx/bin/python"}
        rec = sr.recipe_from_config(_cfg(serve))
        for key in ("hardware", "os", "nodes", "host", "python", "port"):
            self.assertEqual(rec["serve"][key], serve[key])
        plan = sr.materialize_load(rec)
        for key in ("hardware", "os", "nodes", "host", "python", "port"):
            self.assertNotIn(key, plan["asmi_body"])
        self.assertEqual(plan["port"], 19999)
        self.assertEqual(plan["not_sent"], ["hardware", "host", "nodes", "os", "port", "python"])

    def test_serve_request_posts_only_asmi_fields(self):
        rec = sr.recipe_from_config(_cfg({"engine": "fx_engine", "backend": "single", "port": 19999}))
        rec["accelerators"] = [{"type": "fx", "enabled": True, "draft_model": "~/Models/fx-draft"}]
        req = sr.recipe_to_serve_request(rec)
        self.assertEqual(req["port"], 19999)
        self.assertEqual(req["draft_model"], "~/Models/fx-draft")
        body = asmi_load.asmi_body_of(req)
        self.assertNotIn("port", body)
        self.assertNotIn("recipe_id", body)
        self.assertNotIn("accelerators", body)


class TestFailClosed(unittest.TestCase):
    def test_extra_args_string_is_dropped_and_named_never_split(self):
        rec = sr.recipe_from_config(_cfg({"engine": "fx_engine", "backend": "single",
                                           "extra_args": "--fx-flag 1"}))
        self.assertNotIn("extra_args", rec["serve"])
        self.assertFalse(rec["loadable"])
        self.assertTrue(any("serve.extra_args dropped" in p and "'--fx-flag 1'" in p for p in rec["problems"]),
                        rec["problems"])
        with self.assertRaisesRegex(ValueError, "serve.extra_args dropped"):
            sr.recipe_to_serve_request(rec)

    def test_extra_args_with_a_non_string_token_is_dropped(self):
        rec = sr.recipe_from_config(_cfg({"engine": "fx_engine", "backend": "single",
                                           "extra_args": ["--fx-n", 3]}))
        self.assertNotIn("extra_args", rec["serve"])
        self.assertFalse(rec["loadable"])
        self.assertIn("serve.extra_args dropped", rec["problems"][0])

    def test_no_engine_is_not_loadable_and_never_defaulted(self):
        rec = sr.recipe_from_config(_cfg({"backend": "single", "port": 19999}))
        self.assertIsNone(rec["serve"]["engine"])
        self.assertFalse(rec["loadable"])
        self.assertIn("serve.engine missing", rec["problems"][0])
        plan = sr.materialize_load(rec)
        self.assertNotIn("engine", plan["asmi_body"])
        self.assertFalse(plan["loadable"])
        self.assertNotIn("mlx_lm", json.dumps(rec) + json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "serve.engine missing"):
            sr.recipe_to_serve_request(rec)

    def test_a_value_asmi_would_refuse_is_named_not_sent(self):
        rec = sr.recipe_from_config(_cfg({"engine": "fx_engine", "backend": "single",
                                           "prompt_cache_size": "fx-words", "decode_concurrency": True}))
        self.assertFalse(rec["loadable"])
        joined = " ".join(rec["problems"])
        self.assertIn("serve.prompt_cache_size is 'fx-words'", joined)
        self.assertIn("serve.decode_concurrency is True", joined)
        plan = sr.materialize_load(rec)
        self.assertNotIn("prompt_cache_size", plan["asmi_body"])
        self.assertNotIn("decode_concurrency", plan["asmi_body"])

    def test_caller_owned_fields_are_not_taken_from_serve(self):
        rec = sr.recipe_from_config(_cfg({"engine": "fx_engine", "backend": "single",
                                           "dry_run": True, "model_path": "/fx/elsewhere"}))
        plan = sr.materialize_load(rec)
        self.assertNotIn("dry_run", plan["asmi_body"])
        self.assertEqual(plan["asmi_body"]["model_path"], "~/Models/Fx-Model-4bit")
        self.assertTrue(any("serve.dry_run is not sent" in w for w in plan["warnings"]))

    def test_backend_outside_single_jaccl_is_recorded_as_single_and_said(self):
        rec = sr.recipe_from_config(_cfg({"engine": "fx_engine", "backend": "auto"}))
        self.assertEqual(rec["serve"]["backend"], "single")
        self.assertTrue(rec["loadable"])
        self.assertIn("serve.backend 'auto'", rec["warnings"][0])

    def test_recipe_hostfile_wins_and_jaccl_without_one_is_named(self):
        own = sr.recipe_from_config(_cfg({"engine": "fx_engine", "backend": "jaccl",
                                           "hostfile": "~/.r1o/hostfiles/fx-own.json"}))
        self.assertEqual(sr.materialize_load(own, hostfile="~/fallback.json")["asmi_body"]["hostfile"],
                         "~/.r1o/hostfiles/fx-own.json")
        bare = sr.recipe_from_config(_cfg({"engine": "fx_engine", "backend": "jaccl"}))
        self.assertTrue(bare["loadable"], "the hostfile is supplied at load time, not recorded")
        self.assertEqual(sr.materialize_load(bare, hostfile="~/fallback.json")["asmi_body"]["hostfile"],
                         "~/fallback.json")
        self.assertIn("jaccl requires explicit hostfile", sr.materialize_load(bare)["problems"][0])

    def test_a_tilde_hostfile_is_sent_as_saved_with_a_warning(self):
        # asmi checks the hostfile path as given (no ~ expansion) and falls back
        # to single when it is missing — say so, never rewrite the value.
        rec = sr.recipe_from_config(_cfg({"engine": "fx_engine", "backend": "jaccl",
                                           "hostfile": "~/.mlx/fx.json"}))
        plan = sr.materialize_load(rec)
        self.assertEqual(plan["asmi_body"]["hostfile"], "~/.mlx/fx.json")
        self.assertTrue(any("no ~ expansion" in w for w in plan["warnings"]), plan["warnings"])


class TestRecipesCli(unittest.TestCase):
    def test_an_unloadable_recipe_is_recorded_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as d:
            _store(Path(d), [
                _cfg({"engine": "fx_engine", "backend": "single"}, "fx-ok"),
                _cfg({"backend": "single", "extra_args": "--fx 1"}, "fx-bad"),
            ])
            out = Path(d) / "records.jsonl"
            p = subprocess.run([sys.executable, str(ENGINE / "serve_recipes.py"),
                                "--root", d, "--out", str(out)],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("NOT LOADABLE fx-bad", p.stdout)
            recs = {json.loads(l)["recipe_id"]: json.loads(l) for l in out.read_text().splitlines()}
            self.assertTrue(recs["fx-ok"]["loadable"])
            self.assertFalse(recs["fx-bad"]["loadable"])

    def test_model_kb_get_shows_the_asmi_body_and_the_port_apart(self):
        with tempfile.TemporaryDirectory() as d:
            rec = sr.recipe_from_config(_cfg({"engine": "fx_engine", "backend": "single", "port": 19999,
                                               "os": "fx-os", "extra_args": ["--fx-flag", "1"]}))
            records = Path(d) / "records.jsonl"
            records.write_text(json.dumps(rec) + "\n")
            p = subprocess.run([sys.executable, str(ENGINE / "model_kb.py"), "--records", str(records),
                                "get", "fx-recipe"], capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            out = json.loads(p.stdout)
            self.assertEqual(out["asmi_body"]["extra_args"], ["--fx-flag", "1"])
            self.assertNotIn("port", out["asmi_body"])
            self.assertEqual(out["port"], 19999)
            self.assertEqual(out["serve_keys_not_sent"], ["os", "port"])


class TestRoundTrip(unittest.TestCase):
    def test_passthrough_ingest_round_trips_every_key(self):
        cfgs = [_cfg(_every_asmi_field(), "fx-all"),
                _cfg({"engine": "fx_engine", "backend": "single", "os": "fx-os", "nodes": ["fx-a"]}, "fx-meta")]
        recs = [sr.recipe_from_config(c) for c in cfgs]
        rt = verify.serve_round_trip(recs, cfgs)
        self.assertEqual((rt["dropped"], rt["changed"], rt["unmatched"], rt["unexplained"]), ([], [], [], []))

    def test_named_changes_are_explained_and_a_silent_drop_is_not(self):
        cfgs = [_cfg({"engine": "fx_engine", "backend": "auto", "extra_args": "--fx 1"}, "fx-named"),
                _cfg({"engine": "fx_engine", "backend": "single", "ctx_size": 4096}, "fx-lost"),
                _cfg({"engine": "fx_engine"}, "fx-orphan")]
        recs = [sr.recipe_from_config(c) for c in cfgs[:2]]
        del recs[1]["serve"]["ctx_size"]  # an ingest that forgot a field
        rt = verify.serve_round_trip(recs, cfgs)
        self.assertEqual({(r["recipe"], r["key"]) for r in rt["dropped"]},
                         {("fx-named", "extra_args"), ("fx-lost", "ctx_size")})
        self.assertEqual({(r["recipe"], r["key"]) for r in rt["changed"]}, {("fx-named", "backend")})
        self.assertEqual(rt["unmatched"], ["fx-orphan"])
        self.assertEqual([(r["recipe"], r["key"]) for r in rt["unexplained"]], [("fx-lost", "ctx_size")])

    def test_an_unexplained_drop_with_asmi_unreachable_is_a_named_failure(self):
        cfgs = [_cfg({"engine": "fx_engine", "backend": "single", "ctx_size": 4096}, "fx-lost")]
        rec = sr.recipe_from_config(cfgs[0])
        del rec["serve"]["ctx_size"]
        with tempfile.TemporaryDirectory() as d:
            _store(Path(d), cfgs)
            saved = os.environ.get("ASMI_URL")
            os.environ["ASMI_URL"] = f"http://127.0.0.1:{_free_port()}"
            try:
                from io import StringIO
                from contextlib import redirect_stdout
                buf = StringIO()
                with redirect_stdout(buf):
                    rc = verify.check_round_trip([rec], Path(d))
            finally:
                if saved is None:
                    os.environ.pop("ASMI_URL", None)
                else:
                    os.environ["ASMI_URL"] = saved
        self.assertEqual(rc, 1)
        self.assertIn("/serve/engines unreachable", buf.getvalue())

    def test_classification_reads_the_live_engine_field_set(self):
        self.assertEqual(verify.classify_key("ctx_size", {"ctx_size"}), "engine-mapped (a lost knob)")
        self.assertEqual(verify.classify_key("extra_args", set()), "core asmi LoadRequest field")
        self.assertEqual(verify.classify_key("os", {"ctx_size"}), "config-only (not an asmi field)")

    def test_a_store_with_no_serve_configs_is_a_named_failure(self):
        from io import StringIO
        from contextlib import redirect_stdout
        buf = StringIO()
        with tempfile.TemporaryDirectory() as d, redirect_stdout(buf):
            (Path(d) / "recipes.jsonl").write_text("")
            rc = verify.check_round_trip([], Path(d))
        self.assertEqual(rc, 1)
        self.assertIn("holds no serve configs", buf.getvalue())

    def test_no_store_at_all_skips_the_round_trip(self):
        from io import StringIO
        from contextlib import redirect_stdout
        buf = StringIO()
        with tempfile.TemporaryDirectory() as d, redirect_stdout(buf):
            rc = verify.check_round_trip([], Path(d))
        self.assertEqual(rc, 0)
        self.assertIn("no recipe store", buf.getvalue())

    def test_an_unreadable_store_is_a_named_failure(self):
        from io import StringIO
        from contextlib import redirect_stdout
        buf = StringIO()
        with tempfile.TemporaryDirectory() as d, redirect_stdout(buf):
            (Path(d) / "recipes.jsonl").write_text("{not json\n")
            rc = verify.check_round_trip([], Path(d))
        self.assertEqual(rc, 1)
        self.assertIn("recipe store unreadable", buf.getvalue())
        self.assertIn("recipes.jsonl:1", buf.getvalue())


class TestStoreConfigs(unittest.TestCase):
    """serve_recipes.store_configs: the store's named configs, as the recipes' upstream."""

    def _rows(self, d: str, rows: list[dict]) -> Path:
        (Path(d) / "recipes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return Path(d)

    def test_a_migrated_config_gives_the_same_recipe_as_the_config_itself(self):
        cfgs = [_cfg(_every_asmi_field(), "fx-all"),
                _cfg({"engine": "fx_other", "backend": "single", "port": 19999}, "fx-other")]
        with tempfile.TemporaryDirectory() as d:
            _store(Path(d), cfgs)
            got = {c["name"]: c for c in sr.store_configs(Path(d))}
        self.assertEqual(sorted(got), ["fx-all", "fx-other"])
        for c in cfgs:
            self.assertEqual(sr.recipe_from_config(got[c["name"]]), sr.recipe_from_config(c))

    def test_sources_merge_research_then_benchmark_then_user(self):
        base = {"model": "Fx-Model-4bit", "engine": "fx_engine", "history": []}
        with tempfile.TemporaryDirectory() as d:
            root = self._rows(d, [
                {**base, "source": "user", "recipe": {"configs": {"fx-a": {"serve": {"ctx_size": 8192}}}}},
                {**base, "source": "research", "recipe": {"configs": {"fx-a": {
                    "serve": {"backend": "single", "ctx_size": 4096, "port": 19999}}}}},
            ])
            (cfg,) = sr.store_configs(root)
        self.assertEqual(cfg["serve"], {"backend": "single", "ctx_size": 8192, "port": 19999, "engine": "fx_engine"})

    def test_a_config_takes_the_model_and_engine_its_record_is_filed_under(self):
        with tempfile.TemporaryDirectory() as d:
            root = self._rows(d, [{"model": "Fx-Model-4bit", "engine": "fx_engine", "source": "research",
                                   "recipe": {"configs": {"fx-bare": {"serve": {"backend": "single"}}}},
                                   "history": []}])
            (cfg,) = sr.store_configs(root)
        rec = sr.recipe_from_config(cfg)
        self.assertEqual((rec["artifact"]["name"], rec["serve"]["engine"]), ("Fx-Model-4bit", "fx_engine"))
        self.assertTrue(rec["loadable"], rec["problems"])

    def test_one_name_in_two_records_is_refused(self):
        row = {"source": "research", "recipe": {"configs": {"fx-dup": {"serve": {"backend": "single"}}}}, "history": []}
        with tempfile.TemporaryDirectory() as d:
            root = self._rows(d, [{**row, "model": "Fx-A", "engine": "fx_engine"},
                                  {**row, "model": "Fx-B", "engine": "fx_engine"}])
            with self.assertRaises(ValueError) as cm:
                sr.store_configs(root)
        self.assertIn("fx-dup", str(cm.exception))
        self.assertIn("Fx-A @ fx_engine", str(cm.exception))
        self.assertIn("Fx-B @ fx_engine", str(cm.exception))

    def test_configs_that_is_not_an_object_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = self._rows(d, [{"model": "Fx-A", "engine": "fx_engine", "source": "research",
                                   "recipe": {"configs": ["fx-a"]}, "history": []}])
            with self.assertRaises(ValueError) as cm:
                sr.store_configs(root)
        self.assertIn("recipe.configs must be an object", str(cm.exception))

    def test_an_empty_store_writes_nothing_and_says_why(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "records.jsonl"
            out.write_text("previous\n")
            p = subprocess.run([sys.executable, str(ENGINE / "serve_recipes.py"), "--root", d, "--out", str(out)],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
            self.assertIn("holds no serve configs", p.stdout)
            self.assertIn("model_recipe_put", p.stdout)
            self.assertEqual(out.read_text(), "previous\n")


if __name__ == "__main__":
    unittest.main()
