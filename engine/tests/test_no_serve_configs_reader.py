"""Nothing in model-kb reads or writes the retired serve-config file (profile-cascade W2.8, store G).

Serve recipes come from the recipe store: recipes.jsonl, written only by
recipe_store.put / MCP model_recipe_put. A (model, engine) record holds its
named serve configs under recipe.configs, and each one is a serve_recipe row.
The hand-edited file under ~/.r1o is gone, so:

  * no Python source names it, and the two scripts that read and rewrote it are deleted;
  * a full generation and a research-only generation both pass in a HOME that
    never had the file, and neither creates it;
  * a leftover copy is neither read nor rewritten;
  * an empty store fails closed: the previous generation, with its serve
    recipes, is kept (so this code deployed before the store is seeded cannot
    drop a recipe).

Every pipeline run is against a temp engine copy under a temp HOME (KbCase).

    python3 -I -m unittest discover -s engine/tests -v
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

from test_recipe_put import KbCase, _jsonl, _read_jsonl

ENGINE = Path(__file__).resolve().parents[1]
REPO = ENGINE.parent

# Spelled apart so this file passes the scan it runs.
LEGACY = "serve-configs" + ".json"
RETIRED = ("ingest_" + "serve_configs", "fill_" + "recipe_paths")

FX_CONFIGS = {
    ("Fx-Model-4bit", "fx_engine"): {"fx-model-1node": {
        "model": {"name": "Fx-Model-4bit", "hf_id": "fixture/Fx-Model"},
        "serve": {"engine": "fx_engine", "backend": "single", "decode_concurrency": 2},
        "sampling": {"temperature": 0.6},
    }},
    ("Fx-Gone-4bit", "fx_gone"): {"fx-gone-1node": {
        "model": {"name": "Fx-Gone-4bit", "hf_id": "fixture/Fx-Gone"},
        "serve": {"engine": "fx_gone", "backend": "single"},
    }},
}


def _sources() -> list[Path]:
    return sorted(p for p in REPO.rglob("*.py") if "__pycache__" not in p.parts)


class TestNoReaderInSource(unittest.TestCase):
    def test_sources_found(self):
        self.assertIn(ENGINE / "pipeline.py", _sources())

    def test_no_python_source_names_the_file(self):
        hits = [f"{p.relative_to(REPO)}:{n}: {line.strip()[:120]}"
                for p in _sources()
                for n, line in enumerate(p.read_text().splitlines(), 1) if LEGACY in line]
        self.assertEqual(hits, [], f"{LEGACY} is retired; these lines still name it")

    def test_the_two_scripts_that_read_and_rewrote_it_are_deleted(self):
        for name in RETIRED:
            self.assertFalse((ENGINE / f"{name}.py").exists(), f"engine/{name}.py still exists")
        pat = re.compile(r"\b(?:" + "|".join(RETIRED) + r")\b")
        hits = [f"{p.relative_to(REPO)}:{n}: {line.strip()[:120]}"
                for p in _sources() if p != Path(__file__).resolve()
                for n, line in enumerate(p.read_text().splitlines(), 1) if pat.search(line)]
        self.assertEqual(hits, [], "a retired script is still imported, run or named")

    def test_recipe_gap_sends_the_fix_to_the_recipe_writer(self):
        src = (ENGINE / "recipe_gap.py").read_text()
        self.assertIn("model_recipe_put", src)
        self.assertNotIn("/serve-config", src)


class TestGenerationsWithoutTheFile(KbCase):
    def legacy(self) -> Path:
        return self.home / ".r1o" / LEGACY

    def seed_store(self) -> None:
        for (model, engine), configs in FX_CONFIGS.items():
            self.rs.put(self.kb, model=model, engine=engine, source="research", patch={"configs": configs})

    def recipes(self) -> dict[str, dict]:
        return {r["recipe_id"]: r for r in _read_jsonl(self.kb / "records.jsonl") if r.get("kind") == "serve_recipe"}

    def test_full_generation_builds_serve_recipes_from_the_store(self):
        self.assertFalse(self.legacy().exists())
        self.seed_store()
        self.seed_research()
        _jsonl(self.kb / "records.jsonl", [])

        p = self.run_pipeline("--skip-network", "--skip-verify")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        got = self.recipes()
        self.assertEqual(sorted(got), ["fx-gone-1node", "fx-model-1node"])
        row = got["fx-model-1node"]
        self.assertEqual(row["id"], "recipe:fx-model-1node")
        self.assertEqual(row["artifact"]["hf_id"], "fixture/Fx-Model")
        self.assertEqual(row["serve"]["decode_concurrency"], 2)
        self.assertEqual(row["sampling"], {"temperature": 0.6})
        self.assertIs(row["servable"], True)
        self.assertFalse(self.legacy().exists(), "a generation must not create the retired file")

    def test_a_user_patch_to_one_key_of_a_config_overrides_the_researched_value(self):
        self.seed_store()
        self.rs.put(self.kb, model="Fx-Model-4bit", engine="fx_engine", source="user",
                    patch={"configs": {"fx-model-1node": {"serve": {"decode_concurrency": 4}}}})
        self.seed_research()
        _jsonl(self.kb / "records.jsonl", [])

        p = self.run_pipeline("--skip-network", "--skip-verify")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        serve = self.recipes()["fx-model-1node"]["serve"]
        self.assertEqual(serve["decode_concurrency"], 4)
        self.assertEqual(serve["engine"], "fx_engine")

    def test_research_only_generation_verifies_without_the_file(self):
        self.seed_store()
        self.seed_research()
        self.seed_hf_enrichments()
        _jsonl(self.kb / "records.jsonl", [])
        first = self.run_pipeline("--skip-network", "--skip-verify")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        with (self.kb / "records.jsonl").open("a") as f:
            f.write(json.dumps({"id": "cloud:fx", "kind": "cloud_context", "search_text": "fx cloud pack"}) + "\n")

        p = self.run_pipeline("--research-only")  # verify.py runs inside and must pass
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("verify OK", p.stdout)
        self.assertIn("serve round trip: configs=2", p.stdout)
        self.assertEqual(sorted(self.recipes()), ["fx-gone-1node", "fx-model-1node"])
        self.assertFalse(self.legacy().exists())

    def test_a_leftover_file_is_neither_read_nor_rewritten(self):
        self.seed_store()
        self.seed_research()
        _jsonl(self.kb / "records.jsonl", [])
        self.legacy().write_text(json.dumps({"configs": [{
            "name": "fx-only-in-the-file",
            "model": {"name": "Fx-File-4bit"},
            "serve": {"engine": "fx_engine", "backend": "single"},
        }]}))
        before = (self.legacy().read_bytes(), self.legacy().stat().st_mtime_ns)

        p = self.run_pipeline("--skip-network", "--skip-verify")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertNotIn("fx-only-in-the-file", self.recipes())
        self.assertEqual((self.legacy().read_bytes(), self.legacy().stat().st_mtime_ns), before)

    def test_an_empty_store_fails_closed_and_keeps_the_previous_recipes(self):
        self.seed_research()
        previous = [
            {"id": "recipe:fx-kept-1node", "kind": "serve_recipe", "recipe_id": "fx-kept-1node",
             "artifact": {"name": "Fx-Kept-4bit"}, "serve": {"engine": "fx_engine", "backend": "single"},
             "search_text": "fx kept"},
            {"id": "cloud:fx", "kind": "cloud_context", "search_text": "fx cloud pack"},
        ]
        _jsonl(self.kb / "records.jsonl", previous)
        before = (self.kb / "records.jsonl").read_bytes()
        self.assertFalse((self.kb / "recipes.jsonl").exists())

        p = self.run_pipeline("--skip-network", "--skip-verify")
        self.assertNotEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("holds no serve configs", p.stdout + p.stderr)
        self.assertIn("ROLLED BACK", p.stdout)
        self.assertEqual((self.kb / "records.jsonl").read_bytes(), before)
        self.assertEqual(sorted(self.recipes()), ["fx-kept-1node"])

    def test_recipe_notes_come_from_the_records(self):
        self.seed_store()
        self.seed_research()
        _jsonl(self.kb / "records.jsonl", [])
        p = self.run_pipeline("--skip-network", "--skip-verify")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        sys.modules.pop("export_obsidian", None)
        import export_obsidian  # noqa: PLC0415 — the temp copy

        note = export_obsidian.recipe_note(self.recipes()["fx-model-1node"])
        self.assertIn("# fx-model-1node", note)
        self.assertIn("recipe_id: fx-model-1node", note)
        self.assertIn("fixture/Fx-Model", note)


if __name__ == "__main__":
    unittest.main()
