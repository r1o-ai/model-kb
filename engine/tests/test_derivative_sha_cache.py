"""Derivative audit reuses a cached row when the family listing already carries the commit sha.

The family walk lists every child with its `sha` (Hub `expand[]=sha`). A child whose listing
sha matches the cached row, audited against the same official commit, needs no per-repo call
and no rate-limit sleep. Everything here is fixture data and mocked HTTP: no network, and
nothing under ~/.r1o is read or written.

    python3 -I -m unittest discover -s engine/tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlsplit

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import ingest_derivative_audit as audit  # noqa: E402
import ingest_hf_family as family  # noqa: E402
from lib import hf_hub  # noqa: E402

# Fixture ids. They exist only in this file.
ROOT_ID = "fx-org/fx-base"
CHILD_ID = "fx-community/fx-base-4bit"
SHA_OLD = "a" * 40
SHA_NEW = "b" * 40
OFFICIAL_SHA = "c" * 40
PATH_KINDS = {"fx_gguf": "gguf-file", "fx_dir": "directory"}


def cached_row(sha: str = SHA_OLD, official_sha: str = OFFICIAL_SHA) -> dict:
    return {
        "id": CHILD_ID,
        "relation": "quantized",
        "downloads": 5,
        "access": "ok",
        "sha": sha,
        "official_sha": official_sha,
        "audit_version": audit.AUDIT_VERSION,
        "formats": ["safetensors"],
        "total_bytes": 1234,
        "library_name": "fx-lib",
        "config_status": "ok",
        "quant": {"status": "ok", "quant_method": "fx_q", "fields": {"bits": 4}},
        "quant_parity": {"parity": "match", "diverge_fields": []},
        "components": {"status": "no_safetensors_index"},
        "servability": {"formats": ["safetensors"]},
        "modification": {"status": "skipped_quantized_relation"},
        "cache": "sha_match",
    }


class AuditCalls:
    """Counts every network entry point audit_member can reach, plus sleeps."""

    def __init__(self, api_sha: str | None):
        self.api_sha = api_sha
        self.repo_api = 0
        self.fetches = 0
        self.sleeps = 0

    def fake_repo_api(self, repo):
        self.repo_api += 1
        return "ok", {"id": repo, "sha": self.api_sha, "siblings": [], "library_name": "fx-lib"}

    def fake_fetch(self, *a, **k):
        self.fetches += 1
        return "missing", None

    def fake_text(self, *a, **k):
        self.fetches += 1
        return "missing", 404, ""

    def fake_sleep(self, _s):
        self.sleeps += 1

    def run(self, member: dict, prev: dict | None, official_sha: str = OFFICIAL_SHA) -> dict:
        with mock.patch.object(audit, "repo_api", self.fake_repo_api), \
                mock.patch.object(audit, "fetch_config", self.fake_fetch), \
                mock.patch.object(audit, "fetch_index", self.fake_fetch), \
                mock.patch.object(audit, "get_text", self.fake_text), \
                mock.patch.object(audit, "get_json", side_effect=AssertionError("unexpected get_json")), \
                mock.patch.object(audit.time, "sleep", self.fake_sleep):
            return audit.audit_member(
                member,
                root=ROOT_ID,
                official_groups=None,
                official_quant={"status": "no_quantization_config"},
                official_towers={},
                path_kinds=PATH_KINDS,
                sleep_s=0.25,
                official_sha=official_sha,
                cached={CHILD_ID: prev} if prev else None,
            )


def member(sha: str | None, downloads: int = 99) -> dict:
    m = {"id": CHILD_ID, "relation": "quantized", "downloads": downloads}
    if sha is not None:
        m["sha"] = sha
    return m


class ListingShaCache(unittest.TestCase):
    def test_matching_listing_sha_skips_repo_api_and_sleep(self):
        calls = AuditCalls(api_sha=SHA_OLD)
        row = calls.run(member(SHA_OLD, downloads=99), cached_row())
        self.assertEqual(calls.repo_api, 0)
        self.assertEqual(calls.sleeps, 0)
        self.assertEqual(calls.fetches, 0)
        self.assertEqual(row["cache"], "sha_match_listing")
        self.assertEqual(row["downloads"], 99)
        self.assertEqual(row["sha"], SHA_OLD)
        self.assertEqual(row["components"], {"status": "no_safetensors_index"})
        # servability is recomputed from the live registry, not copied
        self.assertIn("incompatible_engines", row["servability"])

    def test_changed_listing_sha_fetches(self):
        calls = AuditCalls(api_sha=SHA_NEW)
        row = calls.run(member(SHA_NEW), cached_row(sha=SHA_OLD))
        self.assertEqual(calls.repo_api, 1)
        self.assertGreater(calls.sleeps, 0)
        self.assertNotIn(row.get("cache"), {"sha_match_listing", "sha_match"})
        self.assertEqual(row["sha"], SHA_NEW)

    def test_official_commit_moved_fetches(self):
        calls = AuditCalls(api_sha=SHA_OLD)
        row = calls.run(member(SHA_OLD), cached_row(), official_sha="d" * 40)
        self.assertEqual(calls.repo_api, 1)
        self.assertNotEqual(row.get("cache"), "sha_match_listing")

    def test_unreusable_cached_row_fetches(self):
        prev = cached_row()
        prev["audit_version"] = audit.AUDIT_VERSION - 1
        calls = AuditCalls(api_sha=SHA_OLD)
        row = calls.run(member(SHA_OLD), prev)
        self.assertEqual(calls.repo_api, 1)
        self.assertNotEqual(row.get("cache"), "sha_match_listing")

    def test_missing_listing_sha_keeps_old_path(self):
        calls = AuditCalls(api_sha=SHA_OLD)
        row = calls.run(member(None), cached_row())
        self.assertEqual(calls.repo_api, 1)
        self.assertEqual(calls.sleeps, 1)
        self.assertEqual(row["cache"], "sha_match")

    def test_uncached_member_with_listing_sha_fetches(self):
        calls = AuditCalls(api_sha=SHA_OLD)
        row = calls.run(member(SHA_OLD), None)
        self.assertEqual(calls.repo_api, 1)
        self.assertNotIn("cache", row)


class FamilyListingExpand(unittest.TestCase):
    LISTING_ROW = {
        "_id": "fx",
        "id": CHILD_ID,
        "downloads": 42,
        "likes": 3,
        "tags": ["mlx", "4-bit", f"base_model:quantized:{ROOT_ID}"],
        "pipeline_tag": "text-generation",
        "library_name": "mlx",
        "gated": False,
        "sha": SHA_OLD,
        "lastModified": "2026-01-01T00:00:00.000Z",
    }

    def _fake_request(self, urls: list[str]):
        def fake(url, timeout):
            urls.append(url)
            import json

            rows = [self.LISTING_ROW] if f"quantized%3A{ROOT_ID.replace('/', '%2F')}" in url else []
            return 200, json.dumps(rows).encode(), {}, None

        return fake

    def test_listing_requests_expand_fields(self):
        urls: list[str] = []
        with mock.patch.object(hf_hub, "_request", self._fake_request(urls)):
            status, members, truncated = family.fetch_family(ROOT_ID)
        self.assertEqual(status, "ok")
        self.assertFalse(truncated)
        self.assertTrue(urls)
        for url in urls:
            query = parse_qs(urlsplit(url).query)
            expanded = set(query.get("expand[]", []))
            # the cache key, plus every listing field the family code reads
            self.assertTrue(
                {"sha", "lastModified", "downloads", "likes", "tags", "pipeline_tag", "library_name", "gated"}
                <= expanded,
                url,
            )
            self.assertNotIn("full", query)
        self.assertEqual([m["id"] for m in members], [CHILD_ID])

    def test_member_keeps_prior_fields_and_gains_sha(self):
        urls: list[str] = []
        with mock.patch.object(hf_hub, "_request", self._fake_request(urls)):
            _status, members, _t = family.fetch_family(ROOT_ID)
        level = family.classify_member(members[0], ROOT_ID)
        summary = family.summarize_member(members[0], level, ROOT_ID)
        self.assertEqual(summary["id"], CHILD_ID)
        self.assertEqual(summary["downloads"], 42)
        self.assertEqual(summary["likes"], 3)
        self.assertEqual(summary["pipeline_tag"], "text-generation")
        self.assertEqual(summary["library"], "mlx")
        self.assertEqual(summary["level"], "mlx")
        self.assertEqual(summary["relation"], "quantized")
        self.assertFalse(summary["gated"])
        self.assertIn(f"base_model:quantized:{ROOT_ID}", summary["quant_tags"])
        self.assertEqual(summary["sha"], SHA_OLD)
        self.assertEqual(summary["lastModified"], "2026-01-01T00:00:00.000Z")

    def test_member_without_sha_has_no_sha_key(self):
        row = {k: v for k, v in self.LISTING_ROW.items() if k not in {"sha", "lastModified"}}
        summary = family.summarize_member(row, "mlx", ROOT_ID)
        self.assertNotIn("sha", summary)
        self.assertNotIn("lastModified", summary)

    def test_default_list_filter_unchanged(self):
        urls: list[str] = []
        with mock.patch.object(hf_hub, "_request", self._fake_request(urls)):
            hf_hub.list_filter(f"base_model:{ROOT_ID}")
        query = parse_qs(urlsplit(urls[0]).query)
        self.assertEqual(query.get("full"), ["true"])
        self.assertNotIn("expand[]", query)


if __name__ == "__main__":
    unittest.main()
