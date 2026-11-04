"""The public seed never carries cluster identity (2026-11-04 leak regression).

A staged seed carried `fxnode.fxtail-net.ts.net:8000`, a LAN IP and a
/home/<user> path because the anonymizer pinned a host list that lacked the node
and had no tailnet/IP/home-path rules. Fixture data only; the live cluster file
and records are never read.

    python3 -I -m unittest discover -s engine/tests -v
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
import export_seed as es  # noqa: E402

LEAKY = {
    "id": "recipe:fx",
    "kind": "research_note",
    "notes": "Same request OK via https://fxnode.fxtail-net.ts.net:8000. LAN 10.0.0.5, "
             "tailscale 100.64.0.7, path /home/fxuser/models/x and /Users/fxuser/Models/y, "
             "ask fx.person@example.com. Ran on fxnode and fxother; Hugging Face Hub is fine.",
    "mlx_versions": {"fxnode": "0.31", "default": "0.30"},
    "nodes": ["fxnode", "fxother"],
    "artifact": {"path": "/Users/fxuser/Models/Qwen-X", "asmi_path": "/x"},
}


def _cluster(tmp: Path, names: list[str]) -> Path:
    p = tmp / "cluster.json"
    p.write_text(json.dumps({"nodes": [{"hostname": n} for n in names]}))
    return p


class ExportSeed(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_scrubs_every_identity_shape(self):
        anon = es.Anonymizer(es.load_node_names(_cluster(self.tmp, ["fxnode", "fxother"])))
        out = json.dumps(anon.scrub(LEAKY))
        for leaked in ("ts.net", "10.0.0.5", "100.64.0.7", "/home/fxuser", "/Users/fxuser",
                       "fx.person@example.com", '"fxnode"', "fxother"):
            self.assertNotIn(leaked, out, leaked)
        self.assertIn("Hugging Face Hub", out)  # case-sensitive host match keeps prose
        self.assertIn("a-node.tailnet", out)
        self.assertEqual(anon.leak_scan(out), [])

    def test_leak_scan_flags_the_regression_line(self):
        anon = es.Anonymizer(es.load_node_names(_cluster(self.tmp, ["fxnode"])))
        hits = anon.leak_scan('{"notes":"OK via https://fxnode.fxtail-net.ts.net:8000"}')
        self.assertTrue(any(h.startswith("tailnet name") for h in hits), hits)

    def test_node_names_come_from_the_cluster_file(self):
        names = es.load_node_names(_cluster(self.tmp, ["Brand-New-Node.lan"]))
        self.assertIn("brand-new-node", names)
        self.assertIn(socket.gethostname().split(".")[0].lower(), names)

    def test_missing_or_empty_inventory_fails_closed(self):
        with self.assertRaises(es.InventoryUnavailable):
            es.load_node_names(self.tmp / "nope.json")
        empty = self.tmp / "empty.json"
        empty.write_text(json.dumps({"nodes": []}))
        with self.assertRaises(es.InventoryUnavailable):
            es.load_node_names(empty)

    def test_cli_refuses_to_write_without_inventory(self):
        recs = self.tmp / "records.jsonl"
        recs.write_text(json.dumps(LEAKY) + "\n")
        out = self.tmp / "seed.jsonl"
        r = subprocess.run([sys.executable, "-I", str(ENGINE / "export_seed.py"), "--records", str(recs),
                            "--out", str(out), "--cluster", str(self.tmp / "missing.json")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, es.EXIT_NO_INVENTORY, r.stderr)
        self.assertFalse(out.exists())

    def test_cli_export_is_clean(self):
        recs = self.tmp / "records.jsonl"
        recs.write_text(json.dumps(LEAKY) + "\n")
        out = self.tmp / "seed.jsonl"
        r = subprocess.run([sys.executable, "-I", str(ENGINE / "export_seed.py"), "--records", str(recs),
                            "--out", str(out), "--cluster", str(_cluster(self.tmp, ["fxnode", "fxother"]))],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("ts.net", out.read_text())


if __name__ == "__main__":
    unittest.main()
