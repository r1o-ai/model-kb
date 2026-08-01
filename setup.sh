#!/usr/bin/env bash
# setup.sh — stand up model-kb end to end.
#
#   ./setup.sh              # deps + seed corpus + BM25 index + verify
#   ./setup.sh --empty      # deps + index only; start from an EMPTY corpus
#   ./setup.sh --no-seed    # deps only; leave any existing corpus untouched
#
# Idempotent, and it will NOT overwrite an existing records.jsonl without saying so.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="$HERE/engine"
RECORDS="$ENGINE/records.jsonl"
SEED="$HERE/data/records.seed.jsonl"
MODE=seed
for a in "$@"; do
  case "$a" in
    --empty)   MODE=empty ;;
    --no-seed) MODE=none ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
  esac
done

say() { printf "\n\033[1m== %s\033[0m\n" "$*"; }
ok()  { printf "  ✓ %s\n" "$*"; }
bad() { printf "  ✗ %s\n" "$*"; }

say "1/4 Python dependencies"
command -v python3 >/dev/null || { bad "python3 not found"; exit 1; }
ok "python $(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
python3 -m pip install --quiet --upgrade -r "$ENGINE/requirements.txt" 2>&1 | tail -1 || true
python3 - <<'PY'
try:
    import rank_bm25  # noqa: F401
    print("  ✓ rank-bm25 present")
except ImportError:
    print("  ✗ rank-bm25 MISSING — search will not work (pip install rank-bm25)")
PY

say "2/4 Corpus"
if [[ -f "$RECORDS" ]]; then
  N=$(grep -c . "$RECORDS" 2>/dev/null || echo 0)
  ok "existing corpus: $N records — LEFT UNTOUCHED"
  echo "    (snapshot before any ingest:  python3 scripts/kb-guard.py backup engine/records.jsonl)"
elif [[ "$MODE" == "seed" ]]; then
  cp "$SEED" "$RECORDS"
  ok "seeded from data/records.seed.jsonl ($(grep -c . "$RECORDS") records)"
  echo "    Seed is ANONYMIZED reference knowledge: hostnames → node-N, local paths →"
  echo "    /path/to/models/. Re-run the ingests to replace it with YOUR hardware."
elif [[ "$MODE" == "empty" ]]; then
  : > "$RECORDS"; ok "empty corpus created"
else
  echo "  (skipped — no corpus present; ingest or seed before searching)"
fi

say "3/4 BM25 index"
if [[ -s "$RECORDS" ]]; then
  (cd "$ENGINE" && python3 build_bm25.py) && ok "index built"
else
  echo "  (skipped — corpus is empty)"
fi

say "4/4 Health"
if [[ -s "$RECORDS" ]]; then
  python3 "$HERE/scripts/kb-guard.py" verify "$RECORDS" || true
fi

cat <<EOF

Next:
  Search:      cd engine && python3 model_kb.py search "MoE that fits 2 nodes" --top-k 5
  Inspect:     python3 scripts/kb-guard.py check engine/records.jsonl

  ⛔ BEFORE ANY INGEST — the corpus is destructible:
     python3 scripts/kb-guard.py backup engine/records.jsonl
     python3 scripts/kb-guard.py plan   engine/records.jsonl recipes   # exits 4 if destructive

  Rebuild (ORDER MATTERS — recipes clobbers, everything else merges):
     cd engine && python3 pipeline.py
  Then always:
     python3 scripts/kb-guard.py verify engine/records.jsonl --min <expected>

  MCP server:  engine/model-kb-mcp.py   (6 tools: search/get/family/serving/load/transfer)
EOF
