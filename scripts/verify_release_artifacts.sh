#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found. Create .venv or set PYTHON=/path/to/python." >&2
  exit 1
fi

"$PYTHON_BIN" -m unittest discover -s "$ROOT/tests"
"$PYTHON_BIN" "$ROOT/scripts/verify_qalbench_results.py"
"$PYTHON_BIN" "$ROOT/scripts/verify_qalbench_population.py"
