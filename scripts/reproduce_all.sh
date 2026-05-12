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

if [[ -n "${TECTONIC:-}" ]]; then
  TECTONIC_BIN="$TECTONIC"
else
  TECTONIC_BIN="$(command -v tectonic || true)"
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found. Create .venv or set PYTHON=/path/to/python." >&2
  exit 1
fi

if [[ -z "$TECTONIC_BIN" || ! -x "$TECTONIC_BIN" ]]; then
  echo "Tectonic not found. Install tectonic or set TECTONIC=/path/to/tectonic." >&2
  exit 1
fi

EXPECTED_TECTONIC_VERSION="${EXPECTED_TECTONIC_VERSION:-Tectonic 0.16.9}"
TECTONIC_VERSION="$("$TECTONIC_BIN" --version | head -n 1)"
if [[ "$TECTONIC_VERSION" != "$EXPECTED_TECTONIC_VERSION" ]]; then
  echo "Unexpected Tectonic version: $TECTONIC_VERSION" >&2
  echo "Expected $EXPECTED_TECTONIC_VERSION. Override with EXPECTED_TECTONIC_VERSION=... if intentional." >&2
  exit 1
fi

"$PYTHON_BIN" -m unittest discover -s "$ROOT/tests"
"$PYTHON_BIN" "$ROOT/scripts/run_qalbench_sweeps.py"
"$PYTHON_BIN" "$ROOT/scripts/verify_qalbench_results.py"
"$PYTHON_BIN" "$ROOT/scripts/run_qalbench_population.py"
"$PYTHON_BIN" "$ROOT/scripts/verify_qalbench_population.py"
mkdir -p "$ROOT/paper/build"
(cd "$ROOT/paper" && "$TECTONIC_BIN" --outdir build main.tex)
