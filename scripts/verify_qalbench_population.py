#!/usr/bin/env python3
"""Repository wrapper for the package-native population benchmark verifier."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("QALBENCH_ROOT", str(ROOT))

from qalbench.population_verify import main  # noqa: E402


if __name__ == "__main__":
    main()
