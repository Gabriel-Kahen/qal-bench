from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qalbench.metrics import (  # noqa: E402
    chsh_maximum,
    concurrence,
    compute_metrics,
    l1_coherence,
    negativity,
    pauli_correlation_matrix,
    partial_transpose,
    z_expectations,
)


def density(vector: np.ndarray) -> np.ndarray:
    return np.outer(vector, vector.conj())


class MetricFormulaTests(unittest.TestCase):
    def assertClose(self, actual: float, expected: float, tol: float = 1e-10) -> None:
        self.assertLessEqual(abs(actual - expected), tol, f"{actual} != {expected}")

    def test_bell_state_metrics(self) -> None:
        bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / math.sqrt(2.0)
        rho = density(bell)

        neg, ppt_min = negativity(rho)
        self.assertClose(neg, 0.5)
        self.assertClose(ppt_min, -0.5)
        self.assertClose(concurrence(rho), 1.0)
        self.assertClose(chsh_maximum(rho), 2.0 * math.sqrt(2.0))
        self.assertClose(l1_coherence(rho), 1.0)
        self.assertEqual(z_expectations(rho), (0.0, 0.0))
        correlations = pauli_correlation_matrix(rho)
        self.assertClose(correlations[0, 0], 1.0)
        self.assertClose(correlations[1, 1], -1.0)
        metrics = compute_metrics(rho)
        self.assertClose(metrics["xx"], 1.0)
        self.assertClose(metrics["xy"], 0.0)
        self.assertClose(metrics["yx"], 0.0)
        self.assertClose(metrics["yy"], -1.0)

    def test_product_state_is_not_entangled(self) -> None:
        zero_zero = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        rho = density(zero_zero)

        neg, ppt_min = negativity(rho)
        self.assertClose(neg, 0.0)
        self.assertClose(ppt_min, 0.0)
        self.assertClose(concurrence(rho), 0.0)
        self.assertClose(chsh_maximum(rho), 2.0)
        self.assertClose(l1_coherence(rho), 0.0)
        self.assertEqual(z_expectations(rho), (1.0, 1.0))

    def test_separable_coherence_is_only_a_diagnostic(self) -> None:
        plus_plus = np.ones(4, dtype=complex) / 2.0
        rho = density(plus_plus)

        neg, _ = negativity(rho)
        self.assertClose(neg, 0.0)
        self.assertClose(concurrence(rho), 0.0)
        self.assertClose(chsh_maximum(rho), 2.0)
        self.assertGreater(l1_coherence(rho), 0.0)

    def test_werner_state_metrics(self) -> None:
        p = 0.5
        bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / math.sqrt(2.0)
        rho = p * density(bell) + (1.0 - p) * np.eye(4, dtype=complex) / 4.0

        neg, ppt_min = negativity(rho)
        self.assertClose(neg, 0.125)
        self.assertClose(ppt_min, -0.125)
        self.assertClose(concurrence(rho), 0.25)
        self.assertClose(chsh_maximum(rho), p * 2.0 * math.sqrt(2.0))

    def test_partial_transpose_rejects_invalid_subsystem(self) -> None:
        with self.assertRaises(ValueError):
            partial_transpose(np.eye(4, dtype=complex) / 4.0, subsystem=2)


if __name__ == "__main__":
    unittest.main()
