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

from qalbench import ClassicalParams, QuantumParams, run_classical_event, run_quantum_event  # noqa: E402
from qalbench.metrics import compute_metrics  # noqa: E402


class ModelBaselineTests(unittest.TestCase):
    def assertClose(self, actual: float, expected: float, tol: float = 1e-10) -> None:
        self.assertLessEqual(abs(actual - expected), tol, f"{actual} != {expected}")

    def test_quantum_diagonal_matches_classical_markov_family(self) -> None:
        for theta in (0.0, math.pi / 5.0, math.pi / 2.0, math.pi):
            for perturbation_probability in (0.0, 0.25, 1.0):
                for damping_probability in (0.0, 0.4, 1.0):
                    for interaction_angle in (0.0, 0.73):
                        for dephase_probability in (0.0, 0.35):
                            qparams = QuantumParams(
                                theta=theta,
                                phi=0.37,
                                local_perturbation_probability=perturbation_probability,
                                local_perturbation_angle=math.pi / 3.0,
                                interaction_angle=interaction_angle,
                                damping_probability=damping_probability,
                                dephase_probability=dephase_probability,
                            )
                            cparams = ClassicalParams(
                                theta=theta,
                                local_perturbation_probability=perturbation_probability,
                                local_perturbation_angle=math.pi / 3.0,
                                damping_probability=damping_probability,
                            )

                            qstates = run_quantum_event(qparams)
                            classical = run_classical_event(cparams)
                            quantum_probabilities = np.real_if_close(np.diag(qstates["rho"])).real
                            dephased_probabilities = np.real_if_close(
                                np.diag(qstates["dephased"])
                            ).real

                            np.testing.assert_allclose(
                                quantum_probabilities,
                                classical["probabilities"],
                                atol=1e-12,
                                rtol=0.0,
                            )
                            np.testing.assert_allclose(
                                dephased_probabilities,
                                classical["probabilities"],
                                atol=1e-12,
                                rtol=0.0,
                            )

                            qmetrics = compute_metrics(qstates["rho"], prefix="")
                            cmetrics = compute_metrics(classical["rho"], prefix="")
                            for key in ("z_genotype", "z_offspring", "z_copy_agreement"):
                                self.assertClose(qmetrics[key], cmetrics[key])

    def test_invalid_probabilities_raise_clear_errors(self) -> None:
        with self.assertRaises(ValueError):
            run_quantum_event(
                QuantumParams(theta=0.0, local_perturbation_probability=1.1)
            )
        with self.assertRaises(ValueError):
            run_quantum_event(QuantumParams(theta=0.0, damping_probability=-0.1))
        with self.assertRaises(ValueError):
            run_classical_event(ClassicalParams(theta=0.0, damping_probability=1.1))

    def test_interaction_does_not_change_computational_basis_populations(self) -> None:
        base = QuantumParams(theta=math.pi / 2.0, interaction_angle=0.0)
        interacting = QuantumParams(theta=math.pi / 2.0, interaction_angle=0.73)

        base_probabilities = np.real_if_close(np.diag(run_quantum_event(base)["rho"])).real
        interacting_probabilities = np.real_if_close(
            np.diag(run_quantum_event(interacting)["rho"])
        ).real

        np.testing.assert_allclose(
            base_probabilities,
            interacting_probabilities,
            atol=1e-12,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
