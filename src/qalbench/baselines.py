"""Baseline catalog and resource-kernel baseline evaluators."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .classical import ClassicalParams, run_classical_event
from .metrics import compute_metrics
from .quantum import QuantumParams, run_quantum_event


@dataclass(frozen=True)
class BaselineSpec:
    """Declarative baseline family used by suite verification."""

    baseline_id: str
    family: str
    claim_axis: str
    description: str
    destroys_resource: bool
    simulator_family: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


BASELINE_SPECS: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        "exact_density_matrix",
        "quantum_simulator",
        "quantum_resource",
        "Dense state-vector or density-matrix simulator used as an exact small-system reference.",
        False,
        simulator_family=True,
    ),
    BaselineSpec(
        "dephased_density",
        "resource_destroying",
        "quantum_resource",
        "Computational-basis dephasing matched to the event ordering.",
        True,
    ),
    BaselineSpec(
        "diagonal_markov",
        "classical_transition",
        "artificial_life",
        "Classical Markov transition over computational-basis bitstrings.",
        True,
        simulator_family=True,
    ),
    BaselineSpec(
        "separable_product",
        "separable_state",
        "quantum_resource",
        "Product of local marginals preserving one-qubit reduced states.",
        True,
    ),
    BaselineSpec(
        "maximally_mixed",
        "state_null",
        "quantum_resource",
        "Dimension-matched maximally mixed state.",
        True,
    ),
    BaselineSpec(
        "entanglement_breaking",
        "process_null",
        "quantum_resource",
        "Channel-level control that destroys entanglement transmission.",
        True,
    ),
    BaselineSpec(
        "resource_destroying_channel",
        "process_null",
        "quantum_resource",
        "Claim-matched channel control that destroys the declared resource.",
        True,
    ),
    BaselineSpec(
        "no_inheritance",
        "population_null",
        "artificial_life",
        "Transmission-breaking population null with independent offspring state draws.",
        True,
    ),
    BaselineSpec(
        "neutral_selection",
        "population_null",
        "artificial_life",
        "Population control with trait/resource selection terms disabled.",
        True,
        simulator_family=True,
    ),
    BaselineSpec(
        "randomized_neighbor",
        "interaction_null",
        "artificial_life",
        "Interaction null that randomizes or resamples contact records.",
        True,
    ),
    BaselineSpec(
        "readout_error",
        "hardware_null",
        "quantum_resource",
        "Readout or SPAM error model used to bound finite-shot claims.",
        True,
    ),
    BaselineSpec(
        "stabilizer",
        "classical_simulator",
        "computational_nonclassicality",
        "Clifford/stabilizer simulator baseline.",
        False,
        simulator_family=True,
    ),
    BaselineSpec(
        "low_magic",
        "classical_simulator",
        "computational_nonclassicality",
        "Simulator exploiting low nonstabilizer magic or small T-count.",
        False,
        simulator_family=True,
    ),
    BaselineSpec(
        "tensor_network",
        "classical_simulator",
        "computational_nonclassicality",
        "Tensor-network simulator with declared bond-dimension controls.",
        False,
        simulator_family=True,
    ),
    BaselineSpec(
        "classical_shadow",
        "classical_simulator",
        "computational_nonclassicality",
        "Classical-shadow or reduced-observable reconstruction baseline.",
        False,
        simulator_family=True,
    ),
    BaselineSpec(
        "mean_field",
        "population_simulator",
        "artificial_life",
        "Mean-field population approximation with no individual correlations.",
        True,
        simulator_family=True,
    ),
    BaselineSpec(
        "problem_specific",
        "classical_simulator",
        "computational_nonclassicality",
        "Declared task-specific classical algorithm or null model.",
        False,
        simulator_family=True,
    ),
)


def baseline_catalog() -> tuple[BaselineSpec, ...]:
    return BASELINE_SPECS


def baseline_by_id(baseline_id: str) -> BaselineSpec:
    for baseline in BASELINE_SPECS:
        if baseline.baseline_id == baseline_id:
            return baseline
    raise KeyError(f"unknown baseline_id {baseline_id!r}")


def partial_trace_two_qubit(rho: np.ndarray, keep: int) -> np.ndarray:
    """Return one-qubit reduced density matrix for a two-qubit state."""

    if keep not in (0, 1):
        raise ValueError("keep must be 0 or 1")
    tensor = rho.reshape(2, 2, 2, 2)
    if keep == 0:
        return np.trace(tensor, axis1=1, axis2=3)
    return np.trace(tensor, axis1=0, axis2=2)


def product_of_marginals(rho: np.ndarray) -> np.ndarray:
    """Return the separable product state with the same one-qubit marginals."""

    return np.kron(partial_trace_two_qubit(rho, 0), partial_trace_two_qubit(rho, 1))


def maximally_mixed_state(n_qubits: int = 2) -> np.ndarray:
    """Return a dimension-matched maximally mixed density matrix."""

    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    dimension = 2**n_qubits
    return np.eye(dimension, dtype=complex) / dimension


def evaluate_resource_kernel_baselines(
    params: QuantumParams,
) -> dict[str, dict[str, object]]:
    """Evaluate quantum, dephased, Markov, and separable baselines for one event."""

    qstates = run_quantum_event(params)
    cparams = ClassicalParams(
        theta=params.theta,
        local_perturbation_probability=params.local_perturbation_probability,
        local_perturbation_angle=params.local_perturbation_angle,
        damping_probability=params.damping_probability,
    )
    classical = run_classical_event(cparams)
    states = {
        "quantum": ("exact_density_matrix", qstates["rho"]),
        "dephased": ("dephased_density", qstates["dephased"]),
        "final_dephased": ("dephased_density", qstates["basis_dephased_final_only"]),
        "classical_markov": ("diagonal_markov", classical["rho"]),
        "separable_product": ("separable_product", product_of_marginals(qstates["rho"])),
        "maximally_mixed": ("maximally_mixed", maximally_mixed_state(2)),
    }
    return {
        name: {
            "baseline_family": family,
            "metrics": compute_metrics(rho),
        }
        for name, (family, rho) in states.items()
    }
