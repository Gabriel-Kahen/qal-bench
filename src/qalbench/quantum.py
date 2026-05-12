"""Minimal two-qubit QAL-inspired quantum-resource event."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channels import amplitude_damping, full_dephase, phase_damping, validate_probability
from .gates import CNOT_01, apply_unitary, one_qubit_gate, ry, zz_interaction


@dataclass(frozen=True)
class QuantumParams:
    """Parameters for one two-qubit resource-kernel event."""

    theta: float
    phi: float = 0.0
    local_perturbation_probability: float = 0.0
    local_perturbation_angle: float = 0.0
    interaction_angle: float = 0.0
    damping_probability: float = 0.0
    dephase_probability: float = 0.0


def initial_state(theta: float, phi: float = 0.0) -> np.ndarray:
    """Return rho for genotype superposition and offspring in |0>."""

    genotype = np.array(
        [np.cos(theta / 2.0), np.exp(1.0j * phi) * np.sin(theta / 2.0)],
        dtype=complex,
    )
    offspring = np.array([1.0, 0.0], dtype=complex)
    psi = np.kron(genotype, offspring)
    return np.outer(psi, psi.conj())


def local_perturbation_channel(
    rho: np.ndarray, probability: float, angle: float, target: int
) -> np.ndarray:
    """Apply a stochastic local Ry perturbation to one qubit."""

    probability = validate_probability(probability, "local perturbation probability")
    if probability == 0.0 or angle == 0.0:
        return rho
    unitary = one_qubit_gate(ry(angle), target=target)
    mutated = apply_unitary(rho, unitary)
    return (1.0 - probability) * rho + probability * mutated


def _apply_kernel_operations(
    params: QuantumParams, dephase_after_operations: bool
) -> np.ndarray:
    """Run the resource-kernel operations, optionally dephasing after each one."""

    rho = initial_state(params.theta, params.phi)
    rho = apply_unitary(rho, CNOT_01)
    if dephase_after_operations:
        rho = full_dephase(rho)

    rho = local_perturbation_channel(
        rho,
        params.local_perturbation_probability,
        params.local_perturbation_angle,
        target=1,
    )
    if dephase_after_operations:
        rho = full_dephase(rho)

    if params.interaction_angle:
        rho = apply_unitary(rho, zz_interaction(params.interaction_angle))
        if dephase_after_operations:
            rho = full_dephase(rho)

    if params.damping_probability:
        rho = amplitude_damping(rho, params.damping_probability, target=1)
        if dephase_after_operations:
            rho = full_dephase(rho)

    if params.dephase_probability:
        rho = phase_damping(rho, params.dephase_probability, target=0)
        rho = phase_damping(rho, params.dephase_probability, target=1)
        if dephase_after_operations:
            rho = full_dephase(rho)

    return rho


def run_quantum_event(params: QuantumParams) -> dict[str, np.ndarray]:
    """Run one event and return coherent and dephased density matrices."""

    rho = _apply_kernel_operations(params, dephase_after_operations=False)
    dephased = _apply_kernel_operations(params, dephase_after_operations=True)
    return {
        "rho": rho,
        "dephased": dephased,
        "basis_dephased_final_only": full_dephase(rho),
    }
