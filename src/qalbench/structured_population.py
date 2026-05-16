"""Small exact quantum-population register dynamics.

This module complements ``population.py``. The existing population benchmark
keeps classical individuals with event-level quantum diagnostics. Here the
population sites themselves are qubits in one joint density matrix, which makes
the dynamics structurally quantum but necessarily small.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import pi
from operator import mul

import numpy as np

from .channels import validate_probability
from .gates import I2, Z, ry
from .metrics import l1_coherence, negativity


@dataclass(frozen=True)
class StructuredPopulationParams:
    """Parameters for an exact fixed-site quantum population register."""

    site_count: int = 3
    steps: int = 3
    initial_theta: float = pi / 2.0
    mutation_probability: float = 0.0
    mutation_angle: float = pi / 8.0
    interaction_angle: float = 0.0
    damping_probability: float = 0.0
    dephase_after_step: bool = False
    birth_edges: tuple[tuple[int, int], ...] | None = None


def _kron_all(operators: list[np.ndarray]) -> np.ndarray:
    return reduce(np.kron, operators)


def _bit(state: int, qubit: int, n_qubits: int) -> int:
    return (state >> (n_qubits - 1 - qubit)) & 1


def _validate_params(params: StructuredPopulationParams) -> tuple[tuple[int, int], ...]:
    if params.site_count < 2:
        raise ValueError("site_count must be at least 2")
    if params.site_count > 8:
        raise ValueError("site_count above 8 is intentionally not supported by exact dense dynamics")
    if params.steps < 0:
        raise ValueError("steps must be nonnegative")
    validate_probability(params.mutation_probability, "mutation_probability")
    validate_probability(params.damping_probability, "damping_probability")
    edges = params.birth_edges
    if edges is None:
        edges = tuple((index, index + 1) for index in range(params.site_count - 1))
    if not edges:
        raise ValueError("at least one birth edge is required")
    for parent, child in edges:
        if parent == child:
            raise ValueError("birth edges must connect distinct sites")
        if not (0 <= parent < params.site_count and 0 <= child < params.site_count):
            raise ValueError("birth edge index out of range")
    return edges


def one_qubit_operator(operator: np.ndarray, target: int, n_qubits: int) -> np.ndarray:
    """Embed a one-qubit operator in an ``n_qubits`` register."""

    if not 0 <= target < n_qubits:
        raise ValueError("target out of range")
    ops = [I2 for _ in range(n_qubits)]
    ops[target] = operator
    return _kron_all(ops)


def controlled_not_operator(control: int, target: int, n_qubits: int) -> np.ndarray:
    """Return a dense CNOT matrix on an ``n_qubits`` register."""

    if control == target:
        raise ValueError("control and target must differ")
    if not (0 <= control < n_qubits and 0 <= target < n_qubits):
        raise ValueError("control or target out of range")
    dimension = 2**n_qubits
    matrix = np.zeros((dimension, dimension), dtype=complex)
    target_mask = 1 << (n_qubits - 1 - target)
    for state in range(dimension):
        out_state = state ^ target_mask if _bit(state, control, n_qubits) else state
        matrix[out_state, state] = 1.0
    return matrix


def zz_operator(first: int, second: int, angle: float, n_qubits: int) -> np.ndarray:
    """Return ``exp(-i angle Z_i Z_j / 2)`` on an ``n_qubits`` register."""

    if first == second:
        raise ValueError("ZZ operator requires two distinct qubits")
    dimension = 2**n_qubits
    phases = np.empty(dimension, dtype=complex)
    for state in range(dimension):
        zi = 1.0 if _bit(state, first, n_qubits) == 0 else -1.0
        zj = 1.0 if _bit(state, second, n_qubits) == 0 else -1.0
        phases[state] = np.exp(-0.5j * angle * zi * zj)
    return np.diag(phases)


def apply_unitary(rho: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    return unitary @ rho @ unitary.conj().T


def _apply_one_qubit_kraus(
    rho: np.ndarray,
    kraus_ops: tuple[np.ndarray, ...],
    target: int,
    n_qubits: int,
) -> np.ndarray:
    out = np.zeros_like(rho, dtype=complex)
    for op in kraus_ops:
        full = one_qubit_operator(op, target, n_qubits)
        out += full @ rho @ full.conj().T
    return out


def _amplitude_damping(rho: np.ndarray, gamma: float, target: int, n_qubits: int) -> np.ndarray:
    gamma = validate_probability(gamma, "damping_probability")
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
    k1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    return _apply_one_qubit_kraus(rho, (k0, k1), target, n_qubits)


def _stochastic_mutation(
    rho: np.ndarray,
    probability: float,
    angle: float,
    target: int,
    n_qubits: int,
) -> np.ndarray:
    probability = validate_probability(probability, "mutation_probability")
    if probability == 0.0 or angle == 0.0:
        return rho
    rotated = apply_unitary(rho, one_qubit_operator(ry(angle), target, n_qubits))
    return (1.0 - probability) * rho + probability * rotated


def full_dephase(rho: np.ndarray) -> np.ndarray:
    return np.diag(np.diag(rho)).astype(complex)


def initial_structured_state(params: StructuredPopulationParams) -> np.ndarray:
    """Return a product state with site 0 initialized and other sites empty."""

    genotype = np.array(
        [np.cos(params.initial_theta / 2.0), np.sin(params.initial_theta / 2.0)],
        dtype=complex,
    )
    zero = np.array([1.0, 0.0], dtype=complex)
    state = _kron_all([genotype] + [zero for _ in range(params.site_count - 1)])
    return np.outer(state, state.conj())


def reduced_density_matrix(
    rho: np.ndarray,
    keep: tuple[int, ...],
    n_qubits: int,
) -> np.ndarray:
    """Reduced density matrix by explicit environment summation."""

    keep = tuple(keep)
    if len(set(keep)) != len(keep):
        raise ValueError("keep qubits must be unique")
    if any(qubit < 0 or qubit >= n_qubits for qubit in keep):
        raise ValueError("keep qubit out of range")
    traced = tuple(qubit for qubit in range(n_qubits) if qubit not in keep)
    reduced_dimension = 2 ** len(keep)
    reduced = np.zeros((reduced_dimension, reduced_dimension), dtype=complex)
    dimension = 2**n_qubits

    def projected_index(state: int, qubits: tuple[int, ...]) -> int:
        value = 0
        for qubit in qubits:
            value = (value << 1) | _bit(state, qubit, n_qubits)
        return value

    for row in range(dimension):
        row_env = tuple(_bit(row, qubit, n_qubits) for qubit in traced)
        row_keep = projected_index(row, keep)
        for col in range(dimension):
            col_env = tuple(_bit(col, qubit, n_qubits) for qubit in traced)
            if row_env == col_env:
                reduced[row_keep, projected_index(col, keep)] += rho[row, col]
    return reduced


def structured_population_metrics(
    rho: np.ndarray,
    site_count: int,
) -> dict[str, float]:
    """Compute population-level quantum diagnostics for a joint register."""

    site_z: list[float] = []
    for site in range(site_count):
        z_op = one_qubit_operator(Z, site, site_count)
        site_z.append(float(np.real_if_close(np.trace(rho @ z_op)).real))
    pair_negativities = []
    for first in range(site_count):
        for second in range(first + 1, site_count):
            reduced = reduced_density_matrix(rho, (first, second), site_count)
            pair_negativities.append(negativity(reduced)[0])
    probabilities_one = [0.5 * (1.0 - value) for value in site_z]
    return {
        "trace": float(np.real_if_close(np.trace(rho)).real),
        "purity": float(np.real_if_close(np.trace(rho @ rho)).real),
        "global_coherence_l1": l1_coherence(rho),
        "mean_site_p_one": float(np.mean(probabilities_one)),
        "max_site_p_one": float(np.max(probabilities_one)),
        "mean_pair_negativity": float(np.mean(pair_negativities)) if pair_negativities else 0.0,
        "max_pair_negativity": float(np.max(pair_negativities)) if pair_negativities else 0.0,
    }


def _step_row(
    params: StructuredPopulationParams,
    step: int,
    edge: tuple[int, int] | None,
    rho: np.ndarray,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "step": step,
        "parent_site": -1 if edge is None else edge[0],
        "child_site": -1 if edge is None else edge[1],
        "site_count": params.site_count,
        "dephase_after_step": str(params.dephase_after_step),
    }
    row.update(structured_population_metrics(rho, params.site_count))
    return row


def run_structured_population(
    params: StructuredPopulationParams,
) -> dict[str, object]:
    """Run a small exact quantum-population register benchmark."""

    edges = _validate_params(params)
    rho = initial_structured_state(params)
    trajectory = [_step_row(params, 0, None, rho)]
    for step in range(1, params.steps + 1):
        edge = edges[(step - 1) % len(edges)]
        parent, child = edge
        rho = apply_unitary(rho, controlled_not_operator(parent, child, params.site_count))
        rho = _stochastic_mutation(
            rho,
            params.mutation_probability,
            params.mutation_angle,
            child,
            params.site_count,
        )
        if params.interaction_angle:
            rho = apply_unitary(rho, zz_operator(parent, child, params.interaction_angle, params.site_count))
        if params.damping_probability:
            rho = _amplitude_damping(rho, params.damping_probability, child, params.site_count)
        if params.dephase_after_step:
            rho = full_dephase(rho)
        trajectory.append(_step_row(params, step, edge, rho))
    return {
        "params": params,
        "rho": rho,
        "trajectory": trajectory,
        "summary": trajectory[-1],
        "state_dimension": reduce(mul, (2 for _ in range(params.site_count)), 1),
    }
