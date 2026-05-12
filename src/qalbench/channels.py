"""Quantum channels for dense two-qubit density matrices."""

from __future__ import annotations

import numpy as np


I2 = np.eye(2, dtype=complex)


def validate_probability(value: float, name: str = "probability") -> float:
    """Return a probability in [0, 1] or raise a clear error."""

    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _embed_one_qubit(op: np.ndarray, target: int, n_qubits: int = 2) -> np.ndarray:
    if n_qubits != 2:
        raise ValueError("qalbench currently supports exactly two qubits")
    if target not in (0, 1):
        raise ValueError("target must be 0 or 1")
    return np.kron(op, I2) if target == 0 else np.kron(I2, op)


def apply_kraus(
    rho: np.ndarray, kraus_ops: list[np.ndarray], target: int | None = None
) -> np.ndarray:
    """Apply Kraus operators to a density matrix.

    If ``target`` is provided, one-qubit Kraus operators are embedded into the
    two-qubit Hilbert space. Otherwise operators must already be full-sized.
    """

    out = np.zeros_like(rho, dtype=complex)
    for op in kraus_ops:
        full_op = _embed_one_qubit(op, target) if target is not None else op
        out += full_op @ rho @ full_op.conj().T
    return out


def amplitude_damping(rho: np.ndarray, gamma: float, target: int) -> np.ndarray:
    """Apply amplitude damping with probability ``gamma`` to one qubit."""

    gamma = validate_probability(gamma, "gamma")
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
    k1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    return apply_kraus(rho, [k0, k1], target=target)


def phase_damping(rho: np.ndarray, probability: float, target: int) -> np.ndarray:
    """Apply phase damping that suppresses off-diagonal terms."""

    probability = validate_probability(probability, "phase damping probability")
    k0 = np.sqrt(1.0 - probability) * I2
    k1 = np.sqrt(probability) * np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    k2 = np.sqrt(probability) * np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    return apply_kraus(rho, [k0, k1, k2], target=target)


def full_dephase(rho: np.ndarray) -> np.ndarray:
    """Return the computational-basis dephased state."""

    return np.diag(np.diag(rho)).astype(complex)
