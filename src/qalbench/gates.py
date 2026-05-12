"""Dense two-qubit gates used by the benchmark."""

from __future__ import annotations

import numpy as np


I2 = np.eye(2, dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

CNOT_01 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=complex,
)


def ry(angle: float) -> np.ndarray:
    """Single-qubit Y rotation."""

    c = np.cos(angle / 2.0)
    s = np.sin(angle / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def one_qubit_gate(gate: np.ndarray, target: int) -> np.ndarray:
    """Embed a one-qubit gate into a two-qubit system."""

    if target == 0:
        return np.kron(gate, I2)
    if target == 1:
        return np.kron(I2, gate)
    raise ValueError("target must be 0 or 1")


def zz_interaction(angle: float) -> np.ndarray:
    """Return exp(-i angle ZxZ / 2)."""

    zz_diag = np.array([1.0, -1.0, -1.0, 1.0], dtype=complex)
    return np.diag(np.exp(-0.5j * angle * zz_diag))


def apply_unitary(rho: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    """Apply ``unitary`` to density matrix ``rho``."""

    return unitary @ rho @ unitary.conj().T
