"""Resource and correlation metrics for two-qubit density matrices."""

from __future__ import annotations

import numpy as np

from .gates import X, Y, Z


PAULIS = (X, Y, Z)


def _real_if_close(value: complex | float, tol: float = 1e-10) -> float:
    return float(np.real_if_close(value, tol=tol).real)


def z_expectations(rho: np.ndarray) -> tuple[float, float]:
    zi = np.kron(Z, np.eye(2, dtype=complex))
    iz = np.kron(np.eye(2, dtype=complex), Z)
    return _real_if_close(np.trace(rho @ zi)), _real_if_close(np.trace(rho @ iz))


def l1_coherence(rho: np.ndarray) -> float:
    return float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho))))


def partial_transpose(rho: np.ndarray, subsystem: int = 1) -> np.ndarray:
    tensor = rho.reshape(2, 2, 2, 2)
    if subsystem == 0:
        transposed = tensor.transpose(2, 1, 0, 3)
    elif subsystem == 1:
        transposed = tensor.transpose(0, 3, 2, 1)
    else:
        raise ValueError("subsystem must be 0 or 1")
    return transposed.reshape(4, 4)


def negativity(rho: np.ndarray) -> tuple[float, float]:
    eigvals = np.linalg.eigvalsh(partial_transpose(rho, subsystem=1))
    neg = float(np.sum(np.abs(eigvals[eigvals < 0.0])))
    return neg, float(np.min(eigvals))


def concurrence(rho: np.ndarray) -> float:
    yy = np.kron(Y, Y)
    spin_flipped = yy @ rho.conj() @ yy
    eigvals = np.linalg.eigvals(rho @ spin_flipped)
    roots = np.sqrt(np.clip(np.real_if_close(eigvals, tol=1000).real, 0.0, None))
    roots.sort()
    return float(max(0.0, roots[-1] - np.sum(roots[:-1])))


def chsh_maximum(rho: np.ndarray) -> float:
    corr = pauli_correlation_matrix(rho)
    eigvals = np.linalg.eigvalsh(corr.T @ corr)
    two_largest = np.sort(np.clip(eigvals, 0.0, None))[-2:]
    return float(2.0 * np.sqrt(np.sum(two_largest)))


def pauli_correlation_matrix(rho: np.ndarray) -> np.ndarray:
    """Return T_ij = Tr[rho sigma_i x sigma_j] for X, Y, Z order."""

    corr = np.empty((3, 3), dtype=float)
    for i, sigma_i in enumerate(PAULIS):
        for j, sigma_j in enumerate(PAULIS):
            corr[i, j] = _real_if_close(np.trace(rho @ np.kron(sigma_i, sigma_j)))
    return corr


def von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    eigvals = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)
    positive = eigvals[eigvals > 1e-12]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive) / np.log(base)))


def compute_metrics(rho: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Compute scalar metrics for a two-qubit density matrix."""

    z_genotype, z_offspring = z_expectations(rho)
    neg, ppt_min_eig = negativity(rho)
    corr = pauli_correlation_matrix(rho)
    probabilities = np.real_if_close(np.diag(rho), tol=1000).real
    return {
        f"{prefix}trace": _real_if_close(np.trace(rho)),
        f"{prefix}purity": _real_if_close(np.trace(rho @ rho)),
        f"{prefix}p_00": float(probabilities[0]),
        f"{prefix}p_01": float(probabilities[1]),
        f"{prefix}p_10": float(probabilities[2]),
        f"{prefix}p_11": float(probabilities[3]),
        f"{prefix}z_copy_agreement": float(probabilities[0] + probabilities[3]),
        f"{prefix}z_genotype": z_genotype,
        f"{prefix}z_offspring": z_offspring,
        f"{prefix}coherence_l1": l1_coherence(rho),
        f"{prefix}xx": float(corr[0, 0]),
        f"{prefix}xy": float(corr[0, 1]),
        f"{prefix}yx": float(corr[1, 0]),
        f"{prefix}yy": float(corr[1, 1]),
        f"{prefix}concurrence": concurrence(rho),
        f"{prefix}negativity": neg,
        f"{prefix}ppt_min_eig": ppt_min_eig,
        f"{prefix}chsh_max": chsh_maximum(rho),
        f"{prefix}entropy": von_neumann_entropy(rho),
    }
