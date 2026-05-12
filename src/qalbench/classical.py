"""Matched classical Markov baseline over two-bit strings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channels import validate_probability


BITSTRINGS = ("00", "01", "10", "11")


@dataclass(frozen=True)
class ClassicalParams:
    """Parameters for the classical bitstring baseline."""

    theta: float
    local_perturbation_probability: float = 0.0
    local_perturbation_angle: float = 0.0
    damping_probability: float = 0.0


def _apply_transition(probabilities: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return probabilities @ matrix


def _local_perturbation_matrix(bit: int, flip_probability: float) -> np.ndarray:
    flip_probability = validate_probability(
        flip_probability, "local perturbation flip probability"
    )
    matrix = np.zeros((4, 4), dtype=float)
    for state in range(4):
        flipped = state ^ (1 << (1 - bit))
        matrix[state, state] += 1.0 - flip_probability
        matrix[state, flipped] += flip_probability
    return matrix


def _reset_matrix(bit: int, probability: float) -> np.ndarray:
    probability = validate_probability(probability, "loss/reset probability")
    matrix = np.zeros((4, 4), dtype=float)
    mask = 1 << (1 - bit)
    for state in range(4):
        reset = state & ~mask
        matrix[state, state] += 1.0 - probability
        matrix[state, reset] += probability
    return matrix


def initial_distribution(theta: float) -> np.ndarray:
    """Initial genotype basis probabilities and offspring fixed to zero."""

    p_one = np.sin(theta / 2.0) ** 2
    return np.array([1.0 - p_one, 0.0, p_one, 0.0], dtype=float)


def replicate_cnot(probabilities: np.ndarray) -> np.ndarray:
    out = np.zeros(4, dtype=float)
    for state, probability in enumerate(probabilities):
        genotype = (state >> 1) & 1
        offspring = state & 1
        replicated = (genotype << 1) | (offspring ^ genotype)
        out[replicated] += probability
    return out


def run_classical_event(params: ClassicalParams) -> dict[str, np.ndarray]:
    """Run the matched Markov baseline and return probabilities and rho."""

    probabilities = replicate_cnot(initial_distribution(params.theta))
    local_perturbation_probability = validate_probability(
        params.local_perturbation_probability, "local perturbation probability"
    )
    damping_probability = validate_probability(
        params.damping_probability, "damping probability"
    )
    flip_probability = local_perturbation_probability * (
        np.sin(params.local_perturbation_angle / 2.0) ** 2
    )
    if flip_probability:
        probabilities = _apply_transition(
            probabilities,
            _local_perturbation_matrix(bit=1, flip_probability=flip_probability),
        )

    if damping_probability:
        probabilities = _apply_transition(
            probabilities, _reset_matrix(bit=1, probability=damping_probability)
        )

    probabilities = probabilities / probabilities.sum()
    return {"probabilities": probabilities, "rho": np.diag(probabilities).astype(complex)}
