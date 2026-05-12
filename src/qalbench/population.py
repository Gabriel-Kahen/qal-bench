"""Population-level QAL benchmark driver.

The population benchmark keeps explicit individuals, parent-child lineages,
birth/death bookkeeping, heritable variation, simple interaction exposure, and
selection. Quantum resources are evaluated at reproduction events with the
two-qubit cloned-observable kernel; the population state itself is not a
many-body density matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from typing import Literal

import numpy as np

from .channels import validate_probability
from .classical import ClassicalParams, run_classical_event
from .metrics import compute_metrics
from .quantum import QuantumParams
from .quantum import run_quantum_event


PopulationModel = Literal["quantum", "dephased", "classical", "no_inheritance"]
PopulationScenario = Literal["basis_selection", "resource_selection", "neutral"]


@dataclass(frozen=True)
class PopulationParams:
    """Parameters for a stochastic population-level QAL benchmark run."""

    model: PopulationModel = "quantum"
    scenario: PopulationScenario = "basis_selection"
    seed: int = 11
    steps: int = 64
    initial_population: int = 24
    carrying_capacity: int = 80
    base_birth_probability: float = 0.28
    base_death_probability: float = 0.06
    density_death_probability: float = 0.08
    mutation_probability: float = 0.08
    mutation_step: float = 0.35
    local_perturbation_probability: float = 0.15
    local_perturbation_angle: float = pi / 3.0
    damping_probability: float = 0.10
    phase_damping: float = 0.0
    interaction_strength: float = 0.12
    lineage_null_permutations: int = 200
    phenotype_selection_strength: float | None = None
    resource_selection_strength: float | None = None


@dataclass
class Individual:
    """One benchmark individual with audited lineage and state records."""

    id: int
    parent_id: int | None
    birth_step: int
    death_step: int | None
    generation: int
    theta: float
    expressed_bit: int
    birth_event_negativity: float
    birth_event_chsh: float
    birth_event_resource_score: float
    mutation_from_parent: bool
    birth_count: int = 0


@dataclass
class BirthEvent:
    """One reproduction event used for lineage and inheritance metrics."""

    step: int
    parent_id: int
    child_id: int
    parent_bit: int
    child_bit: int
    parent_theta: float
    child_theta: float
    mutation: bool
    event_negativity: float
    event_chsh: float
    event_resource_score: float
    birth_probability: float
    interaction_exposure: float


@dataclass
class ReproductionAttempt:
    """One attempted reproduction, including failed attempts."""

    step: int
    parent_id: int
    parent_bit: int
    parent_theta: float
    parent_generation: int
    parent_age: int
    parent_resource_score: float
    partner_id: int | None
    partner_bit: int | None
    interaction_exposure: float
    birth_probability: float
    born: bool
    capacity_blocked: bool
    child_id: int | None


def scenario_strengths(params: PopulationParams) -> tuple[float, float]:
    """Return phenotype and resource selection strengths for the scenario."""

    phenotype = params.phenotype_selection_strength
    resource = params.resource_selection_strength
    if phenotype is not None and resource is not None:
        return phenotype, resource
    if params.scenario == "basis_selection":
        return 0.55 if phenotype is None else phenotype, 0.0 if resource is None else resource
    if params.scenario == "resource_selection":
        return 0.0 if phenotype is None else phenotype, 1.00 if resource is None else resource
    if params.scenario == "neutral":
        return 0.0 if phenotype is None else phenotype, 0.0 if resource is None else resource
    raise ValueError(f"unknown population scenario {params.scenario!r}")


def scenario_interaction_strength(params: PopulationParams) -> float:
    """Return the interaction-exposure coefficient for the scenario."""

    if params.scenario == "neutral":
        return 0.0
    return params.interaction_strength


def _validate_params(params: PopulationParams) -> None:
    if params.model not in {"quantum", "dephased", "classical", "no_inheritance"}:
        raise ValueError(f"unknown population model {params.model!r}")
    if params.scenario not in {"basis_selection", "resource_selection", "neutral"}:
        raise ValueError(f"unknown population scenario {params.scenario!r}")
    if params.steps < 1:
        raise ValueError("steps must be at least 1")
    if params.initial_population < 2:
        raise ValueError("initial_population must be at least 2")
    if params.carrying_capacity < params.initial_population:
        raise ValueError("carrying_capacity must be at least initial_population")
    for name in (
        "base_birth_probability",
        "base_death_probability",
        "density_death_probability",
        "mutation_probability",
        "local_perturbation_probability",
        "damping_probability",
        "phase_damping",
    ):
        validate_probability(float(getattr(params, name)), name)
    if params.mutation_step < 0.0:
        raise ValueError("mutation_step must be nonnegative")
    if params.lineage_null_permutations < 1:
        raise ValueError("lineage_null_permutations must be at least 1")


def _clip_theta(theta: float) -> float:
    return float(np.clip(theta, 0.0, pi))


def _initial_theta(rng: np.random.Generator) -> float:
    """Sample a heritable genotype angle with variation across the population."""

    return float(rng.beta(2.0, 2.0) * pi)


def _binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return float(-(probability * log(probability, 2) + (1.0 - probability) * log(1.0 - probability, 2)))


def _binary_mutual_information(pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return 0.0
    counts = np.zeros((2, 2), dtype=float)
    for left, right in pairs:
        counts[int(left), int(right)] += 1.0
    joint = counts / counts.sum()
    left_marginal = joint.sum(axis=1)
    right_marginal = joint.sum(axis=0)
    mutual_information = 0.0
    for left in range(2):
        for right in range(2):
            if joint[left, right] > 0.0:
                mutual_information += joint[left, right] * log(
                    joint[left, right] / (left_marginal[left] * right_marginal[right]),
                    2,
                )
    return float(mutual_information)


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _event_metrics(theta: float, params: PopulationParams) -> dict[str, float]:
    qparams = QuantumParams(
        theta=theta,
        phi=0.0,
        local_perturbation_probability=params.local_perturbation_probability,
        local_perturbation_angle=params.local_perturbation_angle,
        interaction_angle=0.0,
        damping_probability=params.damping_probability,
        dephase_probability=params.phase_damping,
    )
    if params.model in {"quantum", "no_inheritance"}:
        rho = run_quantum_event(qparams)["rho"]
    elif params.model == "dephased":
        rho = run_quantum_event(qparams)["dephased"]
    elif params.model == "classical":
        cparams = ClassicalParams(
            theta=theta,
            local_perturbation_probability=params.local_perturbation_probability,
            local_perturbation_angle=params.local_perturbation_angle,
            damping_probability=params.damping_probability,
        )
        rho = run_classical_event(cparams)["rho"]
    else:
        raise ValueError(f"unknown population model {params.model!r}")
    metrics = compute_metrics(rho)
    resource_score = min(1.0, max(0.0, 2.0 * metrics["negativity"]))
    metrics["resource_score"] = resource_score
    return metrics


def _sample_bit_from_theta(
    theta: float,
    params: PopulationParams,
    rng: np.random.Generator,
) -> tuple[int, dict[str, float]]:
    metrics = _event_metrics(theta, params)
    p_one = 0.5 * (1.0 - metrics["z_offspring"])
    bit = int(rng.random() < p_one)
    return bit, metrics


def _alive(individuals: dict[int, Individual]) -> list[Individual]:
    return [individual for individual in individuals.values() if individual.death_step is None]


def _interaction_exposures(
    alive: list[Individual],
    rng: np.random.Generator,
) -> tuple[dict[int, tuple[float, int | None, int | None]], int, int]:
    if len(alive) < 2:
        return {individual.id: (0.0, None, None) for individual in alive}, 0, 0
    ids = [individual.id for individual in alive]
    by_id = {individual.id: individual for individual in alive}
    exposures: dict[int, tuple[float, int | None, int | None]] = {}
    opposite = 0
    total = 0
    for individual in alive:
        choices = [candidate for candidate in ids if candidate != individual.id]
        neighbor = by_id[int(rng.choice(choices))]
        total += 1
        if neighbor.expressed_bit != individual.expressed_bit:
            opposite += 1
            exposure = 1.0 if individual.expressed_bit == 1 else -1.0
        else:
            exposure = 0.0
        exposures[individual.id] = (exposure, neighbor.id, neighbor.expressed_bit)
    return exposures, opposite, total


def _birth_probability(
    individual: Individual,
    params: PopulationParams,
    exposure: float,
) -> float:
    phenotype_strength, resource_strength = scenario_strengths(params)
    phenotype_score = 1.0 if individual.expressed_bit == 1 else -1.0
    multiplier = (
        1.0
        + phenotype_strength * phenotype_score
        + resource_strength * individual.birth_event_resource_score
        + scenario_interaction_strength(params) * exposure
    )
    return float(np.clip(params.base_birth_probability * multiplier, 0.0, 0.95))


def _death_probability(alive_count: int, params: PopulationParams) -> float:
    density = alive_count / params.carrying_capacity
    return float(
        np.clip(
            params.base_death_probability + params.density_death_probability * density,
            0.0,
            0.95,
        )
    )


def _trajectory_row(
    *,
    params: PopulationParams,
    step: int,
    individuals: dict[int, Individual],
    births_this_step: int,
    deaths_this_step: int,
    opposite_interactions: int,
    total_interactions: int,
) -> dict[str, float | int | str]:
    alive = _alive(individuals)
    bits = [individual.expressed_bit for individual in alive]
    thetas = [individual.theta for individual in alive]
    resources = [individual.birth_event_resource_score for individual in alive]
    p_one = float(np.mean(bits)) if bits else 0.0
    return {
        "scenario": params.scenario,
        "model": params.model,
        "seed": params.seed,
        "step": step,
        "alive_population": len(alive),
        "births_this_step": births_this_step,
        "deaths_this_step": deaths_this_step,
        "p_one_alive": p_one,
        "shannon_diversity_alive": _binary_entropy(p_one),
        "theta_mean_alive": float(np.mean(thetas)) if thetas else 0.0,
        "theta_std_alive": float(np.std(thetas)) if thetas else 0.0,
        "mean_resource_score_alive": float(np.mean(resources)) if resources else 0.0,
        "opposite_interaction_rate": (
            float(opposite_interactions / total_interactions) if total_interactions else 0.0
        ),
    }


def _summarize(
    *,
    params: PopulationParams,
    individuals: dict[int, Individual],
    birth_events: list[BirthEvent],
    trajectory: list[dict[str, float | int | str]],
    deaths: int,
    attempts: list[ReproductionAttempt],
) -> dict[str, float | int | str]:
    alive = _alive(individuals)
    parent_child_bits = [(event.parent_bit, event.child_bit) for event in birth_events]
    shuffle_rng = np.random.default_rng(params.seed + 1_000_003)
    shuffled_mi_values: list[float] = []
    if parent_child_bits:
        parent_bits = np.asarray([pair[0] for pair in parent_child_bits], dtype=int)
        child_bits = [pair[1] for pair in parent_child_bits]
        for _ in range(params.lineage_null_permutations):
            shuffled_parent_bits = parent_bits.copy()
            shuffle_rng.shuffle(shuffled_parent_bits)
            shuffled_pairs = list(zip(shuffled_parent_bits.tolist(), child_bits))
            shuffled_mi_values.append(_binary_mutual_information(shuffled_pairs))
    observed_mi = _binary_mutual_information(parent_child_bits)
    mutated_child_ids = [event.child_id for event in birth_events if event.mutation]
    transmitted_mutations = sum(
        1 for child_id in mutated_child_ids if individuals[child_id].birth_count > 0
    )
    attempt_births_by_bit: dict[int, list[float]] = {0: [], 1: []}
    for attempt in attempts:
        attempt_births_by_bit[attempt.parent_bit].append(float(attempt.born))
    if attempt_births_by_bit[0] and attempt_births_by_bit[1]:
        selection_gradient = float(
            np.mean(attempt_births_by_bit[1]) - np.mean(attempt_births_by_bit[0])
        )
    else:
        selection_gradient = 0.0
    exposure_values = [attempt.interaction_exposure for attempt in attempts]
    birth_values = [float(attempt.born) for attempt in attempts]
    interaction_birth_covariance = (
        float(np.cov(exposure_values, birth_values, ddof=0)[0, 1])
        if len(attempts) > 1
        else 0.0
    )
    parent_thetas = [event.parent_theta for event in birth_events]
    child_thetas = [event.child_theta for event in birth_events]
    birth_probabilities = [attempt.birth_probability for attempt in attempts]
    evaluated_attempts = [attempt for attempt in attempts if not attempt.capacity_blocked]
    capacity_blocked_attempts = [attempt for attempt in attempts if attempt.capacity_blocked]
    event_resources = [event.event_resource_score for event in birth_events]
    event_negativities = [event.event_negativity for event in birth_events]
    event_chsh = [event.event_chsh for event in birth_events]
    alive_bits = [individual.expressed_bit for individual in alive]
    alive_thetas = [individual.theta for individual in alive]
    alive_generations = [individual.generation for individual in alive]
    mean_population = float(np.mean([row["alive_population"] for row in trajectory]))
    mean_diversity = float(np.mean([row["shannon_diversity_alive"] for row in trajectory]))
    final_p_one = float(np.mean(alive_bits)) if alive_bits else 0.0
    return {
        "scenario": params.scenario,
        "model": params.model,
        "seed": params.seed,
        "steps": params.steps,
        "initial_population": params.initial_population,
        "carrying_capacity": params.carrying_capacity,
        "birth_count": len(birth_events),
        "death_count": deaths,
        "turnover_events": len(birth_events) + deaths,
        "reproduction_opportunity_count": len(attempts),
        "evaluated_birth_attempt_count": len(evaluated_attempts),
        "capacity_blocked_opportunity_count": len(capacity_blocked_attempts),
        "capacity_blocked_fraction": (
            float(len(capacity_blocked_attempts) / len(attempts)) if attempts else 0.0
        ),
        "final_population": len(alive),
        "mean_population": mean_population,
        "max_lineage_depth": max((individual.generation for individual in individuals.values()), default=0),
        "mean_lineage_depth_alive": float(np.mean(alive_generations)) if alive_generations else 0.0,
        "parent_offspring_agreement": (
            float(np.mean([left == right for left, right in parent_child_bits]))
            if parent_child_bits
            else 0.0
        ),
        "parent_offspring_mutual_information": observed_mi,
        "shuffled_lineage_mutual_information": (
            float(np.mean(shuffled_mi_values)) if shuffled_mi_values else 0.0
        ),
        "shuffled_lineage_mutual_information_p95": (
            float(np.percentile(shuffled_mi_values, 95)) if shuffled_mi_values else 0.0
        ),
        "lineage_mi_permutation_p_value": (
            float((1 + sum(value >= observed_mi for value in shuffled_mi_values)) / (len(shuffled_mi_values) + 1))
            if shuffled_mi_values
            else 1.0
        ),
        "theta_parent_child_correlation": _correlation(parent_thetas, child_thetas),
        "mutation_event_rate": (
            float(sum(event.mutation for event in birth_events) / len(birth_events))
            if birth_events
            else 0.0
        ),
        "transmitted_variant_rate": (
            float(transmitted_mutations / len(mutated_child_ids)) if mutated_child_ids else 0.0
        ),
        "selection_gradient_birth_rate_bit1_minus_bit0": selection_gradient,
        "mean_birth_probability": float(np.mean(birth_probabilities)) if birth_probabilities else 0.0,
        "mean_event_resource_score": float(np.mean(event_resources)) if event_resources else 0.0,
        "mean_event_negativity": float(np.mean(event_negativities)) if event_negativities else 0.0,
        "mean_event_chsh": float(np.mean(event_chsh)) if event_chsh else 0.0,
        "final_p_one_alive": final_p_one,
        "final_shannon_diversity_alive": _binary_entropy(final_p_one),
        "mean_shannon_diversity_alive": mean_diversity,
        "theta_mean_alive": float(np.mean(alive_thetas)) if alive_thetas else 0.0,
        "theta_std_alive": float(np.std(alive_thetas)) if alive_thetas else 0.0,
        "interaction_birth_covariance": interaction_birth_covariance,
    }


def run_population_benchmark(
    params: PopulationParams,
) -> dict[str, object]:
    """Run one population benchmark replicate.

    The returned dictionary contains a summary row, time-series rows,
    individual records, and birth-event records. The CLI writes the first two.
    """

    _validate_params(params)
    rng = np.random.default_rng(params.seed)
    individuals: dict[int, Individual] = {}
    next_id = 0
    for _ in range(params.initial_population):
        theta = _initial_theta(rng)
        expressed_bit, metrics = _sample_bit_from_theta(theta, params, rng)
        individuals[next_id] = Individual(
            id=next_id,
            parent_id=None,
            birth_step=0,
            death_step=None,
            generation=0,
            theta=theta,
            expressed_bit=expressed_bit,
            birth_event_negativity=metrics["negativity"],
            birth_event_chsh=metrics["chsh_max"],
            birth_event_resource_score=metrics["resource_score"],
            mutation_from_parent=False,
        )
        next_id += 1

    birth_events: list[BirthEvent] = []
    trajectory: list[dict[str, float | int | str]] = [
        _trajectory_row(
            params=params,
            step=0,
            individuals=individuals,
            births_this_step=0,
            deaths_this_step=0,
            opposite_interactions=0,
            total_interactions=0,
        )
    ]
    deaths = 0
    attempts: list[ReproductionAttempt] = []

    for step in range(1, params.steps + 1):
        alive_before_death = _alive(individuals)
        death_probability = _death_probability(len(alive_before_death), params)
        deaths_this_step = 0
        for individual in alive_before_death:
            if len(_alive(individuals)) <= 1:
                break
            if rng.random() < death_probability:
                individual.death_step = step
                deaths += 1
                deaths_this_step += 1

        alive_after_death = _alive(individuals)
        exposures, opposite, total_interactions = _interaction_exposures(
            alive_after_death,
            rng,
        )
        parent_order = alive_after_death[:]
        rng.shuffle(parent_order)
        births_this_step = 0
        for parent in parent_order:
            exposure, partner_id, partner_bit = exposures.get(parent.id, (0.0, None, None))
            birth_probability = _birth_probability(parent, params, exposure)
            if len(_alive(individuals)) >= params.carrying_capacity:
                attempts.append(
                    ReproductionAttempt(
                        step=step,
                        parent_id=parent.id,
                        parent_bit=parent.expressed_bit,
                        parent_theta=parent.theta,
                        parent_generation=parent.generation,
                        parent_age=step - parent.birth_step,
                        parent_resource_score=parent.birth_event_resource_score,
                        partner_id=partner_id,
                        partner_bit=partner_bit,
                        interaction_exposure=exposure,
                        birth_probability=birth_probability,
                        born=False,
                        capacity_blocked=True,
                        child_id=None,
                    )
                )
                continue
            born = bool(rng.random() < birth_probability)
            child_id: int | None = None
            if not born:
                attempts.append(
                    ReproductionAttempt(
                        step=step,
                        parent_id=parent.id,
                        parent_bit=parent.expressed_bit,
                        parent_theta=parent.theta,
                        parent_generation=parent.generation,
                        parent_age=step - parent.birth_step,
                        parent_resource_score=parent.birth_event_resource_score,
                        partner_id=partner_id,
                        partner_bit=partner_bit,
                        interaction_exposure=exposure,
                        birth_probability=birth_probability,
                        born=False,
                        capacity_blocked=False,
                        child_id=None,
                    )
                )
                continue

            parent.birth_count += 1
            if params.model == "no_inheritance":
                event_theta = _initial_theta(rng)
                child_theta = event_theta
                mutation = False
            else:
                event_theta = parent.theta
                child_theta = parent.theta
                mutation = bool(rng.random() < params.mutation_probability)
                if mutation:
                    child_theta = _clip_theta(child_theta + float(rng.normal(0.0, params.mutation_step)))
            child_bit, metrics = _sample_bit_from_theta(event_theta, params, rng)
            child_id = next_id
            next_id += 1
            individuals[child_id] = Individual(
                id=child_id,
                parent_id=parent.id,
                birth_step=step,
                death_step=None,
                generation=parent.generation + 1,
                theta=child_theta,
                expressed_bit=child_bit,
                birth_event_negativity=metrics["negativity"],
                birth_event_chsh=metrics["chsh_max"],
                birth_event_resource_score=metrics["resource_score"],
                mutation_from_parent=mutation,
            )
            birth_events.append(
                BirthEvent(
                    step=step,
                    parent_id=parent.id,
                    child_id=child_id,
                    parent_bit=parent.expressed_bit,
                    child_bit=child_bit,
                    parent_theta=parent.theta,
                    child_theta=child_theta,
                    mutation=mutation,
                    event_negativity=metrics["negativity"],
                    event_chsh=metrics["chsh_max"],
                    event_resource_score=metrics["resource_score"],
                    birth_probability=birth_probability,
                    interaction_exposure=exposure,
                )
            )
            attempts.append(
                ReproductionAttempt(
                    step=step,
                    parent_id=parent.id,
                    parent_bit=parent.expressed_bit,
                    parent_theta=parent.theta,
                    parent_generation=parent.generation,
                    parent_age=step - parent.birth_step,
                    parent_resource_score=parent.birth_event_resource_score,
                    partner_id=partner_id,
                    partner_bit=partner_bit,
                    interaction_exposure=exposure,
                    birth_probability=birth_probability,
                    born=True,
                    capacity_blocked=False,
                    child_id=child_id,
                )
            )
            births_this_step += 1

        trajectory.append(
            _trajectory_row(
                params=params,
                step=step,
                individuals=individuals,
                births_this_step=births_this_step,
                deaths_this_step=deaths_this_step,
                opposite_interactions=opposite,
                total_interactions=total_interactions,
            )
        )

    summary = _summarize(
        params=params,
        individuals=individuals,
        birth_events=birth_events,
        trajectory=trajectory,
        deaths=deaths,
        attempts=attempts,
    )
    return {
        "summary": summary,
        "trajectory": trajectory,
        "individuals": individuals,
        "birth_events": birth_events,
        "attempts": attempts,
    }
