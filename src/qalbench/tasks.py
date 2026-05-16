"""Task catalog for the general QALBench suite.

The catalog is deliberately declarative. Existing artifact verifiers remain
tied to the released resource-kernel and population CSVs, while this module
defines the reusable task families that submissions can target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


TaskTier = Literal["T1", "T2", "T3", "T4", "T5", "T6"]
ClaimAxis = Literal[
    "artificial_life",
    "quantum_resource",
    "computational_nonclassicality",
]


@dataclass(frozen=True)
class TaskSpec:
    """A machine-readable benchmark task definition."""

    task_id: str
    tier: TaskTier
    family: str
    title: str
    minimum_supported_claim: str
    claim_axes: tuple[ClaimAxis, ...]
    required_artifacts: tuple[str, ...]
    required_controls: tuple[str, ...]
    metrics: tuple[str, ...]
    baseline_families: tuple[str, ...] = ()
    hardware_ready: bool = False
    scalable_challenge: bool = False
    status: str = "available"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


TASK_SPECS: tuple[TaskSpec, ...] = (
    TaskSpec(
        task_id="t1_basis_inheritance_kernel",
        tier="T1",
        family="cloned_observable",
        title="Basis-Inheritance Kernel",
        minimum_supported_claim=(
            "A named computational-basis observable is copied or transmitted "
            "for the stated event family."
        ),
        claim_axes=("artificial_life",),
        required_artifacts=(
            "basis_counts",
            "event_parameters",
            "implementation_manifest",
        ),
        required_controls=("dephased_density", "diagonal_markov"),
        metrics=("z_expectations", "copy_agreement", "basis_probabilities"),
        baseline_families=("diagonal_markov", "dephased_density"),
        status="implemented",
    ),
    TaskSpec(
        task_id="t1_mutation_channel_kernel",
        tier="T1",
        family="mutation_channel",
        title="Basis Mutation Channel",
        minimum_supported_claim=(
            "A declared mutation or perturbation channel changes a named "
            "basis trait at the reported rate."
        ),
        claim_axes=("artificial_life",),
        required_artifacts=(
            "basis_counts",
            "mutation_records",
            "event_parameters",
        ),
        required_controls=("readout_error_null", "no_mutation_control"),
        metrics=("mutation_rate", "trait_transition_matrix", "confidence_interval"),
        baseline_families=("diagonal_markov", "readout_error"),
    ),
    TaskSpec(
        task_id="t2_state_resource_diagnostic",
        tier="T2",
        family="state_resource",
        title="State-Resource Diagnostic",
        minimum_supported_claim=(
            "A specified state resource is present under the stated diagnostic "
            "assumptions."
        ),
        claim_axes=("quantum_resource",),
        required_artifacts=(
            "density_or_witness_estimates",
            "resource_metrics",
            "diagnostic_assumptions",
        ),
        required_controls=("dephased_density", "separable_product"),
        metrics=("coherence_l1", "negativity", "concurrence", "chsh_max"),
        baseline_families=("dephased_density", "separable_product", "maximally_mixed"),
        status="implemented",
    ),
    TaskSpec(
        task_id="t2_process_resource_diagnostic",
        tier="T2",
        family="process_resource",
        title="Process-Resource Diagnostic",
        minimum_supported_claim=(
            "A channel or update rule preserves a declared process-level "
            "quantum resource under stated assumptions."
        ),
        claim_axes=("quantum_resource",),
        required_artifacts=(
            "process_description",
            "input_ensemble",
            "resource_metrics",
        ),
        required_controls=("entanglement_breaking", "resource_destroying_channel"),
        metrics=("channel_witness", "resource_survival", "control_gap"),
        baseline_families=("entanglement_breaking", "resource_destroying_channel"),
    ),
    TaskSpec(
        task_id="t3_population_lineage_audit",
        tier="T3",
        family="population_lineage",
        title="Population Lineage Audit",
        minimum_supported_claim=(
            "Explicit individuals or registers support the claimed lineage, "
            "turnover, and inherited-state bookkeeping."
        ),
        claim_axes=("artificial_life",),
        required_artifacts=(
            "individual_records",
            "birth_event_records",
            "population_timeseries",
            "lineage_nulls",
        ),
        required_controls=("no_inheritance", "shuffled_lineage"),
        metrics=(
            "lineage_depth",
            "parent_offspring_mutual_information",
            "inherited_state_correlation",
            "diversity",
            "turnover",
        ),
        baseline_families=("no_inheritance", "neutral_selection"),
        status="implemented",
    ),
    TaskSpec(
        task_id="t3_interaction_selection_audit",
        tier="T3",
        family="interaction_selection",
        title="Interaction and Selection Audit",
        minimum_supported_claim=(
            "Interaction or selection labels are backed by recorded contacts, "
            "opportunities, and outcome associations."
        ),
        claim_axes=("artificial_life",),
        required_artifacts=(
            "interaction_records",
            "opportunity_records",
            "selection_rule",
            "population_timeseries",
        ),
        required_controls=("randomized_neighbor", "neutral_selection"),
        metrics=("selection_gradient", "interaction_outcome_covariance", "diversity"),
        baseline_families=("neutral_selection", "randomized_neighbor"),
    ),
    TaskSpec(
        task_id="t4_resource_coupled_outcome",
        tier="T4",
        family="resource_coupled_population",
        title="Resource-Coupled Outcome Control",
        minimum_supported_claim=(
            "A resource-derived event variable changes a declared population "
            "outcome and the change is removed by matched controls."
        ),
        claim_axes=("artificial_life", "quantum_resource"),
        required_artifacts=(
            "event_resource_records",
            "opportunity_records",
            "population_outcomes",
            "effect_size_summary",
        ),
        required_controls=("dephased_resource_ablation", "diagonal_markov", "no_inheritance"),
        metrics=("resource_control_gap", "birth_or_survival_effect_size", "transmission_metric"),
        baseline_families=("dephased_density", "diagonal_markov", "no_inheritance"),
        status="implemented_positive_control",
    ),
    TaskSpec(
        task_id="t4_transmission_breaking_resource_control",
        tier="T4",
        family="resource_lineage_separation",
        title="Transmission-Breaking Resource Control",
        minimum_supported_claim=(
            "Resource positivity is separated from inherited artificial-life "
            "transmission."
        ),
        claim_axes=("artificial_life", "quantum_resource"),
        required_artifacts=(
            "event_resource_records",
            "individual_records",
            "birth_event_records",
            "lineage_nulls",
        ),
        required_controls=("no_inheritance", "resource_preserving_null"),
        metrics=("resource_score", "inherited_state_correlation", "lineage_mi"),
        baseline_families=("no_inheritance", "dephased_density"),
        status="implemented",
    ),
    TaskSpec(
        task_id="t5_finite_shot_resource_certificate",
        tier="T5",
        family="hardware_resource_certificate",
        title="Finite-Shot Resource Certificate",
        minimum_supported_claim=(
            "The claimed resource survives finite sampling under a stated "
            "hardware, readout, and witness model."
        ),
        claim_axes=("quantum_resource",),
        required_artifacts=(
            "shot_counts",
            "witness_estimates",
            "calibration_record",
            "confidence_intervals",
        ),
        required_controls=("finite_shot_dephased_control", "finite_shot_classical_control"),
        metrics=("witness_lower_bound", "chsh_lower_bound", "copy_agreement_interval"),
        baseline_families=("dephased_density", "separable_product", "readout_error"),
        hardware_ready=True,
    ),
    TaskSpec(
        task_id="t5_finite_shot_lineage_certificate",
        tier="T5",
        family="hardware_lineage_certificate",
        title="Finite-Shot Lineage Certificate",
        minimum_supported_claim=(
            "Inheritance, mutation, or interaction statistics survive finite "
            "shot uncertainty and hardware-specific readout treatment."
        ),
        claim_axes=("artificial_life",),
        required_artifacts=(
            "shot_counts",
            "individual_or_register_records",
            "lineage_statistics",
            "calibration_record",
            "confidence_intervals",
        ),
        required_controls=("finite_shot_no_inheritance", "finite_shot_shuffled_lineage"),
        metrics=("lineage_mi_interval", "mutation_rate_interval", "copy_agreement_interval"),
        baseline_families=("no_inheritance", "readout_error", "neutral_selection"),
        hardware_ready=True,
    ),
    TaskSpec(
        task_id="t6_sampling_nonclassicality_challenge",
        tier="T6",
        family="sampling_challenge",
        title="Scalable QAL Sampling Challenge",
        minimum_supported_claim=(
            "A scalable QAL sampling task challenges declared efficient "
            "classical simulator families at a stated error tolerance."
        ),
        claim_axes=(
            "artificial_life",
            "quantum_resource",
            "computational_nonclassicality",
        ),
        required_artifacts=(
            "scaling_spec",
            "sampler_output",
            "shot_budget",
            "classical_baseline_results",
            "verification_protocol",
            "nonclassicality_evidence",
        ),
        required_controls=("resource_destroying_control", "problem_specific_classical_null"),
        metrics=("sample_distance", "resource_depth", "classical_runtime_scaling"),
        baseline_families=(
            "stabilizer",
            "low_magic",
            "tensor_network",
            "classical_shadow",
            "problem_specific",
        ),
        scalable_challenge=True,
    ),
    TaskSpec(
        task_id="t6_simulator_scaling_challenge",
        tier="T6",
        family="simulator_scaling",
        title="Classical-Simulator Scaling Challenge",
        minimum_supported_claim=(
            "A family of structurally quantum population dynamics exceeds "
            "specified simulator baselines under a documented scaling study."
        ),
        claim_axes=("quantum_resource", "computational_nonclassicality"),
        required_artifacts=(
            "scaling_spec",
            "resource_metrics",
            "classical_baseline_results",
            "reproducibility_manifest",
            "population_timeseries",
            "verification_protocol",
            "nonclassicality_evidence",
        ),
        required_controls=("dephased_scaling_control", "low_entanglement_control"),
        metrics=("state_space_size", "entanglement_growth", "baseline_gap"),
        baseline_families=("exact_density_matrix", "tensor_network", "stabilizer", "mean_field"),
        scalable_challenge=True,
    ),
)


def task_catalog() -> tuple[TaskSpec, ...]:
    """Return all registered task definitions."""

    return TASK_SPECS


def task_catalog_dicts() -> list[dict[str, object]]:
    """Return JSON-serializable task definitions."""

    return [task.as_dict() for task in TASK_SPECS]


def task_by_id(task_id: str) -> TaskSpec:
    """Return a task definition by ID or raise ``KeyError``."""

    for task in TASK_SPECS:
        if task.task_id == task_id:
            return task
    raise KeyError(f"unknown QALBench task_id {task_id!r}")


def tasks_by_tier() -> dict[TaskTier, tuple[TaskSpec, ...]]:
    """Group registered tasks by tier."""

    grouped: dict[TaskTier, list[TaskSpec]] = {
        "T1": [],
        "T2": [],
        "T3": [],
        "T4": [],
        "T5": [],
        "T6": [],
    }
    for task in TASK_SPECS:
        grouped[task.tier].append(task)
    return {tier: tuple(tasks) for tier, tasks in grouped.items()}
