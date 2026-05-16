"""Benchmark suite for QAL-inspired resource and population claims."""

from .baselines import baseline_catalog, evaluate_resource_kernel_baselines
from .classical import ClassicalParams, run_classical_event
from .certification import (
    binary_lineage_mutual_information,
    chsh_certificate,
    copy_agreement_certificate,
    finite_shot_lineage_certificate,
    lineage_count_certificate,
)
from .metrics import compute_metrics
from .population import PopulationParams, run_population_benchmark
from .quantum import QuantumParams, run_quantum_event
from .structured_population import StructuredPopulationParams, run_structured_population
from .submission import score_submission, verify_submission
from .tasks import task_catalog, task_by_id
from .workflows import (
    FiniteShotSubmissionConfig,
    PopulationSubmissionConfig,
    ResourceKernelSubmissionConfig,
    SamplingChallengeConfig,
    StructuredPopulationSweepConfig,
    submission_template,
    write_finite_shot_submission,
    write_population_submission,
    write_resource_kernel_submission,
    write_sampling_challenge_submission,
    write_structured_population_submission,
    write_submission_template,
)

__all__ = [
    "ClassicalParams",
    "FiniteShotSubmissionConfig",
    "PopulationParams",
    "PopulationSubmissionConfig",
    "QuantumParams",
    "ResourceKernelSubmissionConfig",
    "SamplingChallengeConfig",
    "StructuredPopulationParams",
    "StructuredPopulationSweepConfig",
    "baseline_catalog",
    "binary_lineage_mutual_information",
    "chsh_certificate",
    "compute_metrics",
    "copy_agreement_certificate",
    "evaluate_resource_kernel_baselines",
    "finite_shot_lineage_certificate",
    "lineage_count_certificate",
    "run_classical_event",
    "run_population_benchmark",
    "run_quantum_event",
    "run_structured_population",
    "score_submission",
    "submission_template",
    "task_by_id",
    "task_catalog",
    "verify_submission",
    "write_finite_shot_submission",
    "write_population_submission",
    "write_resource_kernel_submission",
    "write_sampling_challenge_submission",
    "write_structured_population_submission",
    "write_submission_template",
]
