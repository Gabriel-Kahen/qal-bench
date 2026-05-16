# QALBench Suite Completion Audit

This audit records why the suite layer is a complete benchmark-infrastructure
deliverable, while remaining honest that it does not itself prove external
hardware performance or computational nonclassicality.

## Objective Requirements

- Multiple QAL task families across T1-T6:
  `src/qalbench/tasks.py` defines 12 task IDs spanning T1 through T6, with
  exact catalog coverage tested in `tests/test_suite_contract.py`.
- Hardware-ready finite-shot certification:
  `src/qalbench/certification.py` implements copy-agreement, histogram,
  Pauli-correlation, CHSH, and lineage certificates. T5 resource and lineage
  verifiers recompute submitted intervals from `shot_counts`, validate
  calibration/readout records, and reject stale or malformed certificates.
- Expanded quantum-resource and classical-simulator baselines:
  `src/qalbench/baselines.py` defines resource, hardware-null, and simulator
  baseline families. T1/T2 workflows emit dephased, Markov, separable, and
  mixed-state resource controls. T6 workflows require declared simulator
  baselines and completed per-size baseline rows.
- Structurally quantum population dynamics:
  `src/qalbench/structured_population.py` evolves fixed population sites as
  one joint density matrix. The T6 structured-population workflow emits
  coherent/dephased trajectories and simulator diagnostics, and verification
  checks resource metrics against the final time-series rows.
- Reusable submission, verification, and scoring workflows:
  `src/qalbench/workflows.py`, `src/qalbench/submission.py`, and
  `src/qalbench/suite.py` implement templates, package writers, manifest
  verification, independent axis scoring, and CLI commands for resource,
  population, finite-shot, structured-population, and sampling-challenge
  submissions.
- Independent auditability of artificial-life adequacy, quantum-resource
  relevance, and computational nonclassicality:
  Scoring zeroes undeclared or unsupported axes. Computational
  nonclassicality requires explicit passing evidence with simulator failures,
  compute-budget disclosure, size coverage, and consistency with submitted
  baseline rows; artifact presence alone scores zero.

## Verification Evidence

The suite contract tests cover:

- exact task catalog coverage and public API exports
- finite-shot resource and lineage certificate generation
- T1-T4 package writers and task-specific validators
- T5 resource and lineage workflows, stale-interval rejection, and calibration
  rejection
- T6 structured-population and sampling-challenge workflows
- unsupported-axis overclaim rejection
- schema-version enforcement
- nonclassicality evidence rejection and positive synthetic scoring
- late-row artifact validation and docs/catalog synchronization

Canonical artifact verifiers remain separate from the suite layer. They are
run as read-only audit checks to confirm the suite additions did not weaken the
released v0.2 resource or population artifact contracts.

## Boundary

QALBench now provides auditable benchmark infrastructure for T5 and T6 claims.
It does not claim that the included reference packages themselves demonstrate
loophole-free hardware certification or quantum advantage. External submitters
must provide hardware counts, calibration records, simulator results,
compute-budget disclosures, and nonclassicality evidence for those stronger
claims to score on the corresponding axes.
