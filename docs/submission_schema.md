# QALBench Submission Schema

QALBench submissions are JSON manifests plus referenced artifact files. The
manifest is intentionally task-centered: it names one task from
`qalbench-suite catalog`, declares the claim axes, lists controls and baselines,
and records paths plus SHA-256 hashes for every required artifact.

The canonical v0.2 CSV artifacts are still verified by `qalbench-verify` and
`qalbench-population-verify`. The schema below is for the general suite layer
used by `qalbench-suite verify-submission` and `qalbench-suite score-submission`.

## Manifest Fields

Required fields:

- `schema_version`: integer manifest schema version, currently `1`
- `task_id`: task identifier from the task catalog
- `claim_statement`: explicit statement of the artificial-life,
  quantum-resource, or computational claim being submitted
- `claim_axes`: subset of `artificial_life`, `quantum_resource`, and
  `computational_nonclassicality`
- `controls`: task-specific controls or nulls
- `baselines`: declared resource or simulator baseline families
- `artifacts`: object keyed by artifact name

Each artifact entry should include:

```json
{
  "path": "relative/or/absolute/path.json",
  "sha256": "hex-encoded-sha256"
}
```

`verify-submission` checks both file integrity and task-specific artifact
content. For the implemented reference tasks, empty or malformed artifacts fail
verification even when their SHA-256 values match the manifest.

T5 hardware-ready tasks additionally require `shot_budget > 0`.

T6 scalable tasks additionally require:

- `scaling_variable`: the size parameter, such as `site_count`, depth, or
  population size
- `allowed_error`: the distance, tolerance, or acceptance rule
- simulator baselines listed in the task catalog

Generate a starter manifest with:

```bash
qalbench-suite template t5_finite_shot_resource_certificate --output manifest.json
```

## Task Families

The suite catalog currently defines these auditable task IDs:

- `t1_basis_inheritance_kernel`
- `t1_mutation_channel_kernel`
- `t2_state_resource_diagnostic`
- `t2_process_resource_diagnostic`
- `t3_population_lineage_audit`
- `t3_interaction_selection_audit`
- `t4_resource_coupled_outcome`
- `t4_transmission_breaking_resource_control`
- `t5_finite_shot_resource_certificate`
- `t5_finite_shot_lineage_certificate`
- `t6_sampling_nonclassicality_challenge`
- `t6_simulator_scaling_challenge`

## Built-In Reference Packages

QALBench can write compact verifiable packages for the implemented reference
families:

```bash
qalbench-suite write-resource-kernel-submission --task-id t1_basis_inheritance_kernel --output-dir t1-run
qalbench-suite write-resource-kernel-submission --task-id t1_mutation_channel_kernel --output-dir t1-mutation-run
qalbench-suite write-resource-kernel-submission --task-id t2_state_resource_diagnostic --output-dir t2-run
qalbench-suite write-resource-kernel-submission --task-id t2_process_resource_diagnostic --output-dir t2-process-run
qalbench-suite write-population-submission --task-id t3_population_lineage_audit --output-dir t3-run
qalbench-suite write-population-submission --task-id t3_interaction_selection_audit --output-dir t3-interaction-run
qalbench-suite write-population-submission --task-id t4_resource_coupled_outcome --output-dir t4-run
qalbench-suite write-population-submission --task-id t4_transmission_breaking_resource_control --output-dir t4-transmission-run
```

These commands write a `submission_manifest.json` plus the task-required
artifacts. They are small reference packages, not replacements for the
canonical v0.2 release artifacts.

The T1 basis package reports computational-basis probabilities and exact
dephased and diagonal Markov controls. The T1 mutation package reports a
stochastic descendant perturbation channel with a no-mutation control. The T2
state package reports exact density-matrix resource diagnostics with dephased,
separable, and mixed-state baselines. The T2 process package reports resource
survival across a small input ensemble. The T3 lineage package reports explicit
individual, birth-event, time-series, shuffled-lineage, and no-inheritance
records. The T3 interaction package reports contact exposures, opportunities,
selection-rule fields, and a neutral control. The T4 resource-coupled package
reports a resource-coupled positive control with quantum, dephased, classical,
and no-inheritance population runs. The T4 transmission-breaking package
separates resource-positive event records from inherited-state transmission by
using a resource-preserving no-inheritance control.

The implemented content checks include:

- T1 basis: computational-basis probability objects for quantum, dephased, and
  classical Markov rows; probabilities must be numeric and normalized
- T1 mutation: T1 basis checks plus nonempty mutation rows and declared flip
  probabilities
- T2 state: exact 4x4 density matrix encoding, resource metrics for quantum,
  dephased, and separable-product rows, and nonempty diagnostic assumptions
- T2 process: process description, normalized input ensemble, and resource
  survival metric rows
- T3 lineage: nonempty individual, birth-event, and time-series rows plus
  lineage-null summaries
- T3 interaction: interaction records, opportunity rows, selection-rule fields,
  and basis/neutral time series
- T4 resource-coupled: event-resource, opportunity, outcome, and effect-size
  artifacts covering quantum, dephased, classical, and no-inheritance controls
- T4 transmission-breaking: event-resource, individual, birth-event, and
  lineage-null artifacts for quantum and resource-preserving no-inheritance
  controls

## T5 Count Artifacts

Finite-shot resource tasks require a `shot_counts` artifact and a
`confidence_intervals` artifact.

For basis or copy-agreement counts:

```json
{
  "counts": {
    "00": 480,
    "01": 20,
    "10": 10,
    "11": 490
  }
}
```

For CHSH counts:

```json
{
  "setting_counts": {
    "ab": {"00": 4500, "01": 500, "10": 500, "11": 4500},
    "ab_prime": {"00": 4500, "01": 500, "10": 500, "11": 4500},
    "a_prime_b": {"00": 4500, "01": 500, "10": 500, "11": 4500},
    "a_prime_b_prime": {"00": 500, "01": 4500, "10": 4500, "11": 500}
  }
}
```

Create a finite-shot certificate with:

```bash
qalbench-suite certify-counts counts.json --kind chsh --output confidence_intervals.json
qalbench-suite certify-counts lineage-counts.json --kind lineage --output lineage_intervals.json
```

Create a complete T5 finite-shot package with manifest, counts, witness,
calibration, and confidence-interval artifacts:

```bash
qalbench-suite write-finite-shot-submission counts.json --kind chsh --output-dir finite-shot-run
qalbench-suite verify-submission finite-shot-run/submission_manifest.json
qalbench-suite score-submission finite-shot-run/submission_manifest.json
```

Resource-certificate verification recomputes submitted copy-agreement,
histogram, or CHSH intervals from `shot_counts`; stale intervals fail even when
their hashes match. `calibration_record` must also declare a readout model. The
minimal built-in package uses an explicit ideal-readout assumption:

```json
{
  "readout_mitigation": "none",
  "readout_error_model": {
    "type": "ideal_readout_assumption",
    "mitigation_applied": false,
    "calibration_shots": 0
  },
  "assumptions": ["counts are accepted as submitted"]
}
```

For mitigated submissions, `readout_mitigation` is
`inverse_confusion_matrix`, and `readout_error_model` must include unique
`outcomes`, a row-stochastic `confusion_matrix`, a square numeric
`inverse_confusion_matrix`, and positive `calibration_shots`.

Lineage count payloads use observed and finite-shot control groups:

```json
{
  "lineage_counts": {
    "observed": {"00": 480, "01": 20, "10": 10, "11": 490},
    "no_inheritance": {"00": 250, "01": 250, "10": 250, "11": 250},
    "shuffled_lineage": {"00": 260, "01": 240, "10": 240, "11": 260}
  }
}
```

Create a complete T5 lineage package with:

```bash
qalbench-suite write-finite-shot-submission lineage-counts.json --kind lineage --task-id t5_finite_shot_lineage_certificate --output-dir finite-lineage-run
```

The CHSH certificate reports an estimate, a conservative lower bound using a
union bound across settings, and whether the lower bound exceeds the local
bound. This is a finite-shot statistical certificate under the stated count and
measurement assumptions, not a loophole-free Bell-test claim by itself.
Lineage certificates report Wilson intervals for copy agreement and mutation
rate, a bounded plug-in interval for binary lineage mutual information, and an
inheritance flag that requires the observed copy-agreement lower bound to
exceed the control upper bounds. Lineage certificate verification recomputes
those intervals and control comparisons from `shot_counts`, and validates the
same `calibration_record` schema used by finite-shot resource certificates.

## T6 Simulator Artifacts

T6 tasks require a `scaling_spec` artifact and a
`classical_baseline_results` artifact. Sampling challenges also require
`sampler_output`, `shot_budget`, `verification_protocol`, and
`nonclassicality_evidence` artifacts.

Minimal `scaling_spec`:

```json
{
  "scaling_variable": "site_count",
  "sizes": [2, 3, 4],
  "allowed_error": "total variation distance <= 0.05"
}
```

Minimal `classical_baseline_results`:

```json
{
  "rows": [
    {
      "baseline_id": "exact_density_matrix",
      "site_count": 2,
      "runtime_seconds": 0.01,
      "error_metric": "reference",
      "error_value": 0.0
    },
    {
      "baseline_id": "tensor_network",
      "site_count": 2,
      "runtime_seconds": 0.02,
      "error_metric": "total_variation",
      "error_value": 0.001
    }
  ]
}
```

The built-in structured-population workflow writes a small exact-register T6
reference package:

```bash
qalbench-suite run-structured-population --output-dir structured-run --site-counts 2,3,4
qalbench-suite verify-submission structured-run/submission_manifest.json --root structured-run
qalbench-suite score-submission structured-run/submission_manifest.json --root structured-run
```

The generated package is a reference and workflow check. It includes
`population_timeseries`, `verification_protocol`, and
`nonclassicality_evidence` artifacts, but it is not a
computational-nonclassicality claim until participant-supplied simulator
baselines, error metrics, and compute disclosures support that claim. The
verifier checks that structured-population resource metrics agree with final
coherent/dephased time-series rows.

For the built-in structured-population reference package, QALBench writes
completed small-system baselines where they are honest:

- `exact_density_matrix`: dense exact reference
- `mean_field`: independent-site product distribution matched to exact one-site
  marginals, scored by computational-basis total variation distance
- `stabilizer`: exact only when the configured circuit is stabilizer-compatible
- `tensor_network`: exact small-system Schmidt-rank diagnostic

These rows make simulator evidence auditable, but they remain small-system
reference baselines. They do not establish asymptotic computational
nonclassicality without a scaling study and disclosed compute budgets.

The built-in sampling-challenge workflow writes a compact T6 package that
exercises the sampling schema and all required simulator baseline rows:

```bash
qalbench-suite write-sampling-challenge-submission --output-dir sampling-run --sizes 2,3,4 --shots-per-size 256
qalbench-suite verify-submission sampling-run/submission_manifest.json --root sampling-run
qalbench-suite score-submission sampling-run/submission_manifest.json --root sampling-run
```

Its `nonclassicality_evidence.json` explicitly marks
`nonclassicality_claim_supported` as false. QALBench therefore gives the
reference sampling package zero computational-nonclassicality credit even
though its artifacts and baselines verify. Nonclassicality credit requires
passing evidence rows with size coverage, simulator baseline completion,
accepted sample distance, and compute-budget disclosure.

If a submission sets `nonclassicality_claim_supported` to true, the evidence
must also declare:

- `claim_type`: `baseline_failure`
- `classical_failure_criterion`: the rule used to count simulator failure
- `minimum_sizes`: at least 3
- `compute_budget`: numeric `total_wall_time_seconds` and `max_memory_mb`
- per-size rows with `wall_time_seconds`, `memory_mb`, accepted sample
  distance, completed baselines, `baseline_failures` matching simulator rows
  whose errors exceed `allowed_error`, and `best_classical_error` matching the
  submitted baseline rows

This keeps computational-nonclassicality scoring tied to explicit, auditable
evidence rather than manifest completeness.
