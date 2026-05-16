# qalbench

`qalbench` is a compact benchmark package for claim-specific quantum
artificial-life (QAL) reporting. It includes:

- a two-qubit cloned-observable resource kernel
- a population-level lineage benchmark with event-level quantum diagnostics
- a reusable T1-T6 suite layer for task definitions, finite-shot
  certification, baselines, structured quantum population registers, and
  submission verification/scoring

The package is designed to separate artificial-life adequacy from
quantum-resource relevance. It does not claim quantum advantage and does not
simulate a full many-body population density matrix.

## Suite Layer

The suite layer is separate from the canonical artifact verifiers. It provides
reusable contracts for external or future submissions:

- `qalbench.tasks`: machine-readable task families across T1-T6
- `qalbench.certification`: finite-shot intervals, copy-agreement estimates,
  Pauli-correlation estimates, CHSH certificates, and declared readout
  mitigation
- `qalbench.baselines`: quantum-resource and classical-simulator baseline
  registry plus resource-kernel baseline evaluators
- `qalbench.structured_population`: small exact fixed-site quantum population
  dynamics with one joint density matrix
- `qalbench.submission`: JSON manifest verification and independent scoring
  for artificial-life adequacy, quantum-resource relevance, and computational
  nonclassicality

The suite command prints the task/baseline catalog and verifies or scores
submission manifests:

```bash
qalbench-suite catalog --include-baselines
qalbench-suite template t5_finite_shot_resource_certificate --output manifest.json
qalbench-suite write-resource-kernel-submission --task-id t2_state_resource_diagnostic --output-dir resource-run
qalbench-suite write-population-submission --task-id t3_population_lineage_audit --output-dir population-run
qalbench-suite certify-counts counts.json --kind chsh --output certificate.json
qalbench-suite write-finite-shot-submission counts.json --kind chsh --output-dir finite-shot-run
qalbench-suite write-finite-shot-submission lineage-counts.json --kind lineage --task-id t5_finite_shot_lineage_certificate --output-dir finite-lineage-run
qalbench-suite run-structured-population --output-dir structured-run --site-counts 2,3,4
qalbench-suite write-sampling-challenge-submission --output-dir sampling-run --sizes 2,3,4
qalbench-suite verify-submission manifest.json --root artifacts/
qalbench-suite score-submission manifest.json --root artifacts/
```

The built-in resource writer supports `t1_basis_inheritance_kernel`,
`t1_mutation_channel_kernel`, `t2_state_resource_diagnostic`, and
`t2_process_resource_diagnostic`. The population writer supports
`t3_population_lineage_audit`, `t3_interaction_selection_audit`,
`t4_resource_coupled_outcome`, and
`t4_transmission_breaking_resource_control`.

## Resource Kernel

The quantum event uses two qubits:

- qubit 0: inherited precursor register
- qubit 1: descendant register

One event applies:

1. genotype preparation `cos(theta/2)|0> + exp(i phi) sin(theta/2)|1>`
2. descendant reset to `|0>`
3. cloned-observable copying by `CNOT(genotype -> descendant)`
4. optional stochastic local perturbation/noise channel: with probability
   `local_perturbation_probability`, apply
   `Ry(local_perturbation_angle)` to the descendant
5. optional `ZZ` phase entangler `exp(-i interaction_angle ZxZ / 2)`
6. amplitude damping on the descendant as a fixed-register loss channel
7. optional dephasing to make a decohered comparison state

The matched classical baseline is a Markov model over bitstrings
`00, 01, 10, 11`. It uses the same initial genotype basis probabilities,
deterministic CNOT observable copying, local perturbation bit flips with
probability
`local_perturbation_probability * sin^2(local_perturbation_angle/2)`,
and loss as stochastic descendant reset-to-zero with probability
`damping_probability`.

The resource-kernel metrics include:

- Z expectation values for precursor and descendant
- computational-basis probabilities and Z copy agreement
- density-matrix `l1` coherence
- reported X/Y Pauli correlations `XX`, `XY`, `YX`, and `YY`
- concurrence
- negativity and minimum partial-transpose eigenvalue
- Horodecki maximum CHSH value
- von Neumann entropy
- trace and purity

## Population Benchmark

The population benchmark keeps explicit individual records rather than only
register states. Each individual has:

- stable ID and parent ID
- birth and death/reset step
- lineage generation
- inherited genotype angle `theta`
- expressed computational-basis phenotype bit
- event-level resource diagnostics from its birth

The default population sweep runs three scenarios:

- `basis_selection`: reproduction rewards expressed bit `1`
- `resource_selection`: reproduction rewards event-level resource score
- `neutral`: reproduction has no phenotype, resource, or interaction selection
  term

Each scenario is run under four controls:

- `quantum`: coherent two-qubit event diagnostics
- `dephased`: stepwise computational-basis dephasing after event operations
- `classical`: matched diagonal Markov event distribution
- `no_inheritance`: quantum event diagnostics but parent-to-child inherited
  `theta` transmission is broken by independent draws

Population outputs include final and mean population size, births, deaths,
turnover, maximum lineage depth, parent-offspring mutual information,
shuffled-lineage mutual information, parent-child genotype-parameter
correlation, inherited-model mutation and transmitted-variant rates,
opportunity-level selection gradients, diversity, reproduction-opportunity
records, capacity-blocked opportunity records, and interaction/birth covariance.

## Commands

Resource-kernel artifact:

```bash
qalbench-sweep
qalbench-verify
```

Population artifact:

```bash
qalbench-population-sweep
qalbench-population-verify
```

General suite workflow:

```bash
qalbench-suite catalog --include-baselines
qalbench-suite template t5_finite_shot_resource_certificate --output manifest.json
qalbench-suite write-resource-kernel-submission --task-id t2_state_resource_diagnostic --output-dir resource-run
qalbench-suite write-population-submission --task-id t3_population_lineage_audit --output-dir population-run
qalbench-suite certify-counts counts.json --kind chsh --output certificate.json
qalbench-suite write-finite-shot-submission counts.json --kind chsh --output-dir finite-shot-run
qalbench-suite write-finite-shot-submission lineage-counts.json --kind lineage --task-id t5_finite_shot_lineage_certificate --output-dir finite-lineage-run
qalbench-suite run-structured-population --output-dir structured-run --site-counts 2,3,4
qalbench-suite write-sampling-challenge-submission --output-dir sampling-run --sizes 2,3,4
qalbench-suite verify-submission manifest.json --root artifacts/
qalbench-suite score-submission manifest.json --root artifacts/
```

The same T1-T4 task IDs listed above are valid for the resource and population
submission writers.

The repository wrappers in `scripts/` call the same package-native entry points.
The general submission schema is documented in `docs/submission_schema.md` in
the source repository.
