# qalbench

`qalbench` is a compact benchmark package for claim-specific quantum
artificial-life (QAL) reporting. It includes:

- a two-qubit cloned-observable resource kernel
- a population-level lineage benchmark with event-level quantum diagnostics

The package is designed to separate artificial-life adequacy from
quantum-resource relevance. It does not claim quantum advantage and does not
simulate a full many-body population density matrix.

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

The repository wrappers in `scripts/` call the same package-native entry points.
