# QALBench

This repository contains the manuscript and benchmark code for:

**QALBench: An Audit Protocol and Reference Implementation for Quantum Artificial-Life Claims**

QALBench separates three kinds of quantum artificial-life claims:

- artificial-life adequacy: individuals, parent-offspring lineages, heritable
  variation, turnover, interaction, selection, and diversity
- quantum-resource relevance: coherence, entanglement, CHSH potential, and
  resource-destroying controls
- computational nonclassicality: a stronger axis specified as a future
  benchmark requirement, not claimed by the current artifact

The repository contains two implemented benchmark modules.

The resource-kernel module is a QAL-inspired cloned-observable inheritance
event. It compares:

- a full density-matrix quantum simulation
- a dephased quantum simulation
- a matched classical Markov baseline

The population module adds explicit individuals, parent IDs, inherited genotype
parameters, expressed basis phenotypes, births, deaths, mutations, interaction
exposure, selection scenarios, no-inheritance controls, shuffled-lineage nulls,
capacity-blocked reproduction-opportunity records, and quantum/dephased/classical
event-resource ablations. It is a population benchmark with event-level quantum
diagnostics, not an exact many-body quantum state simulation of the whole
population.

## Reproduce Results

Create an environment and install dependencies. For reproduction of the
checked-in scientific artifacts, use the pinned dependency file:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e .
```

The pinned file records the direct and transitive Python packages used for the
included CSV and figure artifacts, with SHA-256 hashes for the wheel set used
on the recorded macOS arm64/Python 3.12 environment. It does not provision
Python itself. The artifacts were produced with Python 3.12.13,
NumPy 2.4.4, Matplotlib 3.10.9, and Tectonic 0.16.9. Regenerated scientific
CSV output should match the canonical CSV hashes. If `python3.12` is not
available under that name, create the virtual environment with another Python
3.12 interpreter explicitly. PNG byte hashes are checked
for the packaged canonical figures under the recorded rendering stack; exact
PNG byte reproducibility is platform-sensitive because Matplotlib backend,
FreeType, and font selection affect rendered files. On a different renderer,
run the verifier commands with `--skip-figures` and inspect regenerated figures
visually. Run metadata includes timestamps, platform strings, and command
records, so it is intentionally not byte-for-byte stable.

For a portable development environment, use the broader major-version bounds
instead:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
```

Verify the checked-in artifacts without regenerating them:

```bash
scripts/verify_release_artifacts.sh
```

That command runs the unit tests and checks both the resource-kernel and
population artifacts using `.venv/bin/python` when that environment exists. Use
it for artifact audit before mutating the results tree. Running bare `python3`
from the system shell is not expected to work unless that interpreter has the
pinned dependencies installed.

Run the full regeneration pipeline:

```bash
scripts/reproduce_all.sh
```

This command regenerates `results/` and `paper/figures/` before verification, so
it is a reproduction workflow rather than a read-only audit of the shipped
artifacts.

If `tectonic` is not on your `PATH`, set `TECTONIC=/path/to/tectonic` before
running the script. Tectonic is an external executable, not installed by the
Python requirements files. The reproduction script checks for Tectonic 0.16.9
by default; set `EXPECTED_TECTONIC_VERSION` only when intentionally rebuilding
with a different engine version.

Or run the steps separately. First run the unit tests:

```bash
.venv/bin/python -m unittest discover -s tests
```

Then run the sweeps and regenerate figures:

```bash
.venv/bin/python scripts/run_qalbench_sweeps.py
.venv/bin/python scripts/run_qalbench_population.py
```

Verify the numerical artifacts underlying the values quoted in the paper:

```bash
.venv/bin/python scripts/verify_qalbench_results.py
.venv/bin/python scripts/verify_qalbench_population.py
```

Outputs:

- `results/qalbench_sweep.csv`
- `results/qalbench_sweep_metadata.json`
- `results/qalbench_population.csv`
- `results/qalbench_population_timeseries.csv`
- `results/qalbench_population_individuals.csv.xz`
- `results/qalbench_population_birth_events.csv.xz`
- `results/qalbench_population_attempts.csv.xz`
- `results/qalbench_population_metadata.json`
- `results/figures/z_population.png`
- `results/figures/quantum_diagnostics.png`
- `results/figures/phase_diagram.png`
- `results/figures/population_outcomes.png`
- `results/figures/lineage_nulls.png`
- `results/figures/resource_relevance.png`
- `paper/figures/z_population.png`
- `paper/figures/quantum_diagnostics.png`
- `paper/figures/phase_diagram.png`
- `paper/figures/population_outcomes.png`
- `paper/figures/lineage_nulls.png`
- `paper/figures/resource_relevance.png`

The resource-kernel verifier checks the exact Cartesian grid, recomputes every
row from the current implementation, validates the diagonal transition map with
a separate analytic reference, and validates the numerical claims quoted in the
manuscript. The population verifier checks scenario/model/seed/step coverage,
recomputes every stochastic replicate from its fixed seed, and validates the
summary, time-series, individual-record, birth-event, and reproduction-opportunity
schemas. Both verifiers check hard-coded canonical CSV and figure hashes,
metadata hashes, and the source-file SHA-256 manifest.
Expert escape hatches such as `--skip-metadata` or `--skip-figures` print
degraded-verification warnings. When no local `results/` tree is present, an
installed wheel falls back to packaged canonical artifacts for the verifier
commands.

The default sweeps do not modify packaged canonical artifacts. Maintainers
refresh the wheel fallback copy only from a clean Git tree with
`qalbench-sweep --sync-package-artifacts` or
`qalbench-population-sweep --sync-package-artifacts` after intentionally
accepting a new canonical run. This keeps release metadata from recording a
dirty provenance state.

## Build Paper

The paper source is in `paper/main.tex` with references in `paper/references.bib`.
The compiled PDF is written to `paper/build/main.pdf`.

Using Tectonic 0.16.9:

```bash
cd paper
mkdir -p build
tectonic --outdir build main.tex
```

## Packaging

The `qalbench` Python package is installable from this repository:

```bash
python3 -m pip install .
```

The package depends on NumPy and Matplotlib so the default commands can write
CSV and figures. Normal and editable installs expose `qalbench-sweep`,
`qalbench-verify`, `qalbench-population-sweep`, and
`qalbench-population-verify` console commands. The repository `scripts/` files
are thin wrappers around those package-native entry points.

The wheel intentionally includes the canonical CSV, compressed row-level
population artifacts, and figure files so installed verifier commands can audit
the shipped reference data without a separate repository checkout. This makes
the wheel large; for lightweight source-only inspection, use the repository tag
or archive deposit instead of a binary wheel.

## Citation and License

The repository includes `CITATION.cff` and is distributed under the MIT license.
The source repository is
[Gabriel-Kahen/qal-bench](https://github.com/Gabriel-Kahen/qal-bench). It
should be made public, tagged, and archived before journal submission.
`CITATION.cff` separates the software release citation from the manuscript
`preferred-citation`. A DOI should be added to the citation metadata after the
submission package is deposited in a DOI-minting archive such as Zenodo, OSF, or
an institutional repository. Until that deposit exists, the repository is not a
final citable release. See `ARCHIVE_RELEASE.md` for the fields that must be
filled after deposit.
