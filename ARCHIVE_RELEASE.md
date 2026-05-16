# Archive Release Checklist

- Public repository URL: https://github.com/Gabriel-Kahen/qal-bench
- Release commit: TBD
- Release tag: v0.3.0
- Release URL: https://github.com/Gabriel-Kahen/qal-bench/releases/tag/v0.3.0
- Archive DOI: TBD
- Concept DOI: 10.5281/zenodo.20215830
- Archive URL: TBD

Before tagging:

- Commit all source, workflow, documentation, and dependency changes.
- From the clean release commit, refresh packaged canonical artifacts with:
  - `qalbench-sweep --sync-package-artifacts`
  - `qalbench-population-sweep --sync-package-artifacts`
- Confirm the refreshed metadata records `git_dirty: false`.
- Run `scripts/verify_release_artifacts.sh` without degraded-verification flags.
- Tag the verified release commit.

After deposit, updated:

- `paper/main.tex` Data and Code Availability
- `README.md` Citation and License
- `CITATION.cff` with DOI metadata
