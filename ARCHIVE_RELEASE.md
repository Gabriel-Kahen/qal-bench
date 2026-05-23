# Archive Release Checklist

- Public repository URL: https://github.com/Gabriel-Kahen/qal-bench
- Release commit: 9715627a0a3fc55ac17f5ce0273ad348fe013bcb
- Release tag: v0.3.0
- Release URL: https://github.com/Gabriel-Kahen/qal-bench/releases/tag/v0.3.0
- Archive DOI: 10.5281/zenodo.20222166
- Concept DOI: 10.5281/zenodo.20215830
- Archive URL: https://zenodo.org/records/20222166
- Manuscript preprint DOI: 10.5281/zenodo.20350285
- Manuscript preprint concept DOI: 10.5281/zenodo.20350284
- Manuscript preprint URL: https://zenodo.org/records/20350285

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

Post-release manuscript state:

- The canonical scientific artifact release remains `v0.3.0`.
- The manuscript preprint is archived separately from the software artifact on
  Zenodo record 20350285.
- Run artifact verification from the release tag so source-manifest hashes are
  checked against the archived source snapshot.
- Later manuscript-only edits should be archived separately if a venue requires
  the exact submitted PDF source, while preserving `v0.3.0` as the cited
  software-evidence release.
