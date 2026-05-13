# Archive Release Checklist

This local package is prepared for archival release, but it is not submission-complete
until the following externally assigned fields exist.

- Public repository URL: https://github.com/Gabriel-Kahen/qal-bench
- Release commit:
- Release tag:
- Release URL:
- Archive DOI:
- Archive URL:

Before tagging:

- Commit all source, workflow, documentation, and dependency changes.
- From the clean release commit, refresh packaged canonical artifacts with:
  - `qalbench-sweep --sync-package-artifacts`
  - `qalbench-population-sweep --sync-package-artifacts`
- Confirm the refreshed metadata records `git_dirty: false`.
- Run `scripts/verify_release_artifacts.sh` without degraded-verification flags.
- Tag the verified release commit.

After deposit, update:

- `paper/main.tex` Data and Code Availability
- `README.md` Citation and License
- `CITATION.cff` with DOI metadata

Do not invent these values in the manuscript or citation metadata before the
public archive exists.
