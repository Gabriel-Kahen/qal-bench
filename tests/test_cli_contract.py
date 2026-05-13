from __future__ import annotations

import argparse
import csv
import sys
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qalbench import sweep, verify  # noqa: E402


class CliContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sweep = sweep
        cls.verify = verify

    def test_grid_counts_must_be_at_least_two(self) -> None:
        for value in ("0", "1", "-5"):
            with self.assertRaises(argparse.ArgumentTypeError):
                self.sweep.grid_count_arg(value)

        self.assertEqual(self.sweep.grid_count_arg("2"), 2)

    def test_probability_arguments_are_bounded(self) -> None:
        for value in ("-0.1", "1.1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                self.sweep.probability_arg(value)

        self.assertEqual(self.sweep.probability_arg("0.25"), 0.25)

    def test_custom_output_dirs_do_not_sync_paper_figures_by_default(self) -> None:
        default_args = argparse.Namespace(sync_paper_figures=None)
        self.assertEqual(
            self.sweep.paper_figure_dir_for_args(default_args, ROOT / "results"),
            ROOT / "paper" / "figures",
        )
        self.assertIsNone(
            self.sweep.paper_figure_dir_for_args(default_args, ROOT / "scratch-results")
        )

        forced_args = argparse.Namespace(sync_paper_figures=True)
        self.assertEqual(
            self.sweep.paper_figure_dir_for_args(forced_args, ROOT / "scratch-results"),
            ROOT / "paper" / "figures",
        )
        disabled_args = argparse.Namespace(sync_paper_figures=False)
        self.assertIsNone(
            self.sweep.paper_figure_dir_for_args(disabled_args, ROOT / "results")
        )

    def test_package_artifact_sync_is_opt_in(self) -> None:
        with mock.patch.object(sys, "argv", ["qalbench-sweep"]):
            args = self.sweep.parse_args()
        self.assertFalse(args.sync_package_artifacts)
        with mock.patch.object(sys, "argv", ["qalbench-sweep", "--sync-package-artifacts"]):
            args = self.sweep.parse_args()
        self.assertTrue(args.sync_package_artifacts)

    def test_source_metadata_shape_is_stable(self) -> None:
        metadata = self.sweep._source_metadata()
        self.assertEqual(
            set(metadata),
            {
                "package_version",
                "git_commit",
                "git_tag",
                "git_describe",
                "git_dirty",
                "source_files_sha256",
                "source_manifest_sha256",
            },
        )
        self.assertIsInstance(metadata["source_files_sha256"], dict)
        self.assertIsInstance(metadata["source_manifest_sha256"], str)

    def test_package_artifact_sync_requires_clean_tree(self) -> None:
        with mock.patch.object(self.sweep, "_git_status", return_value=" M README.md\n"):
            with self.assertRaises(SystemExit):
                self.sweep._require_clean_tree_for_package_artifact_sync()

        with mock.patch.object(self.sweep, "_git_status", return_value=None):
            with self.assertRaises(SystemExit):
                self.sweep._require_clean_tree_for_package_artifact_sync()

        with mock.patch.object(self.sweep, "_git_status", return_value=""):
            self.sweep._require_clean_tree_for_package_artifact_sync()

    def test_noncanonical_results_directory_is_not_default_artifact_root(self) -> None:
        artifact_root = Path("/tmp/qalbench-isolated-run")
        self.assertFalse(self.verify._is_canonical_artifact_root(artifact_root))
        self.assertTrue(self.verify._is_canonical_artifact_root(ROOT))
        self.assertTrue(
            self.verify._is_canonical_artifact_root(ROOT / "src" / "qalbench" / "data")
        )

    def test_verifier_rejects_duplicate_or_missing_grid_points(self) -> None:
        csv_path = ROOT / "results" / "qalbench_sweep.csv"
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

        duplicated = rows[:-1] + [dict(rows[0])]
        with self.assertRaisesRegex(AssertionError, "duplicate Cartesian grid points"):
            self.verify._assert_exact_cartesian_grid(duplicated)

        with self.assertRaisesRegex(AssertionError, "missing Cartesian grid points"):
            self.verify._assert_exact_cartesian_grid(rows[:-1])


if __name__ == "__main__":
    unittest.main()
