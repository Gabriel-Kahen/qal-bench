#!/usr/bin/env python3
"""Verify the canonical population-level QAL benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import lzma
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, TextIO

from .population_sweep import (
    ATTEMPTS_FILENAME,
    ATTEMPT_FIELDS,
    BIRTH_EVENTS_FILENAME,
    BIRTH_EVENT_FIELDS,
    INDIVIDUALS_FILENAME,
    INDIVIDUAL_FIELDS,
    MODELS,
    SCENARIOS,
    SUMMARY_FIELDS,
    TRAJECTORY_FIELDS,
    run_population_sweep,
)
from .sweep import ROOT, _sha256


PACKAGE_ROOT = Path(__file__).resolve().parent
EXPECTED_ARTIFACT = "qalbench population benchmark"
EXPECTED_SCHEMA_VERSION = 2
EXPECTED_FIRST_SEED = 11
EXPECTED_SEED_COUNT = 100
EXPECTED_STEPS = 64
EXPECTED_SUMMARY_ROW_COUNT = len(SCENARIOS) * len(MODELS) * EXPECTED_SEED_COUNT
EXPECTED_TRAJECTORY_ROW_COUNT = EXPECTED_SUMMARY_ROW_COUNT * (EXPECTED_STEPS + 1)
EXPECTED_ATTEMPT_ROW_COUNT = 4971676
EXPECTED_SUMMARY_SHA256 = "3baf233e03125f5c1a096d375ac1fc20940c9a0e32de42474974bd19adc5c460"
EXPECTED_TRAJECTORY_SHA256 = "32188555b5226f2770c07dd474e1cdb6eca1e6ad8d0f4135b7ee11f09c2c399a"
EXPECTED_INDIVIDUAL_SHA256 = "473d5cb5b69f2ef8a086d7a527f0ae48efd90ade77eddbca026e8d668a2a5e55"
EXPECTED_BIRTH_EVENT_SHA256 = "5aee904e88710db2a3886f2bebe3990fc814f623793d86ff1bf53674b405c394"
EXPECTED_ATTEMPT_SHA256 = "97391de4515bb71092c47e54ee17a3fdf5cb70dfbe986c9c0e47d012ac29087a"
EXPECTED_FIGURE_SHA256_BY_NAME = {
    "lineage_nulls.png": "b63a9647325a7dddbd1897af2911bce9e0b62c8983027ddc7b5ace1ae5148528",
    "population_outcomes.png": "29b92c61a254a44055313e36a4516d84465c96ebd27dc999116e12422db83c3f",
    "resource_relevance.png": "6956faa873860efd048e9a9a4bc659122d414fd7ee912124f5e8fd71f9b02173",
}
EXPECTED_CANONICAL_FIGURE_SHA256 = {
    f"{directory}/{name}": digest
    for directory in ("results/figures", "paper/figures")
    for name, digest in EXPECTED_FIGURE_SHA256_BY_NAME.items()
}
EXPECTED_PARAMETERS = {
    "scenarios": list(SCENARIOS),
    "models": list(MODELS),
    "first_seed": EXPECTED_FIRST_SEED,
    "seed_count": EXPECTED_SEED_COUNT,
    "steps": EXPECTED_STEPS,
    "initial_population": 24,
    "carrying_capacity": 80,
    "base_birth_probability": 0.28,
    "base_death_probability": 0.06,
    "density_death_probability": 0.08,
    "mutation_probability": 0.08,
    "mutation_step": 0.35,
    "local_perturbation_probability": 0.15,
    "local_perturbation_angle": 1.0471975511965976,
    "damping_probability": 0.10,
    "phase_damping": 0.0,
    "interaction_strength": 0.12,
    "lineage_null_permutations": 200,
}


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _isclose(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return abs(left - right) <= tolerance


@contextmanager
def _csv_read_handle(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", newline="") as handle:
            yield handle
    elif path.suffix == ".xz":
        with lzma.open(path, "rt", newline="") as handle:
            yield handle
    else:
        with path.open(newline="") as handle:
            yield handle


def _csv_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.suffix == ".xz":
        with lzma.open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    return _sha256(path)


def _default_attempts_path(artifact_root: Path) -> Path:
    return artifact_root / "results" / ATTEMPTS_FILENAME


def _default_individuals_path(artifact_root: Path) -> Path:
    return artifact_root / "results" / INDIVIDUALS_FILENAME


def _default_birth_events_path(artifact_root: Path) -> Path:
    return artifact_root / "results" / BIRTH_EVENTS_FILENAME


def _artifact_root_for_summary(summary_path: Path) -> Path:
    summary_path = summary_path.resolve()
    if summary_path.name == "qalbench_population.csv" and summary_path.parent.name == "results":
        return summary_path.parent.parent
    return ROOT


def _is_source_checkout_root(artifact_root: Path) -> bool:
    return (artifact_root / "src" / "qalbench").resolve() == PACKAGE_ROOT.resolve()


def _is_canonical_artifact_root(artifact_root: Path) -> bool:
    artifact_root = artifact_root.resolve()
    return _is_source_checkout_root(artifact_root) or artifact_root == (
        PACKAGE_ROOT / "data"
    ).resolve()


def default_artifact_root() -> Path:
    if "QALBENCH_ROOT" in __import__("os").environ:
        return ROOT
    cwd = Path.cwd().resolve()
    if (cwd / "results" / "qalbench_population.csv").exists():
        return cwd
    packaged_root = PACKAGE_ROOT / "data"
    if (packaged_root / "results" / "qalbench_population.csv").exists():
        return packaged_root
    return cwd


def _assert_summary_coverage(rows: list[dict[str, str]]) -> None:
    seen = {(row["scenario"], row["model"], int(row["seed"])) for row in rows}
    expected = {
        (scenario, model, seed)
        for scenario in SCENARIOS
        for model in MODELS
        for seed in range(EXPECTED_FIRST_SEED, EXPECTED_FIRST_SEED + EXPECTED_SEED_COUNT)
    }
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise AssertionError(f"summary coverage mismatch; missing={missing[:5]}, extra={extra[:5]}")


def _assert_trajectory_coverage(rows: list[dict[str, str]]) -> None:
    seen = {
        (row["scenario"], row["model"], int(row["seed"]), int(row["step"]))
        for row in rows
    }
    expected = {
        (scenario, model, seed, step)
        for scenario in SCENARIOS
        for model in MODELS
        for seed in range(EXPECTED_FIRST_SEED, EXPECTED_FIRST_SEED + EXPECTED_SEED_COUNT)
        for step in range(EXPECTED_STEPS + 1)
    }
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise AssertionError(
            f"trajectory coverage mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )


def _namespace_from_expected_parameters() -> SimpleNamespace:
    parameters = EXPECTED_PARAMETERS
    return SimpleNamespace(
        first_seed=parameters["first_seed"],
        seed_count=parameters["seed_count"],
        steps=parameters["steps"],
        initial_population=parameters["initial_population"],
        carrying_capacity=parameters["carrying_capacity"],
        base_birth_probability=parameters["base_birth_probability"],
        base_death_probability=parameters["base_death_probability"],
        density_death_probability=parameters["density_death_probability"],
        mutation_probability=parameters["mutation_probability"],
        mutation_step=parameters["mutation_step"],
        local_perturbation_probability=parameters["local_perturbation_probability"],
        local_perturbation_angle=parameters["local_perturbation_angle"],
        damping_probability=parameters["damping_probability"],
        phase_damping=parameters["phase_damping"],
        interaction_strength=parameters["interaction_strength"],
        lineage_null_permutations=parameters["lineage_null_permutations"],
    )


def _row_key(row: dict[str, str] | dict[str, float | int | str]) -> tuple[str, str, int]:
    return (str(row["scenario"]), str(row["model"]), int(row["seed"]))


def _trajectory_key(
    row: dict[str, str] | dict[str, float | int | str],
) -> tuple[str, str, int, int]:
    return (str(row["scenario"]), str(row["model"]), int(row["seed"]), int(row["step"]))


def _individual_key(
    row: dict[str, str] | dict[str, float | int | str | bool | None],
) -> tuple[str, str, int, int]:
    return (str(row["scenario"]), str(row["model"]), int(row["seed"]), int(row["id"]))


def _birth_event_key(
    row: dict[str, str] | dict[str, float | int | str | bool],
) -> tuple[str, str, int, int]:
    return (
        str(row["scenario"]),
        str(row["model"]),
        int(row["seed"]),
        int(row["child_id"]),
    )


def _attempt_key(
    row: dict[str, str] | dict[str, float | int | str | bool | None],
) -> tuple[str, str, int, int, int]:
    return (
        str(row["scenario"]),
        str(row["model"]),
        int(row["seed"]),
        int(row["step"]),
        int(row["parent_id"]),
    )


def _compare_rows(
    actual_rows: list[dict[str, str]],
    expected_rows: list[dict[str, float | int | str | bool | None]],
    fieldnames: list[str],
    key_name: str,
) -> float:
    key_fn_by_name = {
        "summary": _row_key,
        "trajectory": _trajectory_key,
        "individual": _individual_key,
        "birth_event": _birth_event_key,
        "attempt": _attempt_key,
    }
    key_fn = key_fn_by_name[key_name]
    actual = {key_fn(row): row for row in actual_rows}
    expected = {key_fn(row): row for row in expected_rows}
    if set(actual) != set(expected):
        raise AssertionError(f"{key_name} recompute key mismatch")
    max_gap = 0.0
    for key, expected_row in expected.items():
        actual_row = actual[key]
        for field in fieldnames:
            expected_value = expected_row[field]
            actual_value = actual_row[field]
            if isinstance(expected_value, str) or expected_value is None or isinstance(expected_value, bool):
                normalized_actual = None if actual_value == "" and expected_value is None else actual_value
                if isinstance(expected_value, bool):
                    normalized_actual = str(actual_value) == "True"
                if normalized_actual != expected_value:
                    raise AssertionError(
                        f"{key_name} field {field} mismatch at {key}: "
                        f"expected {expected_value!r}, got {actual_value!r}"
                    )
            else:
                gap = abs(float(actual_value) - float(expected_value))
                max_gap = max(max_gap, gap)
                if gap > 1e-9:
                    raise AssertionError(
                        f"{key_name} field {field} mismatch at {key}: "
                        f"expected {expected_value}, got {actual_value}"
                    )
    return max_gap


def _verify_hashes(
    summary_path: Path,
    trajectory_path: Path,
    individuals_path: Path,
    birth_events_path: Path,
    attempts_path: Path,
) -> None:
    if EXPECTED_SUMMARY_SHA256 and _sha256(summary_path) != EXPECTED_SUMMARY_SHA256:
        raise AssertionError(
            f"population summary CSV hash mismatch: expected {EXPECTED_SUMMARY_SHA256}, "
            f"computed {_sha256(summary_path)}"
        )
    if EXPECTED_TRAJECTORY_SHA256 and _sha256(trajectory_path) != EXPECTED_TRAJECTORY_SHA256:
        raise AssertionError(
            "population trajectory CSV hash mismatch: "
            f"expected {EXPECTED_TRAJECTORY_SHA256}, computed {_sha256(trajectory_path)}"
        )
    if EXPECTED_INDIVIDUAL_SHA256 and _csv_content_sha256(individuals_path) != EXPECTED_INDIVIDUAL_SHA256:
        raise AssertionError(
            "population individual CSV hash mismatch: "
            f"expected {EXPECTED_INDIVIDUAL_SHA256}, computed {_csv_content_sha256(individuals_path)}"
        )
    if EXPECTED_BIRTH_EVENT_SHA256 and _csv_content_sha256(birth_events_path) != EXPECTED_BIRTH_EVENT_SHA256:
        raise AssertionError(
            "population birth-event CSV hash mismatch: "
            f"expected {EXPECTED_BIRTH_EVENT_SHA256}, computed {_csv_content_sha256(birth_events_path)}"
        )
    if EXPECTED_ATTEMPT_SHA256 and _csv_content_sha256(attempts_path) != EXPECTED_ATTEMPT_SHA256:
        raise AssertionError(
            "population attempt CSV hash mismatch: "
            f"expected {EXPECTED_ATTEMPT_SHA256}, computed {_csv_content_sha256(attempts_path)}"
        )


def _verify_figure_hashes(artifact_root: Path) -> None:
    if not EXPECTED_CANONICAL_FIGURE_SHA256:
        return
    for relative_path, expected_hash in EXPECTED_CANONICAL_FIGURE_SHA256.items():
        figure_path = artifact_root / relative_path
        if not figure_path.exists():
            raise AssertionError(f"canonical population figure is missing: {figure_path}")
        actual_hash = _sha256(figure_path)
        if expected_hash and actual_hash != expected_hash:
            raise AssertionError(
                f"canonical population figure hash mismatch for {relative_path}: "
                f"expected {expected_hash}, computed {actual_hash}"
            )


def _verify_metadata(
    *,
    metadata_path: Path,
    summary_path: Path,
    trajectory_path: Path,
    individuals_path: Path,
    birth_events_path: Path,
    attempts_path: Path,
    artifact_root: Path,
) -> None:
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    if metadata.get("artifact") != EXPECTED_ARTIFACT:
        raise AssertionError(f"metadata artifact mismatch: {metadata.get('artifact')!r}")
    if metadata.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise AssertionError("metadata schema_version mismatch")
    if metadata.get("summary_row_count") != EXPECTED_SUMMARY_ROW_COUNT:
        raise AssertionError("metadata summary_row_count mismatch")
    if metadata.get("trajectory_row_count") != EXPECTED_TRAJECTORY_ROW_COUNT:
        raise AssertionError("metadata trajectory_row_count mismatch")
    if not isinstance(metadata.get("individual_row_count"), int):
        raise AssertionError("metadata individual_row_count must be present")
    if not isinstance(metadata.get("birth_event_row_count"), int):
        raise AssertionError("metadata birth_event_row_count must be present")
    if EXPECTED_ATTEMPT_ROW_COUNT and metadata.get("attempt_row_count") != EXPECTED_ATTEMPT_ROW_COUNT:
        raise AssertionError("metadata attempt_row_count mismatch")
    parameters = metadata.get("parameters")
    if parameters != EXPECTED_PARAMETERS:
        raise AssertionError(
            f"metadata parameters mismatch: expected {EXPECTED_PARAMETERS}, found {parameters}"
        )
    if metadata.get("summary_csv_sha256") != _sha256(summary_path):
        raise AssertionError("metadata summary hash mismatch")
    if metadata.get("trajectory_csv_sha256") != _sha256(trajectory_path):
        raise AssertionError("metadata trajectory hash mismatch")
    if metadata.get("individual_csv_sha256") != _csv_content_sha256(individuals_path):
        raise AssertionError("metadata individual hash mismatch")
    if metadata.get("birth_event_csv_sha256") != _csv_content_sha256(birth_events_path):
        raise AssertionError("metadata birth-event hash mismatch")
    if metadata.get("attempt_csv_sha256") != _csv_content_sha256(attempts_path):
        raise AssertionError("metadata attempt hash mismatch")
    if EXPECTED_SUMMARY_SHA256 and metadata.get("summary_csv_sha256") != EXPECTED_SUMMARY_SHA256:
        raise AssertionError("metadata summary hash is not canonical")
    if EXPECTED_TRAJECTORY_SHA256 and metadata.get("trajectory_csv_sha256") != EXPECTED_TRAJECTORY_SHA256:
        raise AssertionError("metadata trajectory hash is not canonical")

    source = metadata.get("source", {})
    if not isinstance(source, dict):
        raise AssertionError("metadata source must be an object")
    manifest = source.get("source_files_sha256")
    manifest_digest = source.get("source_manifest_sha256")
    if not isinstance(manifest, dict) or not isinstance(manifest_digest, str):
        raise AssertionError("metadata source must include a source-file hash manifest")
    encoded_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded_manifest).hexdigest() != manifest_digest:
        raise AssertionError("metadata source manifest digest mismatch")
    if _is_source_checkout_root(artifact_root):
        for relative_path, expected_hash in manifest.items():
            source_path = artifact_root / relative_path
            if not source_path.exists():
                raise AssertionError(f"source file listed in metadata is missing: {source_path}")
            actual_hash = _sha256(source_path)
            if actual_hash != expected_hash:
                raise AssertionError(
                    f"source file hash mismatch for {relative_path}: "
                    f"metadata has {expected_hash}, computed {actual_hash}"
                )

    figure_hashes = metadata.get("figures_sha256", {})
    if not isinstance(figure_hashes, dict):
        raise AssertionError("metadata figures_sha256 must be an object")
    for relative_path, expected_hash in figure_hashes.items():
        figure_path = artifact_root / relative_path
        if not figure_path.exists():
            raise AssertionError(f"figure listed in metadata is missing: {figure_path}")
        if _sha256(figure_path) != expected_hash:
            raise AssertionError(f"figure hash mismatch for {relative_path}")
        canonical_hash = EXPECTED_FIGURE_SHA256_BY_NAME.get(figure_path.name)
        if canonical_hash and expected_hash != canonical_hash:
            raise AssertionError(f"metadata figure hash is not canonical for {relative_path}")


def verify(
    summary_path: Path,
    trajectory_path: Path,
    individuals_path: Path,
    birth_events_path: Path,
    attempts_path: Path,
    metadata_path: Path | None = None,
    *,
    check_metadata: bool = True,
    check_figures: bool = True,
) -> dict[str, float]:
    artifact_root = _artifact_root_for_summary(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"population summary artifact not found: {summary_path}")
    if not trajectory_path.exists():
        raise FileNotFoundError(f"population trajectory artifact not found: {trajectory_path}")
    if not individuals_path.exists():
        raise FileNotFoundError(f"population individual artifact not found: {individuals_path}")
    if not birth_events_path.exists():
        raise FileNotFoundError(f"population birth-event artifact not found: {birth_events_path}")
    if not attempts_path.exists():
        raise FileNotFoundError(f"population attempt artifact not found: {attempts_path}")
    with _csv_read_handle(summary_path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SUMMARY_FIELDS:
            raise AssertionError("unexpected population summary CSV schema")
        summary_rows = list(reader)
    with _csv_read_handle(trajectory_path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != TRAJECTORY_FIELDS:
            raise AssertionError("unexpected population trajectory CSV schema")
        trajectory_rows = list(reader)
    with _csv_read_handle(individuals_path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != INDIVIDUAL_FIELDS:
            raise AssertionError("unexpected population individual CSV schema")
        individual_rows = list(reader)
    with _csv_read_handle(birth_events_path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != BIRTH_EVENT_FIELDS:
            raise AssertionError("unexpected population birth-event CSV schema")
        birth_event_rows = list(reader)
    with _csv_read_handle(attempts_path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ATTEMPT_FIELDS:
            raise AssertionError("unexpected population attempt CSV schema")
        attempt_rows = list(reader)
    if len(summary_rows) != EXPECTED_SUMMARY_ROW_COUNT:
        raise AssertionError(
            f"expected {EXPECTED_SUMMARY_ROW_COUNT} summary rows, found {len(summary_rows)}"
        )
    if len(trajectory_rows) != EXPECTED_TRAJECTORY_ROW_COUNT:
        raise AssertionError(
            f"expected {EXPECTED_TRAJECTORY_ROW_COUNT} trajectory rows, found {len(trajectory_rows)}"
        )
    _assert_summary_coverage(summary_rows)
    _assert_trajectory_coverage(trajectory_rows)
    if EXPECTED_ATTEMPT_ROW_COUNT and len(attempt_rows) != EXPECTED_ATTEMPT_ROW_COUNT:
        raise AssertionError(
            f"expected {EXPECTED_ATTEMPT_ROW_COUNT} attempt rows, found {len(attempt_rows)}"
        )
    _verify_hashes(summary_path, trajectory_path, individuals_path, birth_events_path, attempts_path)
    if check_figures and _is_canonical_artifact_root(artifact_root):
        _verify_figure_hashes(artifact_root)
    elif check_figures:
        print(
            "warning: canonical population figure-hash validation skipped for "
            "noncanonical artifact root; verification is degraded",
            file=sys.stderr,
        )
    if check_metadata:
        if metadata_path is None:
            raise AssertionError("metadata checks requested but no metadata path was provided")
        _verify_metadata(
            metadata_path=metadata_path,
            summary_path=summary_path,
            trajectory_path=trajectory_path,
            individuals_path=individuals_path,
            birth_events_path=birth_events_path,
            attempts_path=attempts_path,
            artifact_root=artifact_root,
        )

    (
        expected_summary,
        expected_trajectory,
        expected_individuals,
        expected_birth_events,
        expected_attempts,
    ) = run_population_sweep(
        _namespace_from_expected_parameters()
    )
    max_summary_recompute_gap = _compare_rows(
        summary_rows,
        expected_summary,
        SUMMARY_FIELDS,
        "summary",
    )
    max_trajectory_recompute_gap = _compare_rows(
        trajectory_rows,
        expected_trajectory,
        TRAJECTORY_FIELDS,
        "trajectory",
    )
    max_individual_recompute_gap = _compare_rows(
        individual_rows,
        expected_individuals,
        INDIVIDUAL_FIELDS,
        "individual",
    )
    max_birth_event_recompute_gap = _compare_rows(
        birth_event_rows,
        expected_birth_events,
        BIRTH_EVENT_FIELDS,
        "birth_event",
    )
    max_attempt_recompute_gap = _compare_rows(
        attempt_rows,
        expected_attempts,
        ATTEMPT_FIELDS,
        "attempt",
    )

    resource_quantum = [
        _float(row, "final_population")
        for row in summary_rows
        if row["scenario"] == "resource_selection" and row["model"] == "quantum"
    ]
    resource_classical = [
        _float(row, "final_population")
        for row in summary_rows
        if row["scenario"] == "resource_selection" and row["model"] == "classical"
    ]
    basis_quantum_mi = [
        _float(row, "parent_offspring_mutual_information")
        for row in summary_rows
        if row["scenario"] == "basis_selection" and row["model"] == "quantum"
    ]
    basis_shuffled_mi = [
        _float(row, "shuffled_lineage_mutual_information")
        for row in summary_rows
        if row["scenario"] == "basis_selection" and row["model"] == "quantum"
    ]
    quantum_resource_scores = [
        _float(row, "mean_event_resource_score")
        for row in summary_rows
        if row["model"] == "quantum"
    ]
    dephased_resource_scores = [
        _float(row, "mean_event_resource_score")
        for row in summary_rows
        if row["model"] == "dephased"
    ]
    if np_mean(quantum_resource_scores) <= np_mean(dephased_resource_scores):
        raise AssertionError("quantum resource score should exceed dephased control")
    if np_mean(basis_quantum_mi) <= np_mean(basis_shuffled_mi):
        raise AssertionError("basis lineage MI should exceed shuffled-lineage null")

    return {
        "summary_rows": float(len(summary_rows)),
        "trajectory_rows": float(len(trajectory_rows)),
        "individual_rows": float(len(individual_rows)),
        "birth_event_rows": float(len(birth_event_rows)),
        "attempt_rows": float(len(attempt_rows)),
        "max_summary_recompute_gap": max_summary_recompute_gap,
        "max_trajectory_recompute_gap": max_trajectory_recompute_gap,
        "max_individual_recompute_gap": max_individual_recompute_gap,
        "max_birth_event_recompute_gap": max_birth_event_recompute_gap,
        "max_attempt_recompute_gap": max_attempt_recompute_gap,
        "resource_quantum_final_population_mean": np_mean(resource_quantum),
        "resource_classical_final_population_mean": np_mean(resource_classical),
        "basis_quantum_lineage_mi_mean": np_mean(basis_quantum_mi),
        "basis_quantum_shuffled_mi_mean": np_mean(basis_shuffled_mi),
    }


def np_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    artifact_root = default_artifact_root()
    parser.add_argument(
        "--summary-csv",
        default=str(artifact_root / "results" / "qalbench_population.csv"),
    )
    parser.add_argument(
        "--timeseries-csv",
        default=str(artifact_root / "results" / "qalbench_population_timeseries.csv"),
    )
    parser.add_argument(
        "--individuals-csv",
        default=str(_default_individuals_path(artifact_root)),
    )
    parser.add_argument(
        "--birth-events-csv",
        default=str(_default_birth_events_path(artifact_root)),
    )
    parser.add_argument(
        "--attempts-csv",
        default=str(_default_attempts_path(artifact_root)),
    )
    parser.add_argument(
        "--metadata",
        default=str(artifact_root / "results" / "qalbench_population_metadata.json"),
    )
    parser.add_argument("--skip-metadata", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    metadata_path = None if args.metadata == "" else Path(args.metadata)
    summary = verify(
        Path(args.summary_csv),
        Path(args.timeseries_csv),
        Path(args.individuals_csv),
        Path(args.birth_events_csv),
        Path(args.attempts_csv),
        metadata_path=metadata_path,
        check_metadata=not args.skip_metadata,
        check_figures=not args.skip_figures,
    )
    if args.skip_metadata:
        print("warning: population metadata validation skipped", file=sys.stderr)
    if args.skip_figures:
        print("warning: population figure-hash validation skipped", file=sys.stderr)
    for key, value in summary.items():
        print(f"{key}: {value:.16g}")


if __name__ == "__main__":
    main()
