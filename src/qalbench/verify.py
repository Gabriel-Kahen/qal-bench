#!/usr/bin/env python3
"""Verify the canonical numerical artifact reported in the QAL benchmark paper.

This verifier is intentionally tied to the paper-default sweep. It validates the
CSV schema and parameter grid, recomputes every row from the current qalbench
implementation, and then checks the scalar claims quoted in the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("QALBENCH_ROOT", Path.cwd())).resolve()

from . import ClassicalParams, QuantumParams, compute_metrics
from .classical import BITSTRINGS, run_classical_event
from .quantum import run_quantum_event


EXPECTED_ARTIFACT = "qalbench two-qubit resource-kernel sweep"
EXPECTED_SCHEMA_VERSION = 2
EXPECTED_THETA_COUNT = 41
EXPECTED_LOCAL_PERTURBATION_COUNT = 21
EXPECTED_DAMPING_COUNT = 21
EXPECTED_ROW_COUNT = (
    EXPECTED_THETA_COUNT * EXPECTED_LOCAL_PERTURBATION_COUNT * EXPECTED_DAMPING_COUNT
)
EXPECTED_CSV_SHA256 = "3e0a1434e8ad481c399a99dd4ad5e4fe90668a412a5cc2c6ffbaef0b59e0d7e0"
EXPECTED_FIGURE_SHA256_BY_NAME = {
    "phase_diagram.png": "74d3c8651e9d431fba24291dc84fb0db1202e7f369831e347fb73d48c87dba95",
    "quantum_diagnostics.png": "afaba9fd918d273f6248bff826dfb4483c4f857ac895344ffdbbac8cfadeee56",
    "z_population.png": "7d3b0dc82e1b65e91e8b678338400f2d56a94cf310439b06610ef46140a4bf86",
}
EXPECTED_CANONICAL_FIGURE_SHA256 = {
    f"{directory}/{name}": digest
    for directory in ("results/figures", "paper/figures")
    for name, digest in EXPECTED_FIGURE_SHA256_BY_NAME.items()
}
EXPECTED_PARAMETERS = {
    "theta_count": EXPECTED_THETA_COUNT,
    "local_perturbation_count": EXPECTED_LOCAL_PERTURBATION_COUNT,
    "damping_count": EXPECTED_DAMPING_COUNT,
    "max_local_perturbation_probability": 1.0,
    "local_perturbation_angle": math.pi / 3.0,
    "max_damping": 1.0,
    "phi": 0.0,
    "interaction_angle": 0.0,
    "phase_damping": 0.0,
}
METRIC_SUFFIXES = [
    "trace",
    "purity",
    "p_00",
    "p_01",
    "p_10",
    "p_11",
    "z_copy_agreement",
    "z_genotype",
    "z_offspring",
    "coherence_l1",
    "xx",
    "xy",
    "yx",
    "yy",
    "concurrence",
    "negativity",
    "ppt_min_eig",
    "chsh_max",
    "entropy",
]
PARAMETER_FIELDS = [
    "theta",
    "phi",
    "local_perturbation_probability",
    "local_perturbation_angle",
    "interaction_angle",
    "damping_probability",
    "phase_damping",
]
EXPECTED_FIELDNAMES = (
    PARAMETER_FIELDS
    + [f"quantum_{suffix}" for suffix in METRIC_SUFFIXES]
    + [f"dephased_{suffix}" for suffix in METRIC_SUFFIXES]
    + [f"classical_{suffix}" for suffix in METRIC_SUFFIXES]
)


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _isclose(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


def _rows_at(
    rows: list[dict[str, str]],
    *,
    theta: float | None = None,
    local_perturbation_probability: float | None = None,
    damping_probability: float | None = None,
) -> list[dict[str, str]]:
    out = rows
    if theta is not None:
        out = [row for row in out if _isclose(_float(row, "theta"), theta)]
    if local_perturbation_probability is not None:
        out = [
            row
            for row in out
            if _isclose(
                _float(row, "local_perturbation_probability"),
                local_perturbation_probability,
            )
        ]
    if damping_probability is not None:
        out = [
            row
            for row in out
            if _isclose(_float(row, "damping_probability"), damping_probability)
        ]
    return out


def _single(rows: list[dict[str, str]], **kwargs: float) -> dict[str, str]:
    matches = _rows_at(rows, **kwargs)
    if len(matches) != 1:
        raise AssertionError(f"expected one row for {kwargs}, found {len(matches)}")
    return matches[0]


def _assert_close(label: str, actual: float, expected: float, tol: float = 1e-12) -> None:
    if not _isclose(actual, expected, tol):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def _assert_single_value(rows: list[dict[str, str]], key: str, expected: float) -> None:
    values = {_float(row, key) for row in rows}
    if values != {expected}:
        raise AssertionError(f"expected only {key}={expected}, found {values}")


def _expected_grid_values(key: str) -> list[float]:
    if key == "theta":
        count = EXPECTED_THETA_COUNT
        stop = math.pi
    elif key == "local_perturbation_probability":
        count = EXPECTED_LOCAL_PERTURBATION_COUNT
        stop = 1.0
    elif key == "damping_probability":
        count = EXPECTED_DAMPING_COUNT
        stop = 1.0
    else:
        raise ValueError(f"unexpected grid key {key}")
    return [stop * index / (count - 1) for index in range(count)]


def _grid_index(value: float, expected_values: list[float], key: str) -> int:
    matches = [
        index
        for index, expected in enumerate(expected_values)
        if _isclose(value, expected)
    ]
    if len(matches) != 1:
        sample = ", ".join(f"{expected:.16g}" for expected in expected_values[:3])
        raise AssertionError(
            f"{key}={value:.16g} is not on the expected grid "
            f"(starts {sample}, ...)"
        )
    return matches[0]


def _assert_exact_cartesian_grid(rows: list[dict[str, str]]) -> None:
    theta_values = _expected_grid_values("theta")
    perturbation_values = _expected_grid_values("local_perturbation_probability")
    damping_values = _expected_grid_values("damping_probability")
    seen: dict[tuple[int, int, int], int] = {}
    duplicates: list[tuple[int, int, int]] = []

    for row_number, row in enumerate(rows, start=2):
        key = (
            _grid_index(_float(row, "theta"), theta_values, "theta"),
            _grid_index(
                _float(row, "local_perturbation_probability"),
                perturbation_values,
                "local_perturbation_probability",
            ),
            _grid_index(
                _float(row, "damping_probability"),
                damping_values,
                "damping_probability",
            ),
        )
        previous_row = seen.setdefault(key, row_number)
        if previous_row != row_number:
            duplicates.append(key)

    expected = {
        (theta_index, perturbation_index, damping_index)
        for theta_index in range(EXPECTED_THETA_COUNT)
        for perturbation_index in range(EXPECTED_LOCAL_PERTURBATION_COUNT)
        for damping_index in range(EXPECTED_DAMPING_COUNT)
    }
    missing = sorted(expected - set(seen))
    if duplicates:
        preview = ", ".join(str(item) for item in duplicates[:5])
        raise AssertionError(
            f"duplicate Cartesian grid points found: {preview}"
            + (" ..." if len(duplicates) > 5 else "")
        )
    if missing:
        preview = ", ".join(str(item) for item in missing[:5])
        raise AssertionError(
            f"missing Cartesian grid points found: {preview}"
            + (" ..." if len(missing) > 5 else "")
        )


def _analytic_diagonal_probabilities(
    *,
    theta: float,
    local_perturbation_probability: float,
    local_perturbation_angle: float,
    damping_probability: float,
) -> dict[str, float]:
    """Reference diagonal map for the paper-default event family.

    This deliberately does not call qalbench.quantum or qalbench.classical. It is
    the closed-form bit transition used to audit the diagonal-equivalence claim.
    """

    p0 = math.cos(theta / 2.0) ** 2
    p1 = math.sin(theta / 2.0) ** 2
    probabilities = {
        "00": p0,
        "01": 0.0,
        "10": 0.0,
        "11": p1,
    }

    flip = local_perturbation_probability * (
        math.sin(local_perturbation_angle / 2.0) ** 2
    )
    if flip:
        probabilities = {
            "00": (1.0 - flip) * probabilities["00"] + flip * probabilities["01"],
            "01": flip * probabilities["00"] + (1.0 - flip) * probabilities["01"],
            "10": (1.0 - flip) * probabilities["10"] + flip * probabilities["11"],
            "11": flip * probabilities["10"] + (1.0 - flip) * probabilities["11"],
        }

    gamma = damping_probability
    if gamma:
        probabilities = {
            "00": probabilities["00"] + gamma * probabilities["01"],
            "01": (1.0 - gamma) * probabilities["01"],
            "10": probabilities["10"] + gamma * probabilities["11"],
            "11": (1.0 - gamma) * probabilities["11"],
        }

    norm = sum(probabilities.values())
    return {key: value / norm for key, value in probabilities.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_root_for_csv(csv_path: Path) -> Path:
    csv_path = csv_path.resolve()
    if csv_path.name == "qalbench_sweep.csv" and csv_path.parent.name == "results":
        return csv_path.parent.parent
    return ROOT


def _is_canonical_artifact_root(artifact_root: Path) -> bool:
    artifact_root = artifact_root.resolve()
    source_checkout = (artifact_root / "src" / "qalbench").resolve() == PACKAGE_ROOT.resolve()
    packaged_data = artifact_root == (PACKAGE_ROOT / "data").resolve()
    return source_checkout or packaged_data


def _is_source_checkout_root(artifact_root: Path) -> bool:
    return (artifact_root / "src" / "qalbench").resolve() == PACKAGE_ROOT.resolve()


def default_artifact_root() -> Path:
    """Prefer a local artifact tree, then packaged canonical artifacts."""

    if "QALBENCH_ROOT" in os.environ:
        return ROOT
    cwd = Path.cwd().resolve()
    if (cwd / "results" / "qalbench_sweep.csv").exists():
        return cwd
    packaged_root = PACKAGE_ROOT / "data"
    if (packaged_root / "results" / "qalbench_sweep.csv").exists():
        return packaged_root
    return cwd


def _verify_expected_csv_hash(csv_path: Path) -> None:
    actual_hash = _sha256(csv_path)
    if actual_hash != EXPECTED_CSV_SHA256:
        raise AssertionError(
            f"canonical CSV hash mismatch: expected {EXPECTED_CSV_SHA256}, "
            f"computed {actual_hash}"
        )


def _verify_expected_figure_hashes(artifact_root: Path) -> None:
    for relative_path, expected_hash in EXPECTED_CANONICAL_FIGURE_SHA256.items():
        figure_path = artifact_root / relative_path
        if not figure_path.exists():
            raise AssertionError(f"canonical figure is missing: {figure_path}")
        actual_hash = _sha256(figure_path)
        if actual_hash != expected_hash:
            raise AssertionError(
                f"canonical figure hash mismatch for {relative_path}: "
                f"expected {expected_hash}, computed {actual_hash}"
            )


def _verify_metadata(csv_path: Path, metadata_path: Path, artifact_root: Path) -> None:
    with metadata_path.open() as handle:
        metadata = json.load(handle)

    if metadata.get("artifact") != EXPECTED_ARTIFACT:
        raise AssertionError(
            f"metadata artifact mismatch: expected {EXPECTED_ARTIFACT!r}, "
            f"found {metadata.get('artifact')!r}"
        )
    if metadata.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise AssertionError(
            f"metadata schema_version mismatch: expected {EXPECTED_SCHEMA_VERSION}, "
            f"found {metadata.get('schema_version')!r}"
        )
    if metadata.get("row_count") != EXPECTED_ROW_COUNT:
        raise AssertionError(
            f"metadata row_count mismatch: expected {EXPECTED_ROW_COUNT}, "
            f"found {metadata.get('row_count')!r}"
        )
    parameters = metadata.get("parameters")
    if not isinstance(parameters, dict):
        raise AssertionError("metadata parameters must be an object")
    if set(parameters) != set(EXPECTED_PARAMETERS):
        raise AssertionError(
            "metadata parameter keys mismatch: "
            f"expected {sorted(EXPECTED_PARAMETERS)}, found {sorted(parameters)}"
        )
    for key, expected_value in EXPECTED_PARAMETERS.items():
        actual_value = parameters[key]
        if isinstance(expected_value, int):
            if actual_value != expected_value:
                raise AssertionError(
                    f"metadata parameter {key} mismatch: "
                    f"expected {expected_value}, found {actual_value!r}"
                )
        else:
            _assert_close(
                f"metadata parameter {key}",
                float(actual_value),
                expected_value,
            )

    expected_csv_hash = metadata.get("csv_sha256")
    actual_csv_hash = _sha256(csv_path)
    if expected_csv_hash != actual_csv_hash:
        raise AssertionError(
            f"CSV hash mismatch: metadata has {expected_csv_hash}, "
            f"computed {actual_csv_hash}"
        )
    if expected_csv_hash != EXPECTED_CSV_SHA256:
        raise AssertionError(
            f"metadata CSV hash is not the canonical hash: "
            f"expected {EXPECTED_CSV_SHA256}, found {expected_csv_hash}"
        )

    source = metadata.get("source", {})
    if not isinstance(source, dict):
        raise AssertionError("metadata source must be an object")
    manifest = source.get("source_files_sha256")
    manifest_digest = source.get("source_manifest_sha256")
    if not isinstance(manifest, dict) or not isinstance(manifest_digest, str):
        raise AssertionError("metadata source must include a source-file hash manifest")
    encoded_manifest = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    actual_manifest_digest = hashlib.sha256(encoded_manifest).hexdigest()
    if manifest_digest != actual_manifest_digest:
        raise AssertionError(
            f"source manifest digest mismatch: metadata has {manifest_digest}, "
            f"computed {actual_manifest_digest}"
        )
    if _is_source_checkout_root(artifact_root):
        for relative_path, expected_hash in manifest.items():
            source_path = artifact_root / relative_path
            if not source_path.exists():
                raise AssertionError(
                    f"source file listed in metadata is missing: {source_path}"
                )
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
        actual_hash = _sha256(figure_path)
        if actual_hash != expected_hash:
            raise AssertionError(
                f"figure hash mismatch for {relative_path}: "
                f"metadata has {expected_hash}, computed {actual_hash}"
            )
        canonical_hash = EXPECTED_FIGURE_SHA256_BY_NAME.get(figure_path.name)
        if canonical_hash is None:
            raise AssertionError(f"metadata lists unexpected figure: {relative_path}")
        if expected_hash != canonical_hash:
            raise AssertionError(
                f"metadata figure hash is not canonical for {relative_path}: "
                f"expected {canonical_hash}, found {expected_hash}"
            )
    if _is_canonical_artifact_root(artifact_root) and metadata_path.resolve() == (
        artifact_root / "results" / "qalbench_sweep_metadata.json"
    ).resolve():
        if figure_hashes != EXPECTED_CANONICAL_FIGURE_SHA256:
            raise AssertionError(
                "default metadata figures_sha256 does not match the canonical "
                "results and paper figure manifest"
            )
        if metadata.get("paper_figures_synced") is not True:
            raise AssertionError("default metadata must record paper_figures_synced=true")


def _recompute_row(row: dict[str, str]) -> dict[str, float]:
    qparams = QuantumParams(
        theta=_float(row, "theta"),
        phi=_float(row, "phi"),
        local_perturbation_probability=_float(row, "local_perturbation_probability"),
        local_perturbation_angle=_float(row, "local_perturbation_angle"),
        interaction_angle=_float(row, "interaction_angle"),
        damping_probability=_float(row, "damping_probability"),
        dephase_probability=_float(row, "phase_damping"),
    )
    cparams = ClassicalParams(
        theta=_float(row, "theta"),
        local_perturbation_probability=_float(row, "local_perturbation_probability"),
        local_perturbation_angle=_float(row, "local_perturbation_angle"),
        damping_probability=_float(row, "damping_probability"),
    )
    qstates = run_quantum_event(qparams)
    classical = run_classical_event(cparams)

    expected: dict[str, float] = {}
    expected.update(compute_metrics(qstates["rho"], prefix="quantum_"))
    expected.update(compute_metrics(qstates["dephased"], prefix="dephased_"))
    expected.update(compute_metrics(classical["rho"], prefix="classical_"))
    for bitstring, probability in zip(BITSTRINGS, classical["probabilities"]):
        expected[f"classical_p_{bitstring}"] = float(probability)
    return expected


def verify(
    csv_path: Path,
    metadata_path: Path | None = None,
    *,
    check_metadata: bool = True,
    check_figures: bool = True,
) -> dict[str, float]:
    artifact_root = _artifact_root_for_csv(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV artifact not found: {csv_path}. Run from a repository/archive "
            "root, set QALBENCH_ROOT, pass --csv/--metadata, or install a wheel "
            "that includes the packaged canonical artifacts."
        )
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_FIELDNAMES:
            raise AssertionError(
                "unexpected CSV schema: "
                f"expected {EXPECTED_FIELDNAMES}, found {reader.fieldnames}"
            )
        rows = list(reader)

    if len(rows) != EXPECTED_ROW_COUNT:
        raise AssertionError(f"expected {EXPECTED_ROW_COUNT} rows, found {len(rows)}")

    _assert_single_value(rows, "phi", 0.0)
    _assert_single_value(rows, "interaction_angle", 0.0)
    _assert_single_value(rows, "phase_damping", 0.0)
    _assert_single_value(rows, "local_perturbation_angle", math.pi / 3.0)
    _assert_exact_cartesian_grid(rows)

    _verify_expected_csv_hash(csv_path)
    if check_figures and _is_canonical_artifact_root(artifact_root):
        _verify_expected_figure_hashes(artifact_root)
    elif check_figures:
        print(
            "warning: canonical figure-hash validation skipped for noncanonical "
            "artifact root; verification is degraded",
            file=sys.stderr,
        )
    if check_metadata:
        if metadata_path is None:
            raise AssertionError("metadata checks requested but no metadata path was provided")
        _verify_metadata(csv_path, metadata_path, artifact_root)

    max_probability_gap = 0.0
    max_reference_probability_gap = 0.0
    max_z_gap = 0.0
    max_recompute_gap = 0.0
    for row in rows:
        expected = _recompute_row(row)
        for key, expected_value in expected.items():
            max_recompute_gap = max(
                max_recompute_gap,
                abs(_float(row, key) - expected_value),
            )
        for bitstring in ("00", "01", "10", "11"):
            max_probability_gap = max(
                max_probability_gap,
                abs(
                    _float(row, f"quantum_p_{bitstring}")
                    - _float(row, f"classical_p_{bitstring}")
                ),
                abs(
                    _float(row, f"dephased_p_{bitstring}")
                    - _float(row, f"classical_p_{bitstring}")
                ),
            )
        reference_probabilities = _analytic_diagonal_probabilities(
            theta=_float(row, "theta"),
            local_perturbation_probability=_float(
                row, "local_perturbation_probability"
            ),
            local_perturbation_angle=_float(row, "local_perturbation_angle"),
            damping_probability=_float(row, "damping_probability"),
        )
        for bitstring, expected_probability in reference_probabilities.items():
            max_reference_probability_gap = max(
                max_reference_probability_gap,
                abs(_float(row, f"quantum_p_{bitstring}") - expected_probability),
                abs(_float(row, f"dephased_p_{bitstring}") - expected_probability),
                abs(_float(row, f"classical_p_{bitstring}") - expected_probability),
            )
        for suffix in ("z_genotype", "z_offspring", "z_copy_agreement"):
            max_z_gap = max(
                max_z_gap,
                abs(_float(row, f"quantum_{suffix}") - _float(row, f"classical_{suffix}")),
                abs(_float(row, f"dephased_{suffix}") - _float(row, f"classical_{suffix}")),
            )

    if max_recompute_gap > 1e-7:
        raise AssertionError(f"max recompute gap too large: {max_recompute_gap}")
    if max_probability_gap > 1e-12:
        raise AssertionError(f"max probability gap too large: {max_probability_gap}")
    if max_reference_probability_gap > 1e-12:
        raise AssertionError(
            "max analytic reference probability gap too large: "
            f"{max_reference_probability_gap}"
        )
    if max_z_gap > 1e-12:
        raise AssertionError(f"max Z/copy gap too large: {max_z_gap}")

    bell = _single(
        rows,
        theta=math.pi / 2.0,
        local_perturbation_probability=0.0,
        damping_probability=0.0,
    )
    _assert_close("zero-damping negativity", _float(bell, "quantum_negativity"), 0.5)
    _assert_close("zero-damping concurrence", _float(bell, "quantum_concurrence"), 1.0, 1e-8)
    _assert_close("zero-damping CHSH", _float(bell, "quantum_chsh_max"), 2.0 * math.sqrt(2.0))
    _assert_close("dephased CHSH", _float(bell, "dephased_chsh_max"), 2.0)

    max_bell_damping_gap = 0.0
    for row in _rows_at(
        rows,
        theta=math.pi / 2.0,
        local_perturbation_probability=0.0,
    ):
        gamma = _float(row, "damping_probability")
        expected_negativity = (1.0 - gamma) / 2.0
        expected_concurrence = math.sqrt(max(0.0, 1.0 - gamma))
        expected_chsh = 2.0 * math.sqrt(max(0.0, 2.0 * (1.0 - gamma)))
        max_bell_damping_gap = max(
            max_bell_damping_gap,
            abs(_float(row, "quantum_negativity") - expected_negativity),
            abs(_float(row, "quantum_concurrence") - expected_concurrence),
            abs(_float(row, "quantum_chsh_max") - expected_chsh),
        )
    if max_bell_damping_gap > 1e-8:
        raise AssertionError(
            f"Bell-damping analytic metric gap too large: {max_bell_damping_gap}"
        )

    half_perturbation = _single(
        rows,
        theta=math.pi / 2.0,
        local_perturbation_probability=0.5,
        damping_probability=0.0,
    )
    _assert_close(
        "half-perturbation negativity",
        _float(half_perturbation, "quantum_negativity"),
        math.sqrt(3.0) / 4.0,
    )

    certain_perturbation = _single(
        rows,
        theta=math.pi / 2.0,
        local_perturbation_probability=1.0,
        damping_probability=0.0,
    )
    _assert_close(
        "certain-perturbation negativity",
        _float(certain_perturbation, "quantum_negativity"),
        0.5,
    )

    chsh_boundary = _single(
        rows,
        theta=math.pi / 2.0,
        local_perturbation_probability=0.0,
        damping_probability=0.5,
    )
    _assert_close("CHSH local-bound crossing", _float(chsh_boundary, "quantum_chsh_max"), 2.0)

    fully_damped = _single(
        rows,
        theta=math.pi / 2.0,
        local_perturbation_probability=0.0,
        damping_probability=1.0,
    )
    _assert_close(
        "complete-damping CHSH endpoint",
        _float(fully_damped, "quantum_chsh_max"),
        0.0,
        1e-12,
    )

    return {
        "rows": float(len(rows)),
        "max_recompute_gap": max_recompute_gap,
        "max_probability_gap": max_probability_gap,
        "max_reference_probability_gap": max_reference_probability_gap,
        "max_z_gap": max_z_gap,
        "max_bell_damping_gap": max_bell_damping_gap,
        "zero_damping_chsh": _float(bell, "quantum_chsh_max"),
        "complete_damping_chsh": _float(fully_damped, "quantum_chsh_max"),
        "half_perturbation_negativity": _float(
            half_perturbation, "quantum_negativity"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    artifact_root = default_artifact_root()
    parser.add_argument(
        "--csv",
        default=str(artifact_root / "results" / "qalbench_sweep.csv"),
        help="Path to qalbench_sweep.csv",
    )
    parser.add_argument(
        "--metadata",
        default=str(artifact_root / "results" / "qalbench_sweep_metadata.json"),
        help="Path to qalbench_sweep_metadata.json",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="skip metadata validation and print a degraded-verification warning",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="skip canonical figure-hash validation and print a degraded-verification warning",
    )
    args = parser.parse_args()

    if args.metadata == "":
        args.skip_metadata = True
        metadata_path = None
    else:
        metadata_path = Path(args.metadata)
    summary = verify(
        Path(args.csv),
        metadata_path=metadata_path,
        check_metadata=not args.skip_metadata,
        check_figures=not args.skip_figures,
    )
    if args.skip_metadata:
        print("warning: metadata validation skipped; verification is degraded", file=sys.stderr)
    if args.skip_figures:
        print("warning: canonical figure-hash validation skipped; verification is degraded", file=sys.stderr)
    for key, value in summary.items():
        print(f"{key}: {value:.16g}")


if __name__ == "__main__":
    main()
