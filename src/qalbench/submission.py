"""Submission verification and independent-axis scoring workflows."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .tasks import ClaimAxis, TaskSpec, task_by_id


@dataclass(frozen=True)
class VerificationIssue:
    severity: str
    field: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationReport:
    passed: bool
    task_id: str | None
    issues: tuple[VerificationIssue, ...]
    artifact_checks: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "task_id": self.task_id,
            "issues": [issue.as_dict() for issue in self.issues],
            "artifact_checks": self.artifact_checks,
        }


@dataclass(frozen=True)
class ScoreBreakdown:
    artificial_life_adequacy: float
    quantum_resource_relevance: float
    computational_nonclassicality: float
    certification_readiness: float
    verification_completeness: float
    claimed_axis_floor: float
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load a manifest from a path or return a shallow copy of a mapping."""

    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    with path.open() as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("submission manifest must be a JSON object")
    return loaded


def _artifact_mapping(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = manifest.get("artifacts", {})
    if isinstance(raw, Mapping):
        out: dict[str, Mapping[str, Any]] = {}
        for name, spec in raw.items():
            if isinstance(spec, str):
                out[str(name)] = {"path": spec}
            elif isinstance(spec, Mapping):
                out[str(name)] = spec
            else:
                out[str(name)] = {}
        return out
    if isinstance(raw, list):
        out = {}
        for item in raw:
            if isinstance(item, Mapping) and "name" in item:
                out[str(item["name"])] = item
        return out
    return {}


def _string_set(manifest: Mapping[str, Any], field: str) -> set[str]:
    value = manifest.get(field, [])
    if isinstance(value, str):
        return {value}
    if isinstance(value, list | tuple | set):
        return {str(item) for item in value}
    return set()


def _check_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> tuple[list[VerificationIssue], dict[str, str]]:
    issues: list[VerificationIssue] = []
    checks: dict[str, str] = {}
    for name, spec in artifacts.items():
        path_value = spec.get("path")
        if not path_value:
            issues.append(VerificationIssue("error", f"artifacts.{name}.path", "missing artifact path"))
            continue
        path = Path(str(path_value))
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            issues.append(VerificationIssue("error", f"artifacts.{name}.path", f"not found: {path}"))
            checks[name] = "missing"
            continue
        expected_hash = spec.get("sha256")
        if expected_hash:
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                issues.append(
                    VerificationIssue(
                        "error",
                        f"artifacts.{name}.sha256",
                        f"sha256 mismatch: expected {expected_hash}, got {actual_hash}",
                    )
                )
                checks[name] = "hash_mismatch"
            else:
                checks[name] = "ok"
        else:
            checks[name] = "exists_unhashed"
    return issues, checks


def _artifact_path(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
    name: str,
) -> Path | None:
    spec = artifacts.get(name)
    if not spec or not spec.get("path"):
        return None
    path = Path(str(spec["path"]))
    return path if path.is_absolute() else root / path


def _load_json_artifact(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
    name: str,
) -> Any:
    path = _artifact_path(artifacts, root, name)
    if path is None:
        raise ValueError(f"missing artifact {name!r}")
    with path.open() as handle:
        return json.load(handle)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _rows_from_payload(value: Any, artifact_name: str) -> list[Any]:
    rows = value.get("rows") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError(f"{artifact_name} must contain a rows list")
    return rows


def _require_fields(row: Mapping[str, Any], fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"{label} missing fields {missing}")


def _require_close(actual: Any, expected: Any, field: str) -> None:
    if isinstance(expected, bool):
        if bool(actual) != expected:
            raise ValueError(f"{field} does not match recomputed value")
        return
    if isinstance(expected, str):
        if str(actual) != expected:
            raise ValueError(f"{field} does not match recomputed value")
        return
    if not _is_number(actual) or abs(float(actual) - float(expected)) > 1e-12:
        raise ValueError(f"{field} does not match recomputed value")


def _validate_probability_metrics(metrics: Mapping[str, Any], label: str) -> None:
    required = ("p_00", "p_01", "p_10", "p_11", "z_copy_agreement")
    _require_fields(metrics, required, label)
    total = 0.0
    for field in ("p_00", "p_01", "p_10", "p_11"):
        value = metrics[field]
        if not _is_number(value):
            raise ValueError(f"{label}.{field} must be numeric")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{label}.{field} must be in [0, 1]")
        total += float(value)
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"{label} probabilities must sum to 1, got {total}")


def _validate_t1_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    try:
        basis = _load_json_artifact(artifacts, root, "basis_counts")
        if not isinstance(basis, dict):
            raise ValueError("basis_counts artifact must be a JSON object")
        if basis.get("basis") != "computational":
            raise ValueError("basis_counts.basis must be 'computational'")
        probabilities = basis.get("probabilities")
        if not isinstance(probabilities, dict):
            raise ValueError("basis_counts.probabilities must be an object")
        for model in ("quantum", "dephased", "classical_markov"):
            metrics = probabilities.get(model)
            if not isinstance(metrics, dict):
                raise ValueError(f"basis_counts missing {model!r} probabilities")
            _validate_probability_metrics(metrics, f"basis_counts.{model}")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.basis_counts", str(exc)))

    try:
        parameters = _load_json_artifact(artifacts, root, "event_parameters")
        if not isinstance(parameters, dict):
            raise ValueError("event_parameters artifact must be a JSON object")
        for field in ("theta", "local_perturbation_probability", "damping_probability"):
            if field not in parameters or not _is_number(parameters[field]):
                raise ValueError(f"event_parameters.{field} must be numeric")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.event_parameters", str(exc)))

    try:
        manifest = _load_json_artifact(artifacts, root, "implementation_manifest")
        if not isinstance(manifest, dict):
            raise ValueError("implementation_manifest artifact must be a JSON object")
        if not manifest.get("workflow"):
            raise ValueError("implementation_manifest.workflow is required")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.implementation_manifest", str(exc)))
    return issues


def _validate_t1_mutation_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> list[VerificationIssue]:
    issues = _validate_t1_artifacts(
        {
            key: value
            for key, value in artifacts.items()
            if key in {"basis_counts", "event_parameters", "implementation_manifest"}
        },
        root,
    )
    issues = [issue for issue in issues if issue.field != "artifacts.implementation_manifest"]
    try:
        mutation = _load_json_artifact(artifacts, root, "mutation_records")
        if not isinstance(mutation, dict):
            raise ValueError("mutation_records artifact must be a JSON object")
        rows = _rows_from_payload(mutation, "mutation_records")
        if not rows:
            raise ValueError("mutation_records rows must not be empty")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"mutation row {index} must be an object")
            _require_fields(
                row,
                (
                    "parent_bit",
                    "declared_flip_probability",
                    "quantum_child_p_one",
                    "no_mutation_child_p_one",
                    "classical_child_p_one",
                ),
                f"mutation row {index}",
            )
            for field in ("declared_flip_probability", "quantum_child_p_one", "classical_child_p_one"):
                if not _is_number(row[field]):
                    raise ValueError(f"mutation row {index}.{field} must be numeric")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.mutation_records", str(exc)))
    return issues


def _complex_matrix_shape(value: Any) -> tuple[int, int]:
    if not isinstance(value, list) or not value:
        raise ValueError("matrix must be a nonempty nested list")
    width = None
    for row in value:
        if not isinstance(row, list) or not row:
            raise ValueError("matrix rows must be nonempty lists")
        width = len(row) if width is None else width
        if len(row) != width:
            raise ValueError("matrix rows must have equal length")
        for entry in row:
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or not _is_number(entry[0])
                or not _is_number(entry[1])
            ):
                raise ValueError("matrix entries must be [real, imag] numeric pairs")
    return len(value), int(width)


def _validate_t2_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    try:
        density = _load_json_artifact(artifacts, root, "density_or_witness_estimates")
        if not isinstance(density, dict):
            raise ValueError("density_or_witness_estimates must be a JSON object")
        if density.get("representation") != "exact_density_matrix":
            raise ValueError("density_or_witness_estimates.representation must be exact_density_matrix")
        shape = _complex_matrix_shape(density.get("rho"))
        if shape != (4, 4):
            raise ValueError(f"density matrix must be 4x4, got {shape}")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.density_or_witness_estimates", str(exc)))

    try:
        resource_metrics = _load_json_artifact(artifacts, root, "resource_metrics")
        if not isinstance(resource_metrics, dict):
            raise ValueError("resource_metrics artifact must be a JSON object")
        for model in ("quantum", "dephased", "separable_product"):
            entry = resource_metrics.get(model)
            if not isinstance(entry, dict) or not isinstance(entry.get("metrics"), dict):
                raise ValueError(f"resource_metrics missing metrics for {model!r}")
            metrics = entry["metrics"]
            for field in ("negativity", "chsh_max", "coherence_l1", "purity"):
                if field not in metrics or not _is_number(metrics[field]):
                    raise ValueError(f"resource_metrics.{model}.{field} must be numeric")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.resource_metrics", str(exc)))

    try:
        assumptions = _load_json_artifact(artifacts, root, "diagnostic_assumptions")
        if not isinstance(assumptions, dict):
            raise ValueError("diagnostic_assumptions artifact must be a JSON object")
        if not isinstance(assumptions.get("diagnostics"), list) or not assumptions["diagnostics"]:
            raise ValueError("diagnostic_assumptions.diagnostics must be a nonempty list")
        if not isinstance(assumptions.get("assumptions"), list) or not assumptions["assumptions"]:
            raise ValueError("diagnostic_assumptions.assumptions must be a nonempty list")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.diagnostic_assumptions", str(exc)))
    return issues


def _validate_t2_process_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    try:
        process = _load_json_artifact(artifacts, root, "process_description")
        if not isinstance(process, dict):
            raise ValueError("process_description artifact must be a JSON object")
        if not process.get("process"):
            raise ValueError("process_description.process is required")
        if not isinstance(process.get("operations"), list) or not process["operations"]:
            raise ValueError("process_description.operations must be a nonempty list")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.process_description", str(exc)))

    try:
        ensemble = _load_json_artifact(artifacts, root, "input_ensemble")
        rows = _rows_from_payload(ensemble, "input_ensemble")
        if not rows:
            raise ValueError("input_ensemble rows must not be empty")
        total_weight = 0.0
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"input row {index} must be an object")
            _require_fields(row, ("theta", "weight"), f"input row {index}")
            if not _is_number(row["theta"]) or not _is_number(row["weight"]):
                raise ValueError(f"input row {index} theta and weight must be numeric")
            total_weight += float(row["weight"])
        if abs(total_weight - 1.0) > 1e-8:
            raise ValueError(f"input_ensemble weights must sum to 1, got {total_weight}")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.input_ensemble", str(exc)))

    try:
        metrics = _load_json_artifact(artifacts, root, "resource_metrics")
        rows = _rows_from_payload(metrics, "resource_metrics")
        if not rows:
            raise ValueError("resource_metrics rows must not be empty")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"resource metric row {index} must be an object")
            _require_fields(
                row,
                ("theta", "quantum_negativity", "dephased_negativity", "resource_survival_gap"),
                f"resource metric row {index}",
            )
            if not _is_number(row["resource_survival_gap"]):
                raise ValueError(f"resource metric row {index}.resource_survival_gap must be numeric")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.resource_metrics", str(exc)))
    return issues


def _validate_t3_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    try:
        individuals = _rows_from_payload(
            _load_json_artifact(artifacts, root, "individual_records"),
            "individual_records",
        )
        if not individuals:
            raise ValueError("individual_records rows must not be empty")
        for index, row in enumerate(individuals):
            if not isinstance(row, dict):
                raise ValueError(f"individual row {index} must be an object")
            _require_fields(
                row,
                ("id", "birth_step", "generation", "theta", "expressed_bit"),
                f"individual row {index}",
            )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.individual_records", str(exc)))

    try:
        birth_events = _rows_from_payload(
            _load_json_artifact(artifacts, root, "birth_event_records"),
            "birth_event_records",
        )
        if not birth_events:
            raise ValueError("birth_event_records rows must not be empty")
        for index, row in enumerate(birth_events):
            if not isinstance(row, dict):
                raise ValueError(f"birth-event row {index} must be an object")
            _require_fields(
                row,
                ("step", "parent_id", "child_id", "parent_theta", "child_theta"),
                f"birth-event row {index}",
            )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.birth_event_records", str(exc)))

    try:
        timeseries = _rows_from_payload(
            _load_json_artifact(artifacts, root, "population_timeseries"),
            "population_timeseries",
        )
        if not timeseries:
            raise ValueError("population_timeseries rows must not be empty")
        for index, row in enumerate(timeseries):
            if not isinstance(row, dict):
                raise ValueError(f"population-timeseries row {index} must be an object")
            _require_fields(row, ("step", "alive_population"), f"population-timeseries row {index}")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.population_timeseries", str(exc)))

    try:
        lineage = _load_json_artifact(artifacts, root, "lineage_nulls")
        if not isinstance(lineage, dict):
            raise ValueError("lineage_nulls artifact must be a JSON object")
        for field in ("primary_summary", "no_inheritance_summary", "shuffled_lineage"):
            if field not in lineage:
                raise ValueError(f"lineage_nulls.{field} is required")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.lineage_nulls", str(exc)))
    return issues


def _validate_t3_interaction_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    try:
        interactions = _load_json_artifact(artifacts, root, "interaction_records")
        rows = _rows_from_payload(interactions, "interaction_records")
        if not rows:
            raise ValueError("interaction_records rows must not be empty")
        first = rows[0]
        if not isinstance(first, dict):
            raise ValueError("interaction_records first row must be an object")
        _require_fields(first, ("step", "parent_id", "interaction_exposure", "born"), "interaction row")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.interaction_records", str(exc)))

    try:
        opportunities = _load_json_artifact(artifacts, root, "opportunity_records")
        rows = _rows_from_payload(opportunities, "opportunity_records")
        if not rows:
            raise ValueError("opportunity_records rows must not be empty")
        first = rows[0]
        if not isinstance(first, dict):
            raise ValueError("opportunity_records first row must be an object")
        _require_fields(first, ("step", "parent_id", "birth_probability", "born"), "opportunity row")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.opportunity_records", str(exc)))

    try:
        selection = _load_json_artifact(artifacts, root, "selection_rule")
        if not isinstance(selection, dict):
            raise ValueError("selection_rule artifact must be a JSON object")
        for field in ("scenario", "selection_gradient_birth_rate_bit1_minus_bit0", "interaction_birth_covariance"):
            if field not in selection:
                raise ValueError(f"selection_rule.{field} is required")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.selection_rule", str(exc)))

    try:
        timeseries = _load_json_artifact(artifacts, root, "population_timeseries")
        if not isinstance(timeseries, dict):
            raise ValueError("population_timeseries artifact must be a JSON object")
        if not isinstance(timeseries.get("basis_selection"), list) or not timeseries["basis_selection"]:
            raise ValueError("population_timeseries.basis_selection must be a nonempty list")
        if not isinstance(timeseries.get("neutral_control"), list) or not timeseries["neutral_control"]:
            raise ValueError("population_timeseries.neutral_control must be a nonempty list")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.population_timeseries", str(exc)))
    return issues


def _validate_t4_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    expected_models = ("quantum", "dephased", "classical", "no_inheritance")
    try:
        event_resources = _load_json_artifact(artifacts, root, "event_resource_records")
        if not isinstance(event_resources, dict):
            raise ValueError("event_resource_records artifact must be a JSON object")
        for model in expected_models:
            rows = event_resources.get(model)
            if not isinstance(rows, list):
                raise ValueError(f"event_resource_records.{model} must be a list")
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise ValueError(f"{model} event-resource row {index} must be an object")
                _require_fields(
                    row,
                    ("step", "child_id", "event_resource_score", "event_negativity"),
                    f"{model} event-resource row {index}",
                )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.event_resource_records", str(exc)))

    try:
        opportunities = _load_json_artifact(artifacts, root, "opportunity_records")
        if not isinstance(opportunities, dict):
            raise ValueError("opportunity_records artifact must be a JSON object")
        for model in expected_models:
            rows = opportunities.get(model)
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"opportunity_records.{model} must be a nonempty list")
            first = rows[0]
            if not isinstance(first, dict):
                raise ValueError(f"opportunity_records.{model}[0] must be an object")
            _require_fields(first, ("step", "parent_id", "birth_probability", "born"), f"{model} opportunity")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.opportunity_records", str(exc)))

    try:
        outcomes = _load_json_artifact(artifacts, root, "population_outcomes")
        if not isinstance(outcomes, dict):
            raise ValueError("population_outcomes artifact must be a JSON object")
        for model in expected_models:
            summary = outcomes.get(model)
            if not isinstance(summary, dict):
                raise ValueError(f"population_outcomes.{model} must be an object")
            _require_fields(summary, ("birth_count", "mean_event_resource_score"), f"{model} outcome")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.population_outcomes", str(exc)))

    try:
        effect = _load_json_artifact(artifacts, root, "effect_size_summary")
        if not isinstance(effect, dict):
            raise ValueError("effect_size_summary artifact must be a JSON object")
        for field in (
            "resource_score_quantum_minus_dephased",
            "birth_count_quantum_minus_classical",
            "theta_correlation_quantum_minus_no_inheritance",
        ):
            if field not in effect or not _is_number(effect[field]):
                raise ValueError(f"effect_size_summary.{field} must be numeric")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.effect_size_summary", str(exc)))
    return issues


def _validate_t4_transmission_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    expected = ("quantum", "resource_preserving_no_inheritance")
    try:
        event_resources = _load_json_artifact(artifacts, root, "event_resource_records")
        if not isinstance(event_resources, dict):
            raise ValueError("event_resource_records artifact must be a JSON object")
        for model in expected:
            rows = event_resources.get(model)
            if not isinstance(rows, list):
                raise ValueError(f"event_resource_records.{model} must be a list")
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise ValueError(f"{model} event-resource row {index} must be an object")
                _require_fields(row, ("step", "child_id", "event_resource_score"), f"{model} event row")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.event_resource_records", str(exc)))

    for artifact_name in ("individual_records", "birth_event_records"):
        try:
            payload = _load_json_artifact(artifacts, root, artifact_name)
            if not isinstance(payload, dict):
                raise ValueError(f"{artifact_name} artifact must be a JSON object")
            for model in expected:
                rows = payload.get(model)
                if not isinstance(rows, list) or not rows:
                    raise ValueError(f"{artifact_name}.{model} must be a nonempty list")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            issues.append(VerificationIssue("error", f"artifacts.{artifact_name}", str(exc)))

    try:
        lineage = _load_json_artifact(artifacts, root, "lineage_nulls")
        if not isinstance(lineage, dict):
            raise ValueError("lineage_nulls artifact must be a JSON object")
        for field in (
            "quantum_summary",
            "resource_preserving_no_inheritance_summary",
            "resource_score_control_gap",
            "theta_correlation_control_gap",
        ):
            if field not in lineage:
                raise ValueError(f"lineage_nulls.{field} is required")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.lineage_nulls", str(exc)))
    return issues


def _validate_readout_calibration_schema(calibration: Mapping[str, Any]) -> None:
    mitigation = calibration.get("readout_mitigation")
    if mitigation not in {"none", "inverse_confusion_matrix"}:
        raise ValueError(
            "calibration_record.readout_mitigation must be 'none' or 'inverse_confusion_matrix'"
        )
    readout_model = calibration.get("readout_error_model")
    if not isinstance(readout_model, dict):
        raise ValueError("calibration_record.readout_error_model must be an object")
    if mitigation == "none":
        if readout_model.get("type") not in {"ideal_readout_assumption", "not_supplied"}:
            raise ValueError(
                "calibration_record.readout_error_model.type must state ideal_readout_assumption or not_supplied"
            )
        return

    outcomes = readout_model.get("outcomes")
    confusion = readout_model.get("confusion_matrix")
    inverse = readout_model.get("inverse_confusion_matrix")
    if not isinstance(outcomes, list) or not outcomes:
        raise ValueError("readout_error_model.outcomes must be a nonempty list")
    if len(set(map(str, outcomes))) != len(outcomes):
        raise ValueError("readout_error_model.outcomes must be unique")
    dimension = len(outcomes)
    for matrix_name, matrix in (
        ("confusion_matrix", confusion),
        ("inverse_confusion_matrix", inverse),
    ):
        if not isinstance(matrix, list) or len(matrix) != dimension:
            raise ValueError(f"readout_error_model.{matrix_name} must be square")
        for row_index, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != dimension:
                raise ValueError(f"readout_error_model.{matrix_name} row {row_index} must have length {dimension}")
            for value in row:
                if not _is_number(value):
                    raise ValueError(f"readout_error_model.{matrix_name} entries must be numeric")
        if matrix_name == "confusion_matrix":
            for row_index, row in enumerate(matrix):
                if any(float(value) < 0.0 for value in row):
                    raise ValueError(f"readout_error_model.confusion_matrix row {row_index} has negative entries")
                row_sum = sum(float(value) for value in row)
                if abs(row_sum - 1.0) > 1e-8:
                    raise ValueError(
                        f"readout_error_model.confusion_matrix row {row_index} must sum to 1"
                    )
    calibration_shots = readout_model.get("calibration_shots")
    if not isinstance(calibration_shots, int) or calibration_shots <= 0:
        raise ValueError("readout_error_model.calibration_shots must be a positive integer")


def _validate_t5_resource_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    shot_counts_payload: Any | None = None
    intervals_payload: Any | None = None
    try:
        shot_counts = _load_json_artifact(artifacts, root, "shot_counts")
        shot_counts_payload = shot_counts
        from .certification import validate_counts

        if isinstance(shot_counts, dict) and "setting_counts" in shot_counts:
            setting_counts = shot_counts["setting_counts"]
            if not isinstance(setting_counts, dict):
                raise ValueError("setting_counts must be an object")
            for setting, counts in setting_counts.items():
                if not isinstance(counts, dict):
                    raise ValueError(f"setting {setting!r} counts must be an object")
                validate_counts(counts)
        elif isinstance(shot_counts, dict) and "counts" in shot_counts:
            counts = shot_counts["counts"]
            if not isinstance(counts, dict):
                raise ValueError("counts must be an object")
            validate_counts(counts)
        elif isinstance(shot_counts, dict):
            validate_counts(shot_counts)
        else:
            raise ValueError("shot_counts artifact must be a JSON object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.shot_counts", str(exc)))

    try:
        intervals = _load_json_artifact(artifacts, root, "confidence_intervals")
        intervals_payload = intervals
        if not isinstance(intervals, dict):
            raise ValueError("confidence_intervals artifact must be a JSON object")
        if not any(
            key in intervals
            for key in (
                "copy_agreement",
                "histogram_probabilities",
                "chsh",
                "certificates",
                "intervals",
            )
        ):
            raise ValueError(
                "confidence_intervals must contain copy_agreement, "
                "histogram_probabilities, chsh, certificates, or intervals"
            )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.confidence_intervals", str(exc)))

    try:
        if isinstance(shot_counts_payload, dict) and isinstance(intervals_payload, dict):
            from .certification import (
                chsh_certificate,
                copy_agreement_certificate,
                histogram_probabilities,
            )

            if "copy_agreement" in intervals_payload:
                counts = shot_counts_payload.get("counts", shot_counts_payload)
                confidence = float(intervals_payload["copy_agreement"].get("confidence", 0.95))
                recomputed = copy_agreement_certificate(counts, confidence=confidence).as_dict()
                for field in ("estimate", "lower", "upper", "confidence", "shots"):
                    _require_close(intervals_payload["copy_agreement"].get(field), recomputed[field], f"copy_agreement.{field}")
            if "histogram_probabilities" in intervals_payload:
                counts = shot_counts_payload.get("counts", shot_counts_payload)
                first_interval = next(iter(intervals_payload["histogram_probabilities"].values()))
                confidence = float(first_interval.get("confidence", 0.95)) if isinstance(first_interval, dict) else 0.95
                recomputed_histogram = {
                    outcome: interval.as_dict()
                    for outcome, interval in histogram_probabilities(counts, confidence=confidence).items()
                }
                for outcome, recomputed in recomputed_histogram.items():
                    submitted = intervals_payload["histogram_probabilities"].get(outcome)
                    if not isinstance(submitted, dict):
                        raise ValueError(f"histogram_probabilities.{outcome} is missing")
                    for field in ("estimate", "lower", "upper", "confidence", "shots"):
                        _require_close(submitted.get(field), recomputed[field], f"histogram_probabilities.{outcome}.{field}")
            if "chsh" in intervals_payload:
                setting_counts = shot_counts_payload.get("setting_counts", shot_counts_payload)
                confidence = float(intervals_payload["chsh"].get("confidence", 0.95))
                recomputed_chsh = chsh_certificate(setting_counts, confidence=confidence).as_dict()
                for field in ("estimate", "lower", "upper", "confidence", "local_bound", "certified"):
                    _require_close(intervals_payload["chsh"].get(field), recomputed_chsh[field], f"chsh.{field}")
    except (TypeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.confidence_intervals", str(exc)))

    try:
        witness = _load_json_artifact(artifacts, root, "witness_estimates")
        if not isinstance(witness, dict):
            raise ValueError("witness_estimates artifact must be a JSON object")
        if not witness.get("kind"):
            raise ValueError("witness_estimates.kind is required")
        if not isinstance(witness.get("certificate"), dict):
            raise ValueError("witness_estimates.certificate must be an object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.witness_estimates", str(exc)))

    try:
        calibration = _load_json_artifact(artifacts, root, "calibration_record")
        if not isinstance(calibration, dict):
            raise ValueError("calibration_record artifact must be a JSON object")
        if not calibration.get("readout_mitigation"):
            raise ValueError("calibration_record.readout_mitigation is required")
        assumptions = calibration.get("assumptions")
        if not isinstance(assumptions, list) or not assumptions:
            raise ValueError("calibration_record.assumptions must be a nonempty list")
        _validate_readout_calibration_schema(calibration)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.calibration_record", str(exc)))
    return issues


def _validate_t5_lineage_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    groups: dict[str, dict[str, int]] = {}
    lineage_payload: Mapping[str, Any] | None = None
    try:
        shot_counts = _load_json_artifact(artifacts, root, "shot_counts")
        from .certification import validate_counts

        if not isinstance(shot_counts, dict):
            raise ValueError("shot_counts artifact must be a JSON object")
        raw_groups = shot_counts.get("lineage_counts")
        if not isinstance(raw_groups, dict):
            raise ValueError("shot_counts.lineage_counts must be an object")
        required_groups = {"observed", "no_inheritance", "shuffled_lineage"}
        missing = required_groups - set(raw_groups)
        if missing:
            raise ValueError(f"shot_counts.lineage_counts missing groups {sorted(missing)}")
        for group_name, counts in raw_groups.items():
            if not isinstance(counts, dict):
                raise ValueError(f"lineage_counts.{group_name} must be an object")
            validate_counts(counts)
            normalized: dict[str, int] = {}
            for outcome, count in counts.items():
                if len(str(outcome)) < 2 or str(outcome)[0] not in "01" or str(outcome)[1] not in "01":
                    raise ValueError(f"lineage outcome {outcome!r} must start with parent/child bits")
                normalized[str(outcome)] = int(count)
            groups[str(group_name)] = normalized
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.shot_counts", str(exc)))

    try:
        records = _load_json_artifact(artifacts, root, "individual_or_register_records")
        rows = _rows_from_payload(records, "individual_or_register_records")
        if not rows:
            raise ValueError("individual_or_register_records rows must not be empty")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"register row {index} must be an object")
            _require_fields(row, ("group", "parent_bit", "child_bit", "count"), f"register row {index}")
            if row["group"] not in groups:
                raise ValueError(f"register row {index} has unknown group {row['group']!r}")
            if row["parent_bit"] not in (0, 1) or row["child_bit"] not in (0, 1):
                raise ValueError(f"register row {index} parent_bit/child_bit must be 0 or 1")
            if not isinstance(row["count"], int) or row["count"] < 0:
                raise ValueError(f"register row {index}.count must be a nonnegative integer")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.individual_or_register_records", str(exc)))

    try:
        intervals = _load_json_artifact(artifacts, root, "confidence_intervals")
        if not isinstance(intervals, dict):
            raise ValueError("confidence_intervals artifact must be a JSON object")
        lineage = intervals.get("lineage")
        if not isinstance(lineage, dict):
            raise ValueError("confidence_intervals.lineage must be an object")
        lineage_payload = lineage
        certificate_groups = lineage.get("groups")
        if not isinstance(certificate_groups, dict):
            raise ValueError("confidence_intervals.lineage.groups must be an object")
        for group_name in ("observed", "no_inheritance", "shuffled_lineage"):
            certificate = certificate_groups.get(group_name)
            if not isinstance(certificate, dict):
                raise ValueError(f"confidence_intervals.lineage.groups.{group_name} must be an object")
            for interval_name in (
                "copy_agreement_interval",
                "mutation_rate_interval",
                "lineage_mi_interval",
            ):
                interval = certificate.get(interval_name)
                if not isinstance(interval, dict):
                    raise ValueError(f"{group_name}.{interval_name} must be an object")
                for field in ("estimate", "lower", "upper", "confidence", "shots", "method"):
                    if field not in interval:
                        raise ValueError(f"{group_name}.{interval_name}.{field} is required")
        if not isinstance(lineage.get("certified_inheritance"), bool):
            raise ValueError("confidence_intervals.lineage.certified_inheritance must be boolean")
        if groups:
            from .certification import finite_shot_lineage_certificate

            confidence = float(lineage.get("confidence", 0.95))
            recomputed_lineage = finite_shot_lineage_certificate(groups, confidence=confidence)
            for field in (
                "confidence",
                "observed_copy_agreement_lower",
                "control_copy_agreement_upper",
                "certified_inheritance",
            ):
                _require_close(lineage.get(field), recomputed_lineage[field], f"lineage.{field}")
            recomputed_groups = recomputed_lineage["groups"]
            for group_name in ("observed", "no_inheritance", "shuffled_lineage"):
                submitted_group = certificate_groups[group_name]
                recomputed_group = recomputed_groups[group_name]
                for interval_name in (
                    "copy_agreement_interval",
                    "mutation_rate_interval",
                    "parent_one_interval",
                    "child_one_interval",
                    "lineage_mi_interval",
                ):
                    submitted_interval = submitted_group.get(interval_name)
                    recomputed_interval = recomputed_group[interval_name]
                    if not isinstance(submitted_interval, dict):
                        raise ValueError(f"{group_name}.{interval_name} must be an object")
                    for field in ("estimate", "lower", "upper", "confidence", "shots", "method"):
                        _require_close(
                            submitted_interval.get(field),
                            recomputed_interval[field],
                            f"lineage.groups.{group_name}.{interval_name}.{field}",
                        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.confidence_intervals", str(exc)))

    try:
        stats = _load_json_artifact(artifacts, root, "lineage_statistics")
        if not isinstance(stats, dict):
            raise ValueError("lineage_statistics artifact must be a JSON object")
        certificate = stats.get("certificate")
        if not isinstance(certificate, dict):
            raise ValueError("lineage_statistics.certificate must be an object")
        if not isinstance(certificate.get("certified_inheritance"), bool):
            raise ValueError("lineage_statistics.certificate.certified_inheritance must be boolean")
        if lineage_payload is not None:
            for field in (
                "confidence",
                "observed_copy_agreement_lower",
                "control_copy_agreement_upper",
                "certified_inheritance",
            ):
                _require_close(certificate.get(field), lineage_payload[field], f"lineage_statistics.certificate.{field}")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.lineage_statistics", str(exc)))

    try:
        calibration = _load_json_artifact(artifacts, root, "calibration_record")
        if not isinstance(calibration, dict):
            raise ValueError("calibration_record artifact must be a JSON object")
        if not calibration.get("readout_mitigation"):
            raise ValueError("calibration_record.readout_mitigation is required")
        assumptions = calibration.get("assumptions")
        if not isinstance(assumptions, list) or not assumptions:
            raise ValueError("calibration_record.assumptions must be a nonempty list")
        _validate_readout_calibration_schema(calibration)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.calibration_record", str(exc)))
    return issues


def _validate_t6_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    try:
        scaling_spec = _load_json_artifact(artifacts, root, "scaling_spec")
        if not isinstance(scaling_spec, dict):
            raise ValueError("scaling_spec artifact must be a JSON object")
        if not scaling_spec.get("scaling_variable"):
            raise ValueError("scaling_spec must declare scaling_variable")
        sizes = scaling_spec.get("sizes")
        if not isinstance(sizes, list) or not sizes:
            raise ValueError("scaling_spec must contain nonempty sizes list")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.scaling_spec", str(exc)))

    try:
        baseline_results = _load_json_artifact(
            artifacts,
            root,
            "classical_baseline_results",
        )
        rows = baseline_results.get("rows") if isinstance(baseline_results, dict) else baseline_results
        if not isinstance(rows, list) or not rows:
            raise ValueError("classical_baseline_results must contain nonempty rows")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"baseline row {index} must be an object")
            if not row.get("baseline_id"):
                raise ValueError(f"baseline row {index} missing baseline_id")
            if "site_count" not in row and "size" not in row and "n" not in row:
                raise ValueError(f"baseline row {index} missing size/site_count")
            if row.get("error_metric") == "not_run":
                issues.append(
                    VerificationIssue(
                        "warning",
                        "artifacts.classical_baseline_results",
                        f"baseline row {index} is declared but not run",
                    )
                )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(
            VerificationIssue("error", "artifacts.classical_baseline_results", str(exc))
        )
    return issues


def _validate_t6_sampling_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
    task: TaskSpec,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    sizes: set[int] = set()
    baseline_errors_by_size: dict[int, dict[str, float]] = {}
    try:
        scaling_spec = _load_json_artifact(artifacts, root, "scaling_spec")
        raw_sizes = scaling_spec.get("sizes") if isinstance(scaling_spec, dict) else None
        if not isinstance(raw_sizes, list) or not raw_sizes:
            raise ValueError("scaling_spec.sizes must be a nonempty list")
        sizes = {int(size) for size in raw_sizes}
        if any(size <= 0 for size in sizes):
            raise ValueError("all scaling_spec sizes must be positive")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.scaling_spec", str(exc)))

    shot_budget_by_size: dict[int, int] = {}
    try:
        shot_budget = _load_json_artifact(artifacts, root, "shot_budget")
        if not isinstance(shot_budget, dict):
            raise ValueError("shot_budget artifact must be a JSON object")
        total_shots = shot_budget.get("total_shots")
        if not isinstance(total_shots, int) or total_shots <= 0:
            raise ValueError("shot_budget.total_shots must be a positive integer")
        raw_by_size = shot_budget.get("by_size")
        if not isinstance(raw_by_size, dict):
            raise ValueError("shot_budget.by_size must be an object")
        for size_key, count in raw_by_size.items():
            if not isinstance(count, int) or count <= 0:
                raise ValueError(f"shot_budget.by_size.{size_key} must be a positive integer")
            shot_budget_by_size[int(size_key)] = count
        if sum(shot_budget_by_size.values()) != total_shots:
            raise ValueError("shot_budget.total_shots must equal the by_size sum")
        if sizes and set(shot_budget_by_size) != sizes:
            raise ValueError("shot_budget.by_size must cover exactly the scaling sizes")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.shot_budget", str(exc)))

    try:
        sampler = _load_json_artifact(artifacts, root, "sampler_output")
        rows = _rows_from_payload(sampler, "sampler_output")
        if not rows:
            raise ValueError("sampler_output rows must not be empty")
        counts_by_size = {size: 0 for size in sizes}
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"sampler row {index} must be an object")
            _require_fields(row, ("size", "outcome", "count", "shot_count"), f"sampler row {index}")
            size = int(row["size"])
            if sizes and size not in sizes:
                raise ValueError(f"sampler row {index} has size outside scaling_spec")
            outcome = str(row["outcome"])
            if len(outcome) != size or any(bit not in "01" for bit in outcome):
                raise ValueError(f"sampler row {index}.outcome must be a bitstring of length size")
            if not isinstance(row["count"], int) or row["count"] < 0:
                raise ValueError(f"sampler row {index}.count must be a nonnegative integer")
            counts_by_size[size] = counts_by_size.get(size, 0) + row["count"]
        for size, total in counts_by_size.items():
            if shot_budget_by_size and shot_budget_by_size.get(size) != total:
                raise ValueError(f"sampler_output counts for size {size} do not match shot_budget")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.sampler_output", str(exc)))

    try:
        protocol = _load_json_artifact(artifacts, root, "verification_protocol")
        if not isinstance(protocol, dict):
            raise ValueError("verification_protocol artifact must be a JSON object")
        for field in ("distance_metric", "acceptance_rule", "required_baselines"):
            if field not in protocol:
                raise ValueError(f"verification_protocol.{field} is required")
        required_baselines = protocol.get("required_baselines")
        if not isinstance(required_baselines, list):
            raise ValueError("verification_protocol.required_baselines must be a list")
        missing = set(task.baseline_families) - {str(value) for value in required_baselines}
        if missing:
            raise ValueError(f"verification_protocol missing baselines {sorted(missing)}")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.verification_protocol", str(exc)))

    try:
        baseline_results = _load_json_artifact(artifacts, root, "classical_baseline_results")
        rows = baseline_results.get("rows") if isinstance(baseline_results, dict) else baseline_results
        if not isinstance(rows, list) or not rows:
            raise ValueError("classical_baseline_results rows must not be empty")
        seen: set[tuple[str, int]] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"baseline row {index} must be an object")
            baseline_id = str(row.get("baseline_id", ""))
            if baseline_id not in task.baseline_families:
                raise ValueError(f"baseline row {index} has unexpected baseline_id {baseline_id!r}")
            size = int(row.get("size", row.get("site_count", 0)))
            if sizes and size not in sizes:
                raise ValueError(f"baseline row {index} has size outside scaling_spec")
            if not _is_number(row.get("runtime_seconds")):
                raise ValueError(f"baseline row {index}.runtime_seconds must be numeric")
            if row.get("error_metric") == "not_run":
                raise ValueError(f"baseline row {index} is declared but not run")
            if not _is_number(row.get("error_value")):
                raise ValueError(f"baseline row {index}.error_value must be numeric")
            baseline_errors_by_size.setdefault(size, {})[baseline_id] = float(row["error_value"])
            seen.add((baseline_id, size))
        missing_rows = {
            (baseline_id, size)
            for baseline_id in task.baseline_families
            for size in sizes
        } - seen
        if missing_rows:
            raise ValueError(f"classical_baseline_results missing rows {sorted(missing_rows)}")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.classical_baseline_results", str(exc)))

    try:
        evidence = _load_json_artifact(artifacts, root, "nonclassicality_evidence")
        if not isinstance(evidence, dict):
            raise ValueError("nonclassicality_evidence artifact must be a JSON object")
        supported = evidence.get("nonclassicality_claim_supported")
        if not isinstance(supported, bool):
            raise ValueError("nonclassicality_evidence.nonclassicality_claim_supported must be boolean")
        compute_budget = evidence.get("compute_budget")
        if not isinstance(compute_budget, dict):
            raise ValueError("nonclassicality_evidence.compute_budget must be an object")
        for field in ("total_wall_time_seconds", "max_memory_mb"):
            if not _is_number(compute_budget.get(field)):
                raise ValueError(f"nonclassicality_evidence.compute_budget.{field} must be numeric")
        if supported:
            if evidence.get("claim_type") != "baseline_failure":
                raise ValueError(
                    "supported nonclassicality evidence must declare claim_type baseline_failure"
                )
            if not evidence.get("classical_failure_criterion"):
                raise ValueError("supported nonclassicality evidence must declare classical_failure_criterion")
            minimum_sizes = evidence.get("minimum_sizes")
            if not isinstance(minimum_sizes, int) or minimum_sizes < 3:
                raise ValueError("supported nonclassicality evidence requires minimum_sizes >= 3")
            if len(sizes) < minimum_sizes:
                raise ValueError("scaling_spec does not meet nonclassicality minimum_sizes")
        size_results = evidence.get("size_results")
        if not isinstance(size_results, list) or not size_results:
            raise ValueError("nonclassicality_evidence.size_results must be a nonempty list")
        evidence_sizes = set()
        for index, row in enumerate(size_results):
            if not isinstance(row, dict):
                raise ValueError(f"evidence size row {index} must be an object")
            for field in (
                "size",
                "sample_distance",
                "allowed_error",
                "all_required_baselines_completed",
                "passed_acceptance",
            ):
                if field not in row:
                    raise ValueError(f"evidence size row {index}.{field} is required")
            if not _is_number(row["sample_distance"]) or not _is_number(row["allowed_error"]):
                raise ValueError(f"evidence size row {index} distances must be numeric")
            if not isinstance(row["all_required_baselines_completed"], bool):
                raise ValueError(f"evidence size row {index}.all_required_baselines_completed must be boolean")
            if not isinstance(row["passed_acceptance"], bool):
                raise ValueError(f"evidence size row {index}.passed_acceptance must be boolean")
            if not _is_number(row.get("wall_time_seconds")) or not _is_number(row.get("memory_mb")):
                raise ValueError(f"evidence size row {index} must disclose wall_time_seconds and memory_mb")
            if supported:
                failures = row.get("baseline_failures")
                best_classical_error = row.get("best_classical_error")
                if not _is_number(best_classical_error):
                    raise ValueError(f"evidence size row {index}.best_classical_error must be numeric")
                baseline_errors = baseline_errors_by_size.get(int(row["size"]), {})
                missing_baseline_errors = set(task.baseline_families) - set(baseline_errors)
                if missing_baseline_errors:
                    raise ValueError(
                        f"evidence size row {index} missing baseline errors {sorted(missing_baseline_errors)}"
                    )
                recomputed_best_error = min(baseline_errors.values()) if baseline_errors else None
                if recomputed_best_error is None or abs(float(best_classical_error) - recomputed_best_error) > 1e-12:
                    raise ValueError(
                        f"evidence size row {index}.best_classical_error does not match baseline results"
                    )
                failed_baselines = {
                    baseline_id
                    for baseline_id, error_value in baseline_errors.items()
                    if error_value > float(row["allowed_error"])
                }
                if not isinstance(failures, list):
                    raise ValueError(f"evidence size row {index}.baseline_failures must be a list")
                if set(map(str, failures)) != failed_baselines:
                    raise ValueError(
                        f"evidence size row {index}.baseline_failures must match baseline errors above allowed_error"
                    )
                if float(best_classical_error) <= float(row["allowed_error"]):
                    raise ValueError(
                        f"evidence size row {index} must show every required baseline above allowed_error"
                    )
            evidence_sizes.add(int(row["size"]))
        if sizes and evidence_sizes != sizes:
            raise ValueError("nonclassicality_evidence.size_results must cover exactly the scaling sizes")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.nonclassicality_evidence", str(exc)))
    return issues


def _validate_t6_simulator_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    root: Path,
    task: TaskSpec,
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    sizes: set[int] = set()
    baseline_errors_by_size: dict[int, dict[str, float]] = {}
    try:
        scaling_spec = _load_json_artifact(artifacts, root, "scaling_spec")
        raw_sizes = scaling_spec.get("sizes") if isinstance(scaling_spec, dict) else None
        if not isinstance(raw_sizes, list) or not raw_sizes:
            raise ValueError("scaling_spec.sizes must be a nonempty list")
        sizes = {int(size) for size in raw_sizes}
        if any(size < 2 for size in sizes):
            raise ValueError("simulator scaling sizes must be at least 2")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.scaling_spec", str(exc)))

    try:
        metrics = _load_json_artifact(artifacts, root, "resource_metrics")
        rows = _rows_from_payload(metrics, "resource_metrics")
        if not rows:
            raise ValueError("resource_metrics rows must not be empty")
        metric_sizes: set[int] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"resource metric row {index} must be an object")
            for field in (
                "site_count",
                "state_dimension",
                "coherent_global_coherence_l1",
                "coherent_max_pair_negativity",
                "dephased_global_coherence_l1",
            ):
                if field not in row:
                    raise ValueError(f"resource metric row {index}.{field} is required")
            site_count = int(row["site_count"])
            if sizes and site_count not in sizes:
                raise ValueError(f"resource metric row {index} has site_count outside scaling_spec")
            metric_sizes.add(site_count)
            for field in (
                "state_dimension",
                "coherent_global_coherence_l1",
                "coherent_max_pair_negativity",
                "dephased_global_coherence_l1",
            ):
                if not _is_number(row[field]):
                    raise ValueError(f"resource metric row {index}.{field} must be numeric")
        if sizes and metric_sizes != sizes:
            raise ValueError("resource_metrics rows must cover exactly the scaling sizes")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.resource_metrics", str(exc)))

    try:
        baseline_results = _load_json_artifact(artifacts, root, "classical_baseline_results")
        rows = baseline_results.get("rows") if isinstance(baseline_results, dict) else baseline_results
        if not isinstance(rows, list) or not rows:
            raise ValueError("classical_baseline_results rows must not be empty")
        seen: set[tuple[str, int]] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"baseline row {index} must be an object")
            baseline_id = str(row.get("baseline_id", ""))
            if baseline_id not in task.baseline_families:
                raise ValueError(f"baseline row {index} has unexpected baseline_id {baseline_id!r}")
            site_count = int(row.get("site_count", row.get("size", 0)))
            if sizes and site_count not in sizes:
                raise ValueError(f"baseline row {index} has site_count outside scaling_spec")
            if row.get("error_metric") == "not_run":
                raise ValueError(f"baseline row {index} is declared but not run")
            if row.get("error_value") is None and row.get("error_metric") != "reference":
                raise ValueError(f"baseline row {index}.error_value is required")
            if row.get("error_value") is not None and not _is_number(row.get("error_value")):
                raise ValueError(f"baseline row {index}.error_value must be numeric")
            if row.get("runtime_seconds") is not None and not _is_number(row.get("runtime_seconds")):
                raise ValueError(f"baseline row {index}.runtime_seconds must be numeric when supplied")
            baseline_errors_by_size.setdefault(site_count, {})[baseline_id] = float(row.get("error_value") or 0.0)
            seen.add((baseline_id, site_count))
        missing_rows = {
            (baseline_id, site_count)
            for baseline_id in task.baseline_families
            for site_count in sizes
        } - seen
        if missing_rows:
            raise ValueError(f"classical_baseline_results missing rows {sorted(missing_rows)}")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.classical_baseline_results", str(exc)))

    try:
        reproducibility = _load_json_artifact(artifacts, root, "reproducibility_manifest")
        if not isinstance(reproducibility, dict):
            raise ValueError("reproducibility_manifest artifact must be a JSON object")
        if not reproducibility.get("workflow"):
            raise ValueError("reproducibility_manifest.workflow is required")
        if not isinstance(reproducibility.get("parameters"), dict):
            raise ValueError("reproducibility_manifest.parameters must be an object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.reproducibility_manifest", str(exc)))

    try:
        protocol = _load_json_artifact(artifacts, root, "verification_protocol")
        if not isinstance(protocol, dict):
            raise ValueError("verification_protocol artifact must be a JSON object")
        for field in ("acceptance_rule", "required_baselines", "required_controls"):
            if field not in protocol:
                raise ValueError(f"verification_protocol.{field} is required")
        if set(map(str, protocol["required_baselines"])) != set(task.baseline_families):
            raise ValueError("verification_protocol.required_baselines must match task baselines")
        if set(map(str, protocol["required_controls"])) != set(task.required_controls):
            raise ValueError("verification_protocol.required_controls must match task controls")
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        issues.append(VerificationIssue("error", "artifacts.verification_protocol", str(exc)))

    try:
        path = _artifact_path(artifacts, root, "population_timeseries")
        if path is None:
            raise ValueError("missing artifact 'population_timeseries'")
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("population_timeseries rows must not be empty")
        seen_models: set[tuple[str, int]] = set()
        final_rows: dict[tuple[str, int], Mapping[str, str]] = {}
        for index, row in enumerate(rows):
            model = row.get("model", "")
            if model not in {"coherent", "dephased"}:
                raise ValueError(f"population_timeseries row {index}.model must be coherent or dephased")
            site_count = int(row.get("site_count", "0"))
            if sizes and site_count not in sizes:
                raise ValueError(f"population_timeseries row {index} has site_count outside scaling_spec")
            step = int(row.get("step", "-1"))
            for field in ("trace", "global_coherence_l1", "max_pair_negativity"):
                float(row[field])
            seen_models.add((model, site_count))
            key = (model, site_count)
            if key not in final_rows or step > int(final_rows[key].get("step", "-1")):
                final_rows[key] = row
        expected_models = {(model, site_count) for model in ("coherent", "dephased") for site_count in sizes}
        if sizes and seen_models != expected_models:
            raise ValueError("population_timeseries must include coherent and dephased rows for every size")
        metrics = _load_json_artifact(artifacts, root, "resource_metrics")
        metric_rows = _rows_from_payload(metrics, "resource_metrics")
        for metric_row in metric_rows:
            if not isinstance(metric_row, dict):
                continue
            site_count = int(metric_row["site_count"])
            coherent_final = final_rows.get(("coherent", site_count))
            dephased_final = final_rows.get(("dephased", site_count))
            if coherent_final is None or dephased_final is None:
                raise ValueError(f"population_timeseries missing final rows for site_count {site_count}")
            _require_close(
                metric_row["coherent_global_coherence_l1"],
                float(coherent_final["global_coherence_l1"]),
                f"resource_metrics.site_count_{site_count}.coherent_global_coherence_l1",
            )
            _require_close(
                metric_row["dephased_global_coherence_l1"],
                float(dephased_final["global_coherence_l1"]),
                f"resource_metrics.site_count_{site_count}.dephased_global_coherence_l1",
            )
    except (OSError, KeyError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.population_timeseries", str(exc)))

    try:
        evidence = _load_json_artifact(artifacts, root, "nonclassicality_evidence")
        if not isinstance(evidence, dict):
            raise ValueError("nonclassicality_evidence artifact must be a JSON object")
        supported = evidence.get("nonclassicality_claim_supported")
        if not isinstance(supported, bool):
            raise ValueError("nonclassicality_evidence.nonclassicality_claim_supported must be boolean")
        compute_budget = evidence.get("compute_budget")
        if not isinstance(compute_budget, dict):
            raise ValueError("nonclassicality_evidence.compute_budget must be an object")
        for field in ("total_wall_time_seconds", "max_memory_mb"):
            if not _is_number(compute_budget.get(field)):
                raise ValueError(f"nonclassicality_evidence.compute_budget.{field} must be numeric")
        size_results = evidence.get("size_results")
        if not isinstance(size_results, list) or not size_results:
            raise ValueError("nonclassicality_evidence.size_results must be a nonempty list")
        evidence_sizes = {int(row.get("size")) for row in size_results if isinstance(row, dict)}
        if sizes and evidence_sizes != sizes:
            raise ValueError("nonclassicality_evidence.size_results must cover exactly the scaling sizes")
        if supported:
            if evidence.get("claim_type") != "baseline_failure":
                raise ValueError("supported simulator evidence must declare baseline_failure")
            if not evidence.get("classical_failure_criterion"):
                raise ValueError("supported simulator evidence must declare classical_failure_criterion")
            if not isinstance(evidence.get("minimum_sizes"), int) or int(evidence["minimum_sizes"]) < 3:
                raise ValueError("supported simulator evidence requires minimum_sizes >= 3")
            if len(sizes) < int(evidence["minimum_sizes"]):
                raise ValueError("scaling_spec does not meet simulator evidence minimum_sizes")
        for index, row in enumerate(size_results):
            if not isinstance(row, dict):
                raise ValueError(f"evidence size row {index} must be an object")
            for field in ("sample_distance", "allowed_error", "all_required_baselines_completed", "passed_acceptance", "wall_time_seconds", "memory_mb"):
                if field not in row:
                    raise ValueError(f"evidence size row {index}.{field} is required")
            if not _is_number(row["sample_distance"]) or not _is_number(row["allowed_error"]):
                raise ValueError(f"evidence size row {index} distances must be numeric")
            if not isinstance(row["all_required_baselines_completed"], bool) or not isinstance(row["passed_acceptance"], bool):
                raise ValueError(f"evidence size row {index} flags must be boolean")
            if supported:
                site_count = int(row["size"])
                baseline_errors = baseline_errors_by_size.get(site_count, {})
                if set(baseline_errors) != set(task.baseline_families):
                    raise ValueError(f"evidence size row {index} missing baseline errors")
                best_classical_error = row.get("best_classical_error")
                if not _is_number(best_classical_error):
                    raise ValueError(f"evidence size row {index}.best_classical_error must be numeric")
                if abs(float(best_classical_error) - min(baseline_errors.values())) > 1e-12:
                    raise ValueError(f"evidence size row {index}.best_classical_error does not match baseline results")
                failures = row.get("baseline_failures")
                if not isinstance(failures, list):
                    raise ValueError(f"evidence size row {index}.baseline_failures must be a list")
                expected_failures = {
                    baseline_id
                    for baseline_id, error_value in baseline_errors.items()
                    if error_value > float(row["allowed_error"])
                }
                if set(map(str, failures)) != expected_failures:
                    raise ValueError(f"evidence size row {index}.baseline_failures must match baseline errors")
                if float(best_classical_error) <= float(row["allowed_error"]):
                    raise ValueError(f"evidence size row {index} must show every required baseline above allowed_error")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issues.append(VerificationIssue("error", "artifacts.nonclassicality_evidence", str(exc)))
    return issues


def _completed_t6_baseline_fraction(
    manifest: Mapping[str, Any],
    task: TaskSpec,
    root: Path,
) -> float:
    required = set(task.baseline_families)
    if not required:
        return 0.0
    artifacts = _artifact_mapping(manifest)
    try:
        baseline_results = _load_json_artifact(
            artifacts,
            root,
            "classical_baseline_results",
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return 0.0
    rows = baseline_results.get("rows") if isinstance(baseline_results, dict) else baseline_results
    if not isinstance(rows, list):
        return 0.0
    completed: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        baseline_id = str(row.get("baseline_id", ""))
        if baseline_id not in required:
            continue
        if row.get("error_metric") == "not_run":
            continue
        if row.get("error_value") is None and row.get("error_metric") != "reference":
            continue
        completed.add(baseline_id)
    return len(completed) / len(required)


def _nonclassicality_evidence_score(
    manifest: Mapping[str, Any],
    root: Path,
) -> float:
    artifacts = _artifact_mapping(manifest)
    if "nonclassicality_evidence" not in artifacts:
        return 0.0
    try:
        evidence = _load_json_artifact(artifacts, root, "nonclassicality_evidence")
    except (OSError, json.JSONDecodeError, ValueError):
        return 0.0
    if not isinstance(evidence, dict):
        return 0.0
    if not evidence.get("nonclassicality_claim_supported"):
        return 0.0
    rows = evidence.get("size_results")
    if not isinstance(rows, list) or not rows:
        return 0.0
    passed = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not row.get("all_required_baselines_completed"):
            continue
        if not row.get("passed_acceptance"):
            continue
        sample_distance = row.get("sample_distance")
        allowed_error = row.get("allowed_error")
        if not _is_number(sample_distance) or not _is_number(allowed_error):
            continue
        if float(sample_distance) > float(allowed_error):
            continue
        failures = row.get("baseline_failures")
        best_classical_error = row.get("best_classical_error")
        if not (
            (isinstance(failures, list) and bool(failures))
            or (_is_number(best_classical_error) and float(best_classical_error) > float(allowed_error))
        ):
            continue
        if not _is_number(row.get("wall_time_seconds")) or not _is_number(row.get("memory_mb")):
            continue
        passed += 1
    return passed / len(rows)


def _hardware_calibration_score(
    manifest: Mapping[str, Any],
    task: TaskSpec,
    root: Path,
) -> float:
    if task.task_id not in {
        "t5_finite_shot_resource_certificate",
        "t5_finite_shot_lineage_certificate",
    }:
        return 1.0
    artifacts = _artifact_mapping(manifest)
    try:
        calibration = _load_json_artifact(artifacts, root, "calibration_record")
    except (OSError, json.JSONDecodeError, ValueError):
        return 0.0
    if not isinstance(calibration, dict):
        return 0.0
    mitigation = calibration.get("readout_mitigation")
    if mitigation == "inverse_confusion_matrix":
        return 1.0
    if mitigation == "none":
        return 0.8
    return 0.0


def verify_submission(
    manifest_value: str | Path | Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> VerificationReport:
    """Verify a QALBench submission manifest for schema and declared artifacts."""

    manifest = load_manifest(manifest_value)
    if root is None and not isinstance(manifest_value, Mapping):
        root_path = Path(manifest_value).resolve().parent
    else:
        root_path = Path.cwd() if root is None else Path(root)
    issues: list[VerificationIssue] = []
    task: TaskSpec | None = None
    schema_version = manifest.get("schema_version")
    if schema_version != 1:
        issues.append(
            VerificationIssue("error", "schema_version", "schema_version must be integer 1")
        )
    task_id = manifest.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        issues.append(VerificationIssue("error", "task_id", "missing task_id"))
    else:
        try:
            task = task_by_id(task_id)
        except KeyError as exc:
            issues.append(VerificationIssue("error", "task_id", str(exc)))

    claim_statement = manifest.get("claim_statement", "")
    if not isinstance(claim_statement, str) or not claim_statement.strip():
        issues.append(VerificationIssue("error", "claim_statement", "claim statement is required"))

    artifacts = _artifact_mapping(manifest)
    declared_artifacts = set(artifacts)
    declared_controls = _string_set(manifest, "controls")
    declared_baselines = _string_set(manifest, "baselines")
    declared_axes = _string_set(manifest, "claim_axes")

    if task is not None:
        missing_artifacts = set(task.required_artifacts) - declared_artifacts
        for artifact in sorted(missing_artifacts):
            issues.append(
                VerificationIssue("error", "artifacts", f"missing required artifact {artifact!r}")
            )
        missing_controls = set(task.required_controls) - declared_controls
        for control in sorted(missing_controls):
            issues.append(
                VerificationIssue("error", "controls", f"missing required control {control!r}")
            )
        missing_axes = set(task.claim_axes) - declared_axes
        for axis in sorted(missing_axes):
            issues.append(
                VerificationIssue("warning", "claim_axes", f"task claims should declare axis {axis!r}")
            )
        unsupported_axes = declared_axes - set(task.claim_axes)
        for axis in sorted(unsupported_axes):
            issues.append(
                VerificationIssue("error", "claim_axes", f"task does not support claimed axis {axis!r}")
            )
        if task.hardware_ready and int(manifest.get("shot_budget", 0) or 0) <= 0:
            issues.append(
                VerificationIssue("error", "shot_budget", "hardware-ready tasks require positive shot_budget")
            )
        if task.scalable_challenge:
            if not manifest.get("scaling_variable"):
                issues.append(
                    VerificationIssue("error", "scaling_variable", "T6 tasks require a scaling_variable")
                )
            if not manifest.get("allowed_error"):
                issues.append(
                    VerificationIssue("error", "allowed_error", "T6 tasks require allowed_error")
                )
            if not declared_baselines:
                issues.append(
                    VerificationIssue("error", "baselines", "T6 tasks require declared simulator baselines")
                )
            missing_baselines = set(task.baseline_families) - declared_baselines
            for baseline in sorted(missing_baselines):
                issues.append(
                    VerificationIssue("error", "baselines", f"T6 task missing simulator baseline {baseline!r}")
                )
        elif task.baseline_families:
            missing_baselines = set(task.baseline_families) - declared_baselines
            for baseline in sorted(missing_baselines):
                issues.append(
                    VerificationIssue("warning", "baselines", f"missing recommended baseline {baseline!r}")
                )

    artifact_issues, artifact_checks = _check_artifacts(artifacts, root_path)
    issues.extend(artifact_issues)
    if task is not None:
        if task.task_id == "t1_basis_inheritance_kernel":
            issues.extend(_validate_t1_artifacts(artifacts, root_path))
        if task.task_id == "t1_mutation_channel_kernel":
            issues.extend(_validate_t1_mutation_artifacts(artifacts, root_path))
        if task.task_id == "t2_state_resource_diagnostic":
            issues.extend(_validate_t2_artifacts(artifacts, root_path))
        if task.task_id == "t2_process_resource_diagnostic":
            issues.extend(_validate_t2_process_artifacts(artifacts, root_path))
        if task.task_id == "t3_population_lineage_audit":
            issues.extend(_validate_t3_artifacts(artifacts, root_path))
        if task.task_id == "t3_interaction_selection_audit":
            issues.extend(_validate_t3_interaction_artifacts(artifacts, root_path))
        if task.task_id == "t4_resource_coupled_outcome":
            issues.extend(_validate_t4_artifacts(artifacts, root_path))
        if task.task_id == "t4_transmission_breaking_resource_control":
            issues.extend(_validate_t4_transmission_artifacts(artifacts, root_path))
        if task.task_id == "t5_finite_shot_resource_certificate":
            issues.extend(_validate_t5_resource_artifacts(artifacts, root_path))
        if task.task_id == "t5_finite_shot_lineage_certificate":
            issues.extend(_validate_t5_lineage_artifacts(artifacts, root_path))
        if task.scalable_challenge:
            issues.extend(_validate_t6_artifacts(artifacts, root_path))
        if task.task_id == "t6_sampling_nonclassicality_challenge":
            issues.extend(_validate_t6_sampling_artifacts(artifacts, root_path, task))
        if task.task_id == "t6_simulator_scaling_challenge":
            issues.extend(_validate_t6_simulator_artifacts(artifacts, root_path, task))
    passed = not any(issue.severity == "error" for issue in issues)
    return VerificationReport(
        passed=passed,
        task_id=task_id if isinstance(task_id, str) else None,
        issues=tuple(issues),
        artifact_checks=artifact_checks,
    )


def _fraction(present: set[str], required: tuple[str, ...]) -> float:
    if not required:
        return 1.0
    return len(set(required) & present) / len(required)


def _axis_score(
    *,
    axis: ClaimAxis,
    task: TaskSpec,
    manifest: Mapping[str, Any],
    report: VerificationReport,
    root: Path,
) -> float:
    artifacts = set(_artifact_mapping(manifest))
    controls = _string_set(manifest, "controls")
    baselines = _string_set(manifest, "baselines")
    artifact_score = _fraction(artifacts, task.required_artifacts)
    control_score = _fraction(controls, task.required_controls)
    baseline_score = _fraction(baselines, task.baseline_families)
    if axis == "artificial_life":
        raw = 0.55 * artifact_score + 0.35 * control_score + 0.10 * baseline_score
    elif axis == "quantum_resource":
        raw = 0.45 * artifact_score + 0.35 * control_score + 0.20 * baseline_score
    else:
        required = set(task.baseline_families)
        simulator_score = len(required & baselines) / len(required) if required else 0.0
        scaling_score = 1.0 if manifest.get("scaling_variable") and manifest.get("allowed_error") else 0.0
        evidence_score = _nonclassicality_evidence_score(manifest, root)
        if evidence_score <= 0.0:
            raw = 0.0
        else:
            baseline_completion = _completed_t6_baseline_fraction(manifest, task, root)
            raw = (
                0.20 * artifact_score
                + 0.20 * simulator_score
                + 0.20 * baseline_completion
                + 0.20 * scaling_score
                + 0.20 * evidence_score
            )
    if axis == "computational_nonclassicality" and not report.passed:
        return 0.0
    if not report.passed:
        error_count = sum(1 for issue in report.issues if issue.severity == "error")
        raw *= max(0.0, 1.0 - 0.15 * error_count)
    return float(max(0.0, min(1.0, raw)))


def score_submission(
    manifest_value: str | Path | Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> ScoreBreakdown:
    """Score a submission along independent audit axes."""

    manifest = load_manifest(manifest_value)
    if root is None and not isinstance(manifest_value, Mapping):
        root_path = Path(manifest_value).resolve().parent
    else:
        root_path = Path.cwd() if root is None else Path(root)
    report = verify_submission(manifest, root=root_path)
    task = task_by_id(str(manifest.get("task_id")))
    claimed_axes = _string_set(manifest, "claim_axes")
    al = _axis_score(
        axis="artificial_life",
        task=task,
        manifest=manifest,
        report=report,
        root=root_path,
    )
    qr = _axis_score(
        axis="quantum_resource",
        task=task,
        manifest=manifest,
        report=report,
        root=root_path,
    )
    cn = _axis_score(
        axis="computational_nonclassicality",
        task=task,
        manifest=manifest,
        report=report,
        root=root_path,
    )
    if "artificial_life" not in claimed_axes:
        al = 0.0
    if "quantum_resource" not in claimed_axes:
        qr = 0.0
    if "computational_nonclassicality" not in claimed_axes:
        cn = 0.0
    if "artificial_life" not in task.claim_axes:
        al = 0.0
    if "quantum_resource" not in task.claim_axes:
        qr = 0.0
    if "computational_nonclassicality" not in task.claim_axes:
        cn = 0.0

    artifact_checks = report.artifact_checks.values()
    checked = sum(1 for value in artifact_checks if value == "ok")
    existing = sum(1 for value in artifact_checks if value in {"ok", "exists_unhashed"})
    verification_completeness = checked / existing if existing else 0.0
    certification_readiness = 0.0
    if task.hardware_ready:
        calibration_score = _hardware_calibration_score(manifest, task, root_path)
        certification_readiness = verification_completeness * calibration_score if report.passed else 0.0
    elif int(manifest.get("shot_budget", 0) or 0) > 0:
        certification_readiness = 0.5 * verification_completeness

    claimed_scores = [
        score
        for axis, score in (
            ("artificial_life", al),
            ("quantum_resource", qr),
            ("computational_nonclassicality", cn),
        )
        if axis in claimed_axes
    ]
    notes = []
    if report.issues:
        notes.append(f"{len(report.issues)} verification issue(s) found")
    if task.scalable_challenge and not report.passed:
        notes.append("T6 claims remain unsupported until scaling and simulator baselines pass verification")
    if task.scalable_challenge and "computational_nonclassicality" in claimed_axes and cn == 0.0:
        notes.append("computational nonclassicality requires explicit passing evidence, not artifact presence alone")
    if task.hardware_ready and not report.passed:
        notes.append("T5 claims remain unsupported until finite-shot artifacts and controls pass verification")
    return ScoreBreakdown(
        artificial_life_adequacy=al,
        quantum_resource_relevance=qr,
        computational_nonclassicality=cn,
        certification_readiness=certification_readiness,
        verification_completeness=verification_completeness,
        claimed_axis_floor=min(claimed_scores) if claimed_scores else 0.0,
        notes=tuple(notes),
    )
