from __future__ import annotations

import hashlib
import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import qalbench  # noqa: E402
from qalbench.baselines import evaluate_resource_kernel_baselines  # noqa: E402
from qalbench.certification import (  # noqa: E402
    chsh_certificate,
    copy_agreement_certificate,
    finite_shot_lineage_certificate,
)
from qalbench.quantum import QuantumParams  # noqa: E402
from qalbench.structured_population import (  # noqa: E402
    StructuredPopulationParams,
    run_structured_population,
)
from qalbench.submission import score_submission, verify_submission  # noqa: E402
from qalbench.tasks import task_catalog, tasks_by_tier  # noqa: E402
from qalbench.workflows import (  # noqa: E402
    FiniteShotSubmissionConfig,
    PopulationSubmissionConfig,
    ResourceKernelSubmissionConfig,
    SamplingChallengeConfig,
    StructuredPopulationSweepConfig,
    write_finite_shot_submission,
    write_population_submission,
    write_resource_kernel_submission,
    write_sampling_challenge_submission,
    write_structured_population_submission,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class SuiteContractTests(unittest.TestCase):
    def test_task_catalog_covers_all_tiers(self) -> None:
        grouped = tasks_by_tier()
        self.assertEqual(set(grouped), {"T1", "T2", "T3", "T4", "T5", "T6"})
        self.assertTrue(all(grouped[tier] for tier in grouped))
        expected_ids = {
            "t1_basis_inheritance_kernel",
            "t1_mutation_channel_kernel",
            "t2_state_resource_diagnostic",
            "t2_process_resource_diagnostic",
            "t3_population_lineage_audit",
            "t3_interaction_selection_audit",
            "t4_resource_coupled_outcome",
            "t4_transmission_breaking_resource_control",
            "t5_finite_shot_resource_certificate",
            "t5_finite_shot_lineage_certificate",
            "t6_sampling_nonclassicality_challenge",
            "t6_simulator_scaling_challenge",
        }
        task_ids = {task.task_id for task in task_catalog()}
        self.assertEqual(task_ids, expected_ids)
        self.assertEqual(len(task_ids), len(task_catalog()))
        valid_axes = {"artificial_life", "quantum_resource", "computational_nonclassicality"}
        for task in task_catalog():
            self.assertTrue(task.required_artifacts, task.task_id)
            self.assertTrue(task.required_controls, task.task_id)
            self.assertTrue(task.metrics, task.task_id)
            self.assertTrue(set(task.claim_axes) <= valid_axes, task.task_id)
            docs_text = (ROOT / "docs" / "submission_schema.md").read_text()
            self.assertIn(task.task_id, docs_text)

    def test_public_package_api_exports_suite_surface(self) -> None:
        for name in (
            "task_catalog",
            "verify_submission",
            "score_submission",
            "write_finite_shot_submission",
            "write_sampling_challenge_submission",
            "StructuredPopulationParams",
            "SamplingChallengeConfig",
        ):
            self.assertTrue(hasattr(qalbench, name), name)
        self.assertGreaterEqual(len(qalbench.task_catalog()), 12)

    def test_finite_shot_copy_and_chsh_certificates(self) -> None:
        copy = copy_agreement_certificate({"00": 480, "01": 20, "10": 10, "11": 490})
        self.assertGreater(copy.estimate, 0.95)
        self.assertGreater(copy.lower, 0.93)

        correlated = {"00": 4500, "11": 4500, "01": 500, "10": 500}
        anticorrelated = {"01": 4500, "10": 4500, "00": 500, "11": 500}
        certificate = chsh_certificate(
            {
                "ab": correlated,
                "ab_prime": correlated,
                "a_prime_b": correlated,
                "a_prime_b_prime": anticorrelated,
            }
        )
        self.assertGreater(certificate.estimate, 3.0)
        self.assertGreater(certificate.lower, 2.0)
        self.assertTrue(certificate.certified)

    def test_finite_shot_lineage_certificate_compares_controls(self) -> None:
        certificate = finite_shot_lineage_certificate(
            {
                "observed": {"00": 480, "01": 20, "10": 10, "11": 490},
                "no_inheritance": {"00": 250, "01": 250, "10": 250, "11": 250},
                "shuffled_lineage": {"00": 260, "01": 240, "10": 240, "11": 260},
            }
        )
        self.assertTrue(certificate["certified_inheritance"])
        observed = certificate["groups"]["observed"]
        self.assertGreater(observed["copy_agreement_interval"]["lower"], 0.93)
        self.assertLess(observed["mutation_rate_interval"]["upper"], 0.06)

    def test_resource_kernel_baselines_include_separable_controls(self) -> None:
        baselines = evaluate_resource_kernel_baselines(QuantumParams(theta=math.pi / 2.0))
        self.assertEqual(
            set(baselines),
            {
                "quantum",
                "dephased",
                "final_dephased",
                "classical_markov",
                "separable_product",
                "maximally_mixed",
            },
        )
        self.assertGreater(baselines["quantum"]["metrics"]["negativity"], 0.0)
        self.assertEqual(baselines["separable_product"]["metrics"]["negativity"], 0.0)
        self.assertEqual(baselines["classical_markov"]["metrics"]["negativity"], 0.0)

    def test_structured_population_keeps_joint_quantum_state(self) -> None:
        coherent = run_structured_population(
            StructuredPopulationParams(site_count=3, steps=2, initial_theta=math.pi / 2.0)
        )
        dephased = run_structured_population(
            StructuredPopulationParams(
                site_count=3,
                steps=2,
                initial_theta=math.pi / 2.0,
                dephase_after_step=True,
            )
        )
        self.assertEqual(coherent["state_dimension"], 8)
        self.assertEqual(len(coherent["trajectory"]), 3)
        self.assertAlmostEqual(coherent["summary"]["trace"], 1.0)
        self.assertGreater(
            max(row["max_pair_negativity"] for row in coherent["trajectory"]),
            0.0,
        )
        self.assertLessEqual(
            max(row["max_pair_negativity"] for row in dephased["trajectory"]),
            1e-12,
        )

    def test_submission_verification_and_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            count_payload = {"counts": {"00": 480, "01": 20, "10": 10, "11": 490}}
            copy_interval = copy_agreement_certificate(count_payload["counts"]).as_dict()
            files = {
                "witness_estimates": json.dumps(
                    {
                        "kind": "copy-agreement",
                        "confidence": 0.95,
                        "certificate": {"copy_agreement": copy_interval},
                    }
                ),
                "calibration_record": json.dumps(
                    {
                        "readout_mitigation": "none",
                        "readout_error_model": {
                            "type": "ideal_readout_assumption",
                            "mitigation_applied": False,
                            "calibration_shots": 0,
                        },
                        "assumptions": ["counts are accepted as submitted"],
                    }
                ),
            }
            artifacts = {}
            for name, content in files.items():
                path = root / f"{name}.txt"
                path.write_text(content)
                artifacts[name] = {"path": path.name, "sha256": sha256(path)}

            (root / "shot_counts.json").write_text(json.dumps(count_payload))
            artifacts["shot_counts"] = {
                "path": "shot_counts.json",
                "sha256": sha256(root / "shot_counts.json"),
            }
            (root / "confidence_intervals.json").write_text(
                json.dumps({"copy_agreement": copy_interval})
            )
            artifacts["confidence_intervals"] = {
                "path": "confidence_intervals.json",
                "sha256": sha256(root / "confidence_intervals.json"),
            }

            manifest = {
                "schema_version": 1,
                "task_id": "t5_finite_shot_resource_certificate",
                "claim_statement": "Finite-shot CHSH resource certificate for a two-qubit event.",
                "claim_axes": ["quantum_resource"],
                "artifacts": artifacts,
                "controls": [
                    "finite_shot_dephased_control",
                    "finite_shot_classical_control",
                ],
                "baselines": ["dephased_density", "separable_product", "readout_error"],
                "shot_budget": 10000,
            }
            report = verify_submission(manifest, root=root)
            self.assertTrue(report.passed, report.as_dict())
            score = score_submission(manifest, root=root)
            self.assertGreaterEqual(score.quantum_resource_relevance, 0.99)
            self.assertEqual(score.artificial_life_adequacy, 0.0)
            self.assertEqual(score.computational_nonclassicality, 0.0)

    def test_submission_rejects_bad_schema_version_and_overclaimed_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = write_resource_kernel_submission(
                root / "t1",
                ResourceKernelSubmissionConfig(task_id="t1_basis_inheritance_kernel"),
            )
            manifest_path = Path(result["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            manifest.pop("schema_version")
            manifest_path.write_text(json.dumps(manifest))
            report = verify_submission(manifest_path)
            self.assertFalse(report.passed)
            self.assertTrue(any(issue.field == "schema_version" for issue in report.issues))

            manifest["schema_version"] = 2
            manifest_path.write_text(json.dumps(manifest))
            report = verify_submission(manifest_path)
            self.assertFalse(report.passed)
            self.assertTrue(any(issue.field == "schema_version" for issue in report.issues))

            manifest["schema_version"] = 1
            manifest["claim_axes"] = ["artificial_life", "quantum_resource"]
            manifest_path.write_text(json.dumps(manifest))
            report = verify_submission(manifest_path)
            self.assertFalse(report.passed)
            self.assertTrue(any(issue.field == "claim_axes" for issue in report.issues))
            score = score_submission(manifest_path)
            self.assertEqual(score.quantum_resource_relevance, 0.0)

    def test_structured_population_workflow_writes_verifiable_t6_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = write_structured_population_submission(
                root,
                StructuredPopulationSweepConfig(site_counts=(2, 3), steps=2),
            )
            manifest_path = result["manifest_path"]
            self.assertTrue((root / "structured_population_timeseries.csv").exists())
            report = verify_submission(manifest_path)
            self.assertTrue(report.passed, report.as_dict())
            score = score_submission(manifest_path)
            self.assertGreater(score.quantum_resource_relevance, 0.0)
            self.assertEqual(score.computational_nonclassicality, 0.0)
            self.assertIn("verification issue", " ".join(score.notes))
            baseline_rows = json.loads((root / "classical_baseline_results.json").read_text())["rows"]
            self.assertFalse(any(row["error_metric"] == "not_run" for row in baseline_rows))
            self.assertTrue(any(row["baseline_id"] == "mean_field" for row in baseline_rows))
            self.assertTrue(any(row["baseline_id"] == "tensor_network" for row in baseline_rows))

    def test_structured_population_verifier_links_metrics_to_timeseries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = write_structured_population_submission(
                root,
                StructuredPopulationSweepConfig(site_counts=(2, 3), steps=2),
            )
            timeseries_path = root / "structured_population_timeseries.csv"
            with timeseries_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
                fieldnames = list(rows[0])
            for row in rows:
                if row["model"] == "coherent" and row["site_count"] == "2" and row["step"] == "2":
                    row["global_coherence_l1"] = "999.0"
                    break
            with timeseries_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            manifest_path = Path(result["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["population_timeseries"]["sha256"] = sha256(timeseries_path)
            manifest_path.write_text(json.dumps(manifest))
            report = verify_submission(manifest_path)
            self.assertFalse(report.passed)
            self.assertTrue(
                any(issue.field == "artifacts.population_timeseries" for issue in report.issues)
            )

    def test_sampling_challenge_workflow_writes_verifiable_t6_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = write_sampling_challenge_submission(
                root,
                SamplingChallengeConfig(sizes=(2, 3), shots_per_size=64),
            )
            report = verify_submission(result["manifest_path"])
            self.assertTrue(report.passed, report.as_dict())
            score = score_submission(result["manifest_path"])
            self.assertGreater(score.artificial_life_adequacy, 0.99)
            self.assertGreater(score.quantum_resource_relevance, 0.99)
            self.assertEqual(score.computational_nonclassicality, 0.0)
            self.assertIn("artifact presence alone", " ".join(score.notes))
            evidence = json.loads((root / "nonclassicality_evidence.json").read_text())
            self.assertFalse(evidence["nonclassicality_claim_supported"])

    def test_sampling_challenge_verifier_rejects_incomplete_baseline_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = write_sampling_challenge_submission(
                root,
                SamplingChallengeConfig(sizes=(2, 3), shots_per_size=32),
            )
            baseline_path = root / "classical_baseline_results.json"
            baselines = json.loads(baseline_path.read_text())
            baselines["rows"] = baselines["rows"][:-1]
            baseline_path.write_text(json.dumps(baselines))
            manifest_path = Path(result["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["classical_baseline_results"]["sha256"] = sha256(baseline_path)
            manifest_path.write_text(json.dumps(manifest))
            report = verify_submission(manifest_path)
            self.assertFalse(report.passed)
            self.assertTrue(
                any(issue.field == "artifacts.classical_baseline_results" for issue in report.issues)
            )

            unsupported_claim = write_sampling_challenge_submission(
                root / "unsupported-claim",
                SamplingChallengeConfig(sizes=(2, 3, 4), shots_per_size=32),
            )
            evidence_path = root / "unsupported-claim" / "nonclassicality_evidence.json"
            evidence = json.loads(evidence_path.read_text())
            evidence["nonclassicality_claim_supported"] = True
            evidence["claim_type"] = "baseline_failure"
            evidence["classical_failure_criterion"] = "best_classical_error > allowed_error"
            evidence["minimum_sizes"] = 3
            evidence_path.write_text(json.dumps(evidence))
            manifest_path = Path(unsupported_claim["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["nonclassicality_evidence"]["sha256"] = sha256(evidence_path)
            manifest_path.write_text(json.dumps(manifest))
            report = verify_submission(manifest_path)
            self.assertFalse(report.passed)
            self.assertTrue(
                any(issue.field == "artifacts.nonclassicality_evidence" for issue in report.issues)
            )

    def test_sampling_challenge_scores_supported_nonclassicality_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = write_sampling_challenge_submission(
                root,
                SamplingChallengeConfig(sizes=(2, 3, 4), shots_per_size=32),
            )
            manifest_path = Path(result["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            baseline_ids = manifest["baselines"]

            baseline_path = root / "classical_baseline_results.json"
            baseline_payload = json.loads(baseline_path.read_text())
            for row in baseline_payload["rows"]:
                row["error_value"] = 0.1
            baseline_path.write_text(json.dumps(baseline_payload))
            manifest["artifacts"]["classical_baseline_results"]["sha256"] = sha256(baseline_path)

            evidence_path = root / "nonclassicality_evidence.json"
            evidence = json.loads(evidence_path.read_text())
            evidence["nonclassicality_claim_supported"] = True
            evidence["claim_type"] = "baseline_failure"
            evidence["classical_failure_criterion"] = "best_classical_error > allowed_error"
            evidence["minimum_sizes"] = 3
            for row in evidence["size_results"]:
                row["best_classical_error"] = 0.1
                row["baseline_failures"] = list(baseline_ids)
            evidence_path.write_text(json.dumps(evidence))
            manifest["artifacts"]["nonclassicality_evidence"]["sha256"] = sha256(evidence_path)
            manifest_path.write_text(json.dumps(manifest))

            report = verify_submission(manifest_path)
            self.assertTrue(report.passed, report.as_dict())
            score = score_submission(manifest_path)
            self.assertGreater(score.computational_nonclassicality, 0.99)

    def test_finite_shot_workflow_writes_verifiable_t5_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counts = {
                "setting_counts": {
                    "ab": {"00": 45, "01": 5, "10": 5, "11": 45},
                    "ab_prime": {"00": 45, "01": 5, "10": 5, "11": 45},
                    "a_prime_b": {"00": 45, "01": 5, "10": 5, "11": 45},
                    "a_prime_b_prime": {"00": 5, "01": 45, "10": 45, "11": 5},
                }
            }
            result = write_finite_shot_submission(
                root,
                counts,
                FiniteShotSubmissionConfig(kind="chsh", confidence=0.95),
            )
            report = verify_submission(result["manifest_path"])
            self.assertTrue(report.passed, report.as_dict())
            score = score_submission(result["manifest_path"])
            self.assertEqual(score.certification_readiness, 0.8)
            self.assertGreater(score.quantum_resource_relevance, 0.99)
            self.assertEqual(score.computational_nonclassicality, 0.0)

    def test_finite_shot_lineage_workflow_writes_verifiable_t5_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lineage_counts = {
                "lineage_counts": {
                    "observed": {"00": 480, "01": 20, "10": 10, "11": 490},
                    "no_inheritance": {"00": 250, "01": 250, "10": 250, "11": 250},
                    "shuffled_lineage": {"00": 260, "01": 240, "10": 240, "11": 260},
                }
            }
            result = write_finite_shot_submission(
                root,
                lineage_counts,
                FiniteShotSubmissionConfig(
                    kind="lineage",
                    confidence=0.95,
                    task_id="t5_finite_shot_lineage_certificate",
                ),
            )
            report = verify_submission(result["manifest_path"])
            self.assertTrue(report.passed, report.as_dict())
            score = score_submission(result["manifest_path"])
            self.assertEqual(score.certification_readiness, 0.8)
            self.assertGreater(score.artificial_life_adequacy, 0.99)
            self.assertEqual(score.quantum_resource_relevance, 0.0)
            intervals = json.loads((root / "confidence_intervals.json").read_text())
            self.assertTrue(intervals["lineage"]["certified_inheritance"])

    def test_resource_kernel_workflows_write_t1_and_t2_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            t1 = write_resource_kernel_submission(
                root / "t1",
                ResourceKernelSubmissionConfig(task_id="t1_basis_inheritance_kernel"),
            )
            t1_report = verify_submission(t1["manifest_path"])
            self.assertTrue(t1_report.passed, t1_report.as_dict())
            t1_score = score_submission(t1["manifest_path"])
            self.assertGreater(t1_score.artificial_life_adequacy, 0.99)
            self.assertEqual(t1_score.quantum_resource_relevance, 0.0)

            t2 = write_resource_kernel_submission(
                root / "t2",
                ResourceKernelSubmissionConfig(task_id="t2_state_resource_diagnostic"),
            )
            t2_report = verify_submission(t2["manifest_path"])
            self.assertTrue(t2_report.passed, t2_report.as_dict())
            metrics = json.loads((root / "t2" / "resource_metrics.json").read_text())
            self.assertGreater(metrics["quantum"]["metrics"]["negativity"], 0.0)
            t2_score = score_submission(t2["manifest_path"])
            self.assertGreater(t2_score.quantum_resource_relevance, 0.99)
            self.assertEqual(t2_score.artificial_life_adequacy, 0.0)

    def test_alternate_resource_kernel_families_are_runnable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mutation = write_resource_kernel_submission(
                root / "mutation",
                ResourceKernelSubmissionConfig(task_id="t1_mutation_channel_kernel"),
            )
            mutation_report = verify_submission(mutation["manifest_path"])
            self.assertTrue(mutation_report.passed, mutation_report.as_dict())
            mutation_rows = json.loads((root / "mutation" / "mutation_records.json").read_text())["rows"]
            self.assertGreater(mutation_rows[0]["declared_flip_probability"], 0.0)
            self.assertGreater(score_submission(mutation["manifest_path"]).artificial_life_adequacy, 0.99)

            process = write_resource_kernel_submission(
                root / "process",
                ResourceKernelSubmissionConfig(task_id="t2_process_resource_diagnostic"),
            )
            process_report = verify_submission(process["manifest_path"])
            self.assertTrue(process_report.passed, process_report.as_dict())
            process_metrics = json.loads((root / "process" / "resource_metrics.json").read_text())
            self.assertIn("mean_resource_survival_gap", process_metrics["summary"])
            self.assertGreater(score_submission(process["manifest_path"]).quantum_resource_relevance, 0.99)

    def test_population_workflows_write_t3_and_t4_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            t3 = write_population_submission(
                root / "t3",
                PopulationSubmissionConfig(
                    task_id="t3_population_lineage_audit",
                    steps=6,
                    initial_population=8,
                    carrying_capacity=18,
                    lineage_null_permutations=5,
                ),
            )
            t3_report = verify_submission(t3["manifest_path"])
            self.assertTrue(t3_report.passed, t3_report.as_dict())
            t3_score = score_submission(t3["manifest_path"])
            self.assertGreater(t3_score.artificial_life_adequacy, 0.99)
            self.assertEqual(t3_score.quantum_resource_relevance, 0.0)

            t4 = write_population_submission(
                root / "t4",
                PopulationSubmissionConfig(
                    task_id="t4_resource_coupled_outcome",
                    steps=6,
                    initial_population=8,
                    carrying_capacity=18,
                    lineage_null_permutations=5,
                ),
            )
            t4_report = verify_submission(t4["manifest_path"])
            self.assertTrue(t4_report.passed, t4_report.as_dict())
            effect = json.loads((root / "t4" / "effect_size_summary.json").read_text())
            self.assertIn("resource_score_quantum_minus_dephased", effect)
            t4_score = score_submission(t4["manifest_path"])
            self.assertGreater(t4_score.artificial_life_adequacy, 0.99)
            self.assertGreater(t4_score.quantum_resource_relevance, 0.99)

    def test_alternate_population_families_are_runnable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            interaction = write_population_submission(
                root / "interaction",
                PopulationSubmissionConfig(
                    task_id="t3_interaction_selection_audit",
                    steps=6,
                    initial_population=8,
                    carrying_capacity=18,
                    lineage_null_permutations=5,
                ),
            )
            interaction_report = verify_submission(interaction["manifest_path"])
            self.assertTrue(interaction_report.passed, interaction_report.as_dict())
            selection = json.loads((root / "interaction" / "selection_rule.json").read_text())
            self.assertIn("interaction_birth_covariance", selection)
            self.assertGreater(score_submission(interaction["manifest_path"]).artificial_life_adequacy, 0.99)

            transmission = write_population_submission(
                root / "transmission",
                PopulationSubmissionConfig(
                    task_id="t4_transmission_breaking_resource_control",
                    steps=6,
                    initial_population=8,
                    carrying_capacity=18,
                    lineage_null_permutations=5,
                ),
            )
            transmission_report = verify_submission(transmission["manifest_path"])
            self.assertTrue(transmission_report.passed, transmission_report.as_dict())
            lineage = json.loads((root / "transmission" / "lineage_nulls.json").read_text())
            self.assertIn("theta_correlation_control_gap", lineage)
            transmission_score = score_submission(transmission["manifest_path"])
            self.assertGreater(transmission_score.artificial_life_adequacy, 0.99)
            self.assertGreater(transmission_score.quantum_resource_relevance, 0.99)

    def test_submission_verifier_rejects_malformed_tier_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            t2 = write_resource_kernel_submission(
                root / "t2",
                ResourceKernelSubmissionConfig(task_id="t2_state_resource_diagnostic"),
            )
            metrics_path = root / "t2" / "resource_metrics.json"
            metrics_path.write_text(json.dumps({"quantum": {"metrics": {}}}))
            manifest_path = Path(t2["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["resource_metrics"]["sha256"] = sha256(metrics_path)
            manifest_path.write_text(json.dumps(manifest))
            report = verify_submission(manifest_path)
            self.assertFalse(report.passed)
            self.assertTrue(
                any(issue.field == "artifacts.resource_metrics" for issue in report.issues)
            )

            t3 = write_population_submission(
                root / "t3",
                PopulationSubmissionConfig(
                    task_id="t3_population_lineage_audit",
                    steps=6,
                    initial_population=8,
                    carrying_capacity=18,
                    lineage_null_permutations=5,
                ),
            )
            individuals_path = root / "t3" / "individual_records.json"
            individuals_path.write_text(json.dumps({"rows": []}))
            manifest_path = Path(t3["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["individual_records"]["sha256"] = sha256(individuals_path)
            manifest_path.write_text(json.dumps(manifest))
            report = verify_submission(manifest_path)
            self.assertFalse(report.passed)
            self.assertTrue(
                any(issue.field == "artifacts.individual_records" for issue in report.issues)
            )

            t3_late = write_population_submission(
                root / "t3-late",
                PopulationSubmissionConfig(
                    task_id="t3_population_lineage_audit",
                    steps=6,
                    initial_population=8,
                    carrying_capacity=18,
                    lineage_null_permutations=5,
                ),
            )
            individuals_path = root / "t3-late" / "individual_records.json"
            individuals = json.loads(individuals_path.read_text())
            individuals["rows"].append({"id": 999999})
            individuals_path.write_text(json.dumps(individuals))
            manifest_path = Path(t3_late["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["individual_records"]["sha256"] = sha256(individuals_path)
            manifest_path.write_text(json.dumps(manifest))
            report = verify_submission(manifest_path)
            self.assertFalse(report.passed)
            self.assertTrue(
                any(issue.field == "artifacts.individual_records" for issue in report.issues)
            )

            finite = write_finite_shot_submission(
                root / "finite",
                {"counts": {"00": 48, "01": 2, "10": 1, "11": 49}},
                FiniteShotSubmissionConfig(kind="copy-agreement"),
            )
            intervals_path = root / "finite" / "confidence_intervals.json"
            intervals = json.loads(intervals_path.read_text())
            intervals["copy_agreement"]["lower"] = 0.0
            intervals_path.write_text(json.dumps(intervals))
            manifest_path = Path(finite["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["confidence_intervals"]["sha256"] = sha256(intervals_path)
            manifest_path.write_text(json.dumps(manifest))
            report = verify_submission(manifest_path)
            self.assertFalse(report.passed)
            self.assertTrue(
                any(issue.field == "artifacts.confidence_intervals" for issue in report.issues)
            )

            finite_bad_calibration = write_finite_shot_submission(
                root / "finite-bad-calibration",
                {"counts": {"00": 48, "01": 2, "10": 1, "11": 49}},
                FiniteShotSubmissionConfig(kind="copy-agreement"),
            )
            calibration_path = root / "finite-bad-calibration" / "calibration_record.json"
            calibration = json.loads(calibration_path.read_text())
            calibration["readout_mitigation"] = "inverse_confusion_matrix"
            calibration["readout_error_model"] = {
                "outcomes": ["00", "01"],
                "confusion_matrix": [[0.9, 0.2], [0.1, 0.8]],
                "inverse_confusion_matrix": [[1.0, 0.0], [0.0, 1.0]],
                "calibration_shots": 100,
            }
            calibration_path.write_text(json.dumps(calibration))
            manifest_path = Path(finite_bad_calibration["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["calibration_record"]["sha256"] = sha256(calibration_path)
            manifest_path.write_text(json.dumps(manifest))
            report = verify_submission(manifest_path)
            self.assertFalse(report.passed)
            self.assertTrue(
                any(issue.field == "artifacts.calibration_record" for issue in report.issues)
            )

            finite_lineage = write_finite_shot_submission(
                root / "finite-lineage-stale",
                {
                    "lineage_counts": {
                        "observed": {"00": 48, "01": 2, "10": 1, "11": 49},
                        "no_inheritance": {"00": 25, "01": 25, "10": 25, "11": 25},
                        "shuffled_lineage": {"00": 26, "01": 24, "10": 24, "11": 26},
                    }
                },
                FiniteShotSubmissionConfig(
                    kind="lineage",
                    task_id="t5_finite_shot_lineage_certificate",
                ),
            )
            intervals_path = root / "finite-lineage-stale" / "confidence_intervals.json"
            intervals = json.loads(intervals_path.read_text())
            intervals["lineage"]["groups"]["observed"]["copy_agreement_interval"]["lower"] = 0.0
            intervals_path.write_text(json.dumps(intervals))
            manifest_path = Path(finite_lineage["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"]["confidence_intervals"]["sha256"] = sha256(intervals_path)
            manifest_path.write_text(json.dumps(manifest))
            report = verify_submission(manifest_path)
            self.assertFalse(report.passed)
            self.assertTrue(
                any(issue.field == "artifacts.confidence_intervals" for issue in report.issues)
            )

    def test_suite_cli_catalog_is_json(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "qalbench.suite",
                "catalog",
                "--include-baselines",
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertGreaterEqual(len(payload["tasks"]), 10)
        self.assertGreaterEqual(len(payload["baselines"]), 8)

    def test_suite_cli_template_certify_and_structured_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template_path = root / "template.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qalbench.suite",
                    "template",
                    "t5_finite_shot_resource_certificate",
                    "--output",
                    str(template_path),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            template = json.loads(template_path.read_text())
            self.assertEqual(template["task_id"], "t5_finite_shot_resource_certificate")

            counts_path = root / "counts.json"
            counts_path.write_text(json.dumps({"counts": {"00": 9, "01": 1, "10": 0, "11": 10}}))
            certified = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qalbench.suite",
                    "certify-counts",
                    str(counts_path),
                    "--kind",
                    "copy-agreement",
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            certificate = json.loads(certified.stdout)
            self.assertIn("copy_agreement", certificate)

            lineage_counts_path = root / "lineage_counts.json"
            lineage_counts_path.write_text(
                json.dumps(
                    {
                        "lineage_counts": {
                            "observed": {"00": 48, "01": 2, "10": 1, "11": 49},
                            "no_inheritance": {"00": 25, "01": 25, "10": 25, "11": 25},
                            "shuffled_lineage": {"00": 26, "01": 24, "10": 24, "11": 26},
                        }
                    }
                )
            )
            lineage_certified = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qalbench.suite",
                    "certify-counts",
                    str(lineage_counts_path),
                    "--kind",
                    "lineage",
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            lineage_certificate = json.loads(lineage_certified.stdout)
            self.assertIn("lineage", lineage_certificate)

            package_dir = root / "structured"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qalbench.suite",
                    "run-structured-population",
                    "--output-dir",
                    str(package_dir),
                    "--site-counts",
                    "2,3",
                    "--steps",
                    "2",
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            self.assertTrue((package_dir / "submission_manifest.json").exists())

            sampling_dir = root / "sampling"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qalbench.suite",
                    "write-sampling-challenge-submission",
                    "--output-dir",
                    str(sampling_dir),
                    "--sizes",
                    "2,3",
                    "--shots-per-size",
                    "32",
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            self.assertTrue((sampling_dir / "submission_manifest.json").exists())

            finite_dir = root / "finite"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qalbench.suite",
                    "write-finite-shot-submission",
                    str(counts_path),
                    "--kind",
                    "copy-agreement",
                    "--output-dir",
                    str(finite_dir),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            self.assertTrue((finite_dir / "submission_manifest.json").exists())

            lineage_finite_dir = root / "finite-lineage"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qalbench.suite",
                    "write-finite-shot-submission",
                    str(lineage_counts_path),
                    "--kind",
                    "lineage",
                    "--task-id",
                    "t5_finite_shot_lineage_certificate",
                    "--output-dir",
                    str(lineage_finite_dir),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            self.assertTrue((lineage_finite_dir / "submission_manifest.json").exists())

            resource_dir = root / "resource"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qalbench.suite",
                    "write-resource-kernel-submission",
                    "--output-dir",
                    str(resource_dir),
                    "--task-id",
                    "t1_basis_inheritance_kernel",
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            self.assertTrue((resource_dir / "submission_manifest.json").exists())

            population_dir = root / "population"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qalbench.suite",
                    "write-population-submission",
                    "--output-dir",
                    str(population_dir),
                    "--task-id",
                    "t3_population_lineage_audit",
                    "--steps",
                    "4",
                    "--initial-population",
                    "6",
                    "--carrying-capacity",
                    "14",
                    "--lineage-null-permutations",
                    "3",
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            self.assertTrue((population_dir / "submission_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
