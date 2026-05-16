#!/usr/bin/env python3
"""QALBench suite catalog, submission verification, and scoring CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .baselines import baseline_catalog
from .certification import (
    chsh_certificate,
    copy_agreement_certificate,
    finite_shot_lineage_certificate,
    histogram_probabilities,
)
from .submission import score_submission, verify_submission
from .tasks import task_catalog_dicts
from .workflows import (
    FiniteShotSubmissionConfig,
    PopulationSubmissionConfig,
    ResourceKernelSubmissionConfig,
    SamplingChallengeConfig,
    StructuredPopulationSweepConfig,
    write_json,
    write_finite_shot_submission,
    write_population_submission,
    write_resource_kernel_submission,
    write_sampling_challenge_submission,
    write_structured_population_submission,
    write_submission_template,
)


def _parse_site_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not counts:
        raise argparse.ArgumentTypeError("site-counts must contain at least one integer")
    return counts


def _print_or_write(payload: object, output: str | None) -> None:
    if output:
        write_json(Path(output), payload)
        print(f"wrote {output}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog", help="print registered task catalog")
    catalog.add_argument("--include-baselines", action="store_true")

    template = subparsers.add_parser("template", help="write a submission manifest template")
    template.add_argument("task_id")
    template.add_argument("--output", required=True)
    template.add_argument("--artifact-root", default="artifacts")

    verify = subparsers.add_parser("verify-submission", help="verify a submission manifest")
    verify.add_argument("manifest")
    verify.add_argument("--root")

    score = subparsers.add_parser("score-submission", help="score a submission manifest")
    score.add_argument("manifest")
    score.add_argument("--root")

    certify = subparsers.add_parser("certify-counts", help="make finite-shot certificates from JSON counts")
    certify.add_argument("counts_json")
    certify.add_argument(
        "--kind",
        choices=("copy-agreement", "histogram", "chsh", "lineage"),
        default="copy-agreement",
    )
    certify.add_argument("--confidence", type=float, default=0.95)
    certify.add_argument("--output")

    finite_shot = subparsers.add_parser(
        "write-finite-shot-submission",
        help="write a complete T5 finite-shot submission package from JSON counts",
    )
    finite_shot.add_argument("counts_json")
    finite_shot.add_argument("--output-dir", required=True)
    finite_shot.add_argument(
        "--kind",
        choices=("copy-agreement", "histogram", "chsh", "lineage"),
        default="copy-agreement",
    )
    finite_shot.add_argument(
        "--task-id",
        choices=(
            "t5_finite_shot_resource_certificate",
            "t5_finite_shot_lineage_certificate",
        ),
        default="t5_finite_shot_resource_certificate",
    )
    finite_shot.add_argument("--confidence", type=float, default=0.95)

    resource = subparsers.add_parser(
        "write-resource-kernel-submission",
        help="write a compact T1/T2 resource-kernel submission package",
    )
    resource.add_argument("--output-dir", required=True)
    resource.add_argument(
        "--task-id",
        choices=(
            "t1_basis_inheritance_kernel",
            "t1_mutation_channel_kernel",
            "t2_state_resource_diagnostic",
            "t2_process_resource_diagnostic",
        ),
        default="t2_state_resource_diagnostic",
    )
    resource.add_argument("--theta", type=float, default=1.5707963267948966)
    resource.add_argument("--phi", type=float, default=0.0)
    resource.add_argument("--local-perturbation-probability", type=float, default=0.0)
    resource.add_argument("--local-perturbation-angle", type=float, default=1.0471975511965976)
    resource.add_argument("--interaction-angle", type=float, default=0.0)
    resource.add_argument("--damping-probability", type=float, default=0.0)
    resource.add_argument("--dephase-probability", type=float, default=0.0)

    population = subparsers.add_parser(
        "write-population-submission",
        help="write a compact T3/T4 population submission package",
    )
    population.add_argument("--output-dir", required=True)
    population.add_argument(
        "--task-id",
        choices=(
            "t3_population_lineage_audit",
            "t3_interaction_selection_audit",
            "t4_resource_coupled_outcome",
            "t4_transmission_breaking_resource_control",
        ),
        default="t3_population_lineage_audit",
    )
    population.add_argument("--seed", type=int, default=101)
    population.add_argument("--steps", type=int, default=12)
    population.add_argument("--initial-population", type=int, default=10)
    population.add_argument("--carrying-capacity", type=int, default=28)
    population.add_argument("--lineage-null-permutations", type=int, default=25)

    structured = subparsers.add_parser(
        "run-structured-population",
        help="write a small exact structured-population T6 reference package",
    )
    structured.add_argument("--output-dir", required=True)
    structured.add_argument("--site-counts", type=_parse_site_counts, default=(2, 3, 4))
    structured.add_argument("--steps", type=int, default=3)
    structured.add_argument("--initial-theta", type=float, default=1.5707963267948966)
    structured.add_argument("--mutation-probability", type=float, default=0.0)
    structured.add_argument("--mutation-angle", type=float, default=0.39269908169872414)
    structured.add_argument("--interaction-angle", type=float, default=0.0)
    structured.add_argument("--damping-probability", type=float, default=0.0)

    sampling = subparsers.add_parser(
        "write-sampling-challenge-submission",
        help="write a compact T6 sampling-challenge reference package",
    )
    sampling.add_argument("--output-dir", required=True)
    sampling.add_argument("--sizes", type=_parse_site_counts, default=(2, 3, 4))
    sampling.add_argument("--shots-per-size", type=int, default=256)
    sampling.add_argument("--allowed-error", type=float, default=0.05)
    sampling.add_argument("--seed", type=int, default=271)
    sampling.add_argument("--resource-depth", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "catalog":
        payload: dict[str, object] = {"tasks": task_catalog_dicts()}
        if args.include_baselines:
            payload["baselines"] = [baseline.as_dict() for baseline in baseline_catalog()]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "template":
        path = write_submission_template(
            args.task_id,
            Path(args.output),
            artifact_root=args.artifact_root,
        )
        print(f"wrote {path}")
        return

    if args.command == "verify-submission":
        report = verify_submission(args.manifest, root=args.root)
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        if not report.passed:
            raise SystemExit(1)
        return

    if args.command == "score-submission":
        score = score_submission(args.manifest, root=args.root)
        print(json.dumps(score.as_dict(), indent=2, sort_keys=True))
        return

    if args.command == "certify-counts":
        with Path(args.counts_json).open() as handle:
            payload = json.load(handle)
        if args.kind == "copy-agreement":
            counts = payload.get("counts", payload) if isinstance(payload, dict) else payload
            certificate = copy_agreement_certificate(counts, confidence=args.confidence)
            _print_or_write({"copy_agreement": certificate.as_dict()}, args.output)
            return
        if args.kind == "histogram":
            counts = payload.get("counts", payload) if isinstance(payload, dict) else payload
            intervals = histogram_probabilities(counts, confidence=args.confidence)
            _print_or_write(
                {
                    "histogram_probabilities": {
                        outcome: interval.as_dict()
                        for outcome, interval in intervals.items()
                    }
                },
                args.output,
            )
            return
        if args.kind == "chsh":
            setting_counts = (
                payload.get("setting_counts", payload)
                if isinstance(payload, dict)
                else payload
            )
            certificate = chsh_certificate(setting_counts, confidence=args.confidence)
            _print_or_write({"chsh": certificate.as_dict()}, args.output)
            return
        if args.kind == "lineage":
            lineage_counts = payload.get("lineage_counts", payload) if isinstance(payload, dict) else payload
            certificate = finite_shot_lineage_certificate(lineage_counts, confidence=args.confidence)
            _print_or_write({"lineage": certificate}, args.output)
            return

    if args.command == "write-finite-shot-submission":
        with Path(args.counts_json).open() as handle:
            counts_payload = json.load(handle)
        result = write_finite_shot_submission(
            Path(args.output_dir),
            counts_payload,
            FiniteShotSubmissionConfig(
                kind=args.kind,
                confidence=args.confidence,
                task_id=args.task_id,
            ),
        )
        print(f"wrote finite-shot package to {result['output_dir']}")
        print(f"manifest: {result['manifest_path']}")
        return

    if args.command == "write-resource-kernel-submission":
        result = write_resource_kernel_submission(
            Path(args.output_dir),
            ResourceKernelSubmissionConfig(
                task_id=args.task_id,
                theta=args.theta,
                phi=args.phi,
                local_perturbation_probability=args.local_perturbation_probability,
                local_perturbation_angle=args.local_perturbation_angle,
                interaction_angle=args.interaction_angle,
                damping_probability=args.damping_probability,
                dephase_probability=args.dephase_probability,
            ),
        )
        print(f"wrote resource-kernel package to {result['output_dir']}")
        print(f"manifest: {result['manifest_path']}")
        return

    if args.command == "write-population-submission":
        result = write_population_submission(
            Path(args.output_dir),
            PopulationSubmissionConfig(
                task_id=args.task_id,
                seed=args.seed,
                steps=args.steps,
                initial_population=args.initial_population,
                carrying_capacity=args.carrying_capacity,
                lineage_null_permutations=args.lineage_null_permutations,
            ),
        )
        print(f"wrote population package to {result['output_dir']}")
        print(f"manifest: {result['manifest_path']}")
        return

    if args.command == "run-structured-population":
        result = write_structured_population_submission(
            Path(args.output_dir),
            StructuredPopulationSweepConfig(
                site_counts=args.site_counts,
                steps=args.steps,
                initial_theta=args.initial_theta,
                mutation_probability=args.mutation_probability,
                mutation_angle=args.mutation_angle,
                interaction_angle=args.interaction_angle,
                damping_probability=args.damping_probability,
            ),
        )
        print(f"wrote structured-population package to {result['output_dir']}")
        print(f"manifest: {result['manifest_path']}")
        return

    if args.command == "write-sampling-challenge-submission":
        result = write_sampling_challenge_submission(
            Path(args.output_dir),
            SamplingChallengeConfig(
                sizes=args.sizes,
                shots_per_size=args.shots_per_size,
                allowed_error=args.allowed_error,
                seed=args.seed,
                resource_depth=args.resource_depth,
            ),
        )
        print(f"wrote sampling-challenge package to {result['output_dir']}")
        print(f"manifest: {result['manifest_path']}")
        return

    print(f"unknown command {args.command!r}", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
