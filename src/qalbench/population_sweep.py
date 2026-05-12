#!/usr/bin/env python3
"""Run population-level QAL benchmark replicates.

The population artifact complements the two-qubit resource-kernel sweep. It
records explicit individuals, parent-offspring lineage metrics, turnover,
heritable variation, interaction exposure, selection gradients, and event-level
quantum-resource ablations across quantum, dephased, classical, and
no-inheritance controls.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import lzma
import shlex
import shutil
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np

from .population import PopulationParams, run_population_benchmark
from .sweep import (
    ROOT,
    _package_version,
    _repo_relative,
    _sha256,
    _source_metadata,
    _sync_paper_figures,
    _sync_package_artifacts,
    probability_arg,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
SCENARIOS = ("basis_selection", "resource_selection", "neutral")
MODELS = ("quantum", "dephased", "classical", "no_inheritance")
INDIVIDUALS_FILENAME = "qalbench_population_individuals.csv.xz"
BIRTH_EVENTS_FILENAME = "qalbench_population_birth_events.csv.xz"
ATTEMPTS_FILENAME = "qalbench_population_attempts.csv.xz"

SUMMARY_FIELDS = [
    "scenario",
    "model",
    "seed",
    "steps",
    "initial_population",
    "carrying_capacity",
    "birth_count",
    "death_count",
    "turnover_events",
    "reproduction_opportunity_count",
    "evaluated_birth_attempt_count",
    "capacity_blocked_opportunity_count",
    "capacity_blocked_fraction",
    "final_population",
    "mean_population",
    "max_lineage_depth",
    "mean_lineage_depth_alive",
    "parent_offspring_agreement",
    "parent_offspring_mutual_information",
    "shuffled_lineage_mutual_information",
    "shuffled_lineage_mutual_information_p95",
    "lineage_mi_permutation_p_value",
    "theta_parent_child_correlation",
    "mutation_event_rate",
    "transmitted_variant_rate",
    "selection_gradient_birth_rate_bit1_minus_bit0",
    "mean_birth_probability",
    "mean_event_resource_score",
    "mean_event_negativity",
    "mean_event_chsh",
    "final_p_one_alive",
    "final_shannon_diversity_alive",
    "mean_shannon_diversity_alive",
    "theta_mean_alive",
    "theta_std_alive",
    "interaction_birth_covariance",
]

TRAJECTORY_FIELDS = [
    "scenario",
    "model",
    "seed",
    "step",
    "alive_population",
    "births_this_step",
    "deaths_this_step",
    "p_one_alive",
    "shannon_diversity_alive",
    "theta_mean_alive",
    "theta_std_alive",
    "mean_resource_score_alive",
    "opposite_interaction_rate",
]

INDIVIDUAL_FIELDS = [
    "scenario",
    "model",
    "seed",
    "id",
    "parent_id",
    "birth_step",
    "death_step",
    "generation",
    "theta",
    "expressed_bit",
    "birth_event_negativity",
    "birth_event_chsh",
    "birth_event_resource_score",
    "mutation_from_parent",
    "birth_count",
]

BIRTH_EVENT_FIELDS = [
    "scenario",
    "model",
    "seed",
    "step",
    "parent_id",
    "child_id",
    "parent_bit",
    "child_bit",
    "parent_theta",
    "child_theta",
    "mutation",
    "event_negativity",
    "event_chsh",
    "event_resource_score",
    "birth_probability",
    "interaction_exposure",
]

ATTEMPT_FIELDS = [
    "scenario",
    "model",
    "seed",
    "step",
    "parent_id",
    "parent_bit",
    "parent_theta",
    "parent_generation",
    "parent_age",
    "parent_resource_score",
    "partner_id",
    "partner_bit",
    "interaction_exposure",
    "birth_probability",
    "born",
    "capacity_blocked",
    "child_id",
]


def positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {parsed}")
    return parsed


def population_size_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError(f"must be at least 2, got {parsed}")
    return parsed


def seed_values(first_seed: int, seed_count: int) -> list[int]:
    return [first_seed + index for index in range(seed_count)]


def run_population_sweep(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str | bool | None]],
    list[dict[str, float | int | str | bool]],
    list[dict[str, float | int | str | bool | None]],
]:
    summaries: list[dict[str, float | int | str]] = []
    trajectories: list[dict[str, float | int | str]] = []
    individuals: list[dict[str, float | int | str | bool | None]] = []
    birth_events: list[dict[str, float | int | str | bool]] = []
    attempts: list[dict[str, float | int | str | bool | None]] = []
    for scenario in SCENARIOS:
        for model in MODELS:
            for seed in seed_values(args.first_seed, args.seed_count):
                params = PopulationParams(
                    model=model,  # type: ignore[arg-type]
                    scenario=scenario,  # type: ignore[arg-type]
                    seed=seed,
                    steps=args.steps,
                    initial_population=args.initial_population,
                    carrying_capacity=args.carrying_capacity,
                    base_birth_probability=args.base_birth_probability,
                    base_death_probability=args.base_death_probability,
                    density_death_probability=args.density_death_probability,
                    mutation_probability=args.mutation_probability,
                    mutation_step=args.mutation_step,
                    local_perturbation_probability=args.local_perturbation_probability,
                    local_perturbation_angle=args.local_perturbation_angle,
                    damping_probability=args.damping_probability,
                    phase_damping=args.phase_damping,
                    interaction_strength=args.interaction_strength,
                    lineage_null_permutations=args.lineage_null_permutations,
                )
                result = run_population_benchmark(params)
                summaries.append(result["summary"])  # type: ignore[arg-type]
                trajectories.extend(result["trajectory"])  # type: ignore[arg-type]
                for individual in result["individuals"].values():  # type: ignore[union-attr]
                    row = {
                        "scenario": scenario,
                        "model": model,
                        "seed": seed,
                        **asdict(individual),
                    }
                    individuals.append(row)
                for event in result["birth_events"]:  # type: ignore[union-attr]
                    row = {
                        "scenario": scenario,
                        "model": model,
                        "seed": seed,
                        **asdict(event),
                    }
                    birth_events.append(row)
                for attempt in result["attempts"]:  # type: ignore[union-attr]
                    row = {
                        "scenario": scenario,
                        "model": model,
                        "seed": seed,
                        **asdict(attempt),
                    }
                    attempts.append(row)
    return summaries, trajectories, individuals, birth_events, attempts


def write_csv(
    rows: list[dict[str, float | int | str | bool | None]],
    output_path: Path,
    fieldnames: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _csv_write_handle(output_path) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def _csv_write_handle(output_path: Path) -> Iterator[TextIO]:
    if output_path.suffix == ".gz":
        with output_path.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                mtime=0,
            ) as gzip_handle:
                with io.TextIOWrapper(gzip_handle, newline="") as text_handle:
                    yield text_handle
    elif output_path.suffix == ".xz":
        with lzma.open(output_path, "wt", newline="") as text_handle:
            yield text_handle
    else:
        with output_path.open("w", newline="") as handle:
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


def _sync_population_package_artifacts(
    *,
    summary_path: Path,
    trajectory_path: Path,
    individuals_path: Path,
    birth_events_path: Path,
    attempts_path: Path,
    metadata_path: Path,
    figure_paths: list[Path],
) -> list[Path]:
    if not ((ROOT / "src" / "qalbench").resolve() == PACKAGE_ROOT.resolve()):
        return []
    data_root = PACKAGE_ROOT / "data"
    copied: list[Path] = []
    for path in [
        summary_path,
        trajectory_path,
        individuals_path,
        birth_events_path,
        attempts_path,
        metadata_path,
        *figure_paths,
    ]:
        try:
            relative_path = path.resolve().relative_to(ROOT)
        except ValueError:
            continue
        if relative_path.parts[0] not in {"results", "paper"}:
            continue
        destination = data_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied.append(destination)
    return copied


def write_metadata(
    *,
    summaries: list[dict[str, float | int | str]],
    trajectories: list[dict[str, float | int | str]],
    individuals: list[dict[str, float | int | str | bool | None]],
    birth_events: list[dict[str, float | int | str | bool]],
    attempts: list[dict[str, float | int | str | bool | None]],
    args: argparse.Namespace,
    summary_path: Path,
    trajectory_path: Path,
    individuals_path: Path,
    birth_events_path: Path,
    attempts_path: Path,
    figure_paths: list[Path],
    output_dir: Path,
    paper_figures_synced: bool,
) -> Path:
    argv = [_repo_relative(arg) for arg in sys.argv]
    figure_hashes = {
        _repo_relative(str(path.resolve())): _sha256(path)
        for path in figure_paths
        if path.exists()
    }
    metadata = {
        "artifact": "qalbench population benchmark",
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(arg) for arg in argv),
        "argv": argv,
        "package_version": _package_version(),
        "summary_row_count": len(summaries),
        "trajectory_row_count": len(trajectories),
        "individual_row_count": len(individuals),
        "birth_event_row_count": len(birth_events),
        "attempt_row_count": len(attempts),
        "summary_csv_sha256": _sha256(summary_path),
        "trajectory_csv_sha256": _sha256(trajectory_path),
        "individual_csv_sha256": _csv_content_sha256(individuals_path),
        "birth_event_csv_sha256": _csv_content_sha256(birth_events_path),
        "attempt_csv_sha256": _csv_content_sha256(attempts_path),
        "figures_sha256": figure_hashes,
        "paper_figures_synced": paper_figures_synced,
        "parameters": {
            "scenarios": list(SCENARIOS),
            "models": list(MODELS),
            "first_seed": args.first_seed,
            "seed_count": args.seed_count,
            "steps": args.steps,
            "initial_population": args.initial_population,
            "carrying_capacity": args.carrying_capacity,
            "base_birth_probability": args.base_birth_probability,
            "base_death_probability": args.base_death_probability,
            "density_death_probability": args.density_death_probability,
            "mutation_probability": args.mutation_probability,
            "mutation_step": args.mutation_step,
            "local_perturbation_probability": args.local_perturbation_probability,
            "local_perturbation_angle": args.local_perturbation_angle,
            "damping_probability": args.damping_probability,
            "phase_damping": args.phase_damping,
            "interaction_strength": args.interaction_strength,
            "lineage_null_permutations": args.lineage_null_permutations,
        },
        "source": _source_metadata(),
        "software": {
            "python": sys.version.split()[0],
            "python_full": sys.version,
            "numpy": np.__version__,
        },
    }
    try:
        import matplotlib

        metadata["software"]["matplotlib"] = matplotlib.__version__
        metadata["software"]["matplotlib_backend"] = matplotlib.get_backend()
    except ModuleNotFoundError:
        metadata["software"]["matplotlib"] = None
        metadata["software"]["matplotlib_backend"] = None

    metadata_path = output_dir / "qalbench_population_metadata.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata_path


def _mean_by_group(
    rows: list[dict[str, float | int | str]],
    key: str,
) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        group = (str(row["scenario"]), str(row["model"]))
        values.setdefault(group, []).append(float(row[key]))
    return {group: float(np.mean(group_values)) for group, group_values in values.items()}


def _mean_sd_by_group(
    rows: list[dict[str, float | int | str]],
    key: str,
) -> dict[tuple[str, str], tuple[float, float]]:
    values: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        group = (str(row["scenario"]), str(row["model"]))
        values.setdefault(group, []).append(float(row[key]))
    return {
        group: (
            float(np.mean(group_values)),
            float(np.std(group_values, ddof=1)) if len(group_values) > 1 else 0.0,
        )
        for group, group_values in values.items()
    }


def write_figures(
    summaries: list[dict[str, float | int | str]],
    trajectories: list[dict[str, float | int | str]],
    output_dir: Path,
    *,
    paper_figure_dir: Path | None,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for PNG figures; rerun with --no-figures "
            "to write CSV only"
        ) from exc

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = [
        figure_dir / "population_outcomes.png",
        figure_dir / "lineage_nulls.png",
        figure_dir / "resource_relevance.png",
    ]

    colors = {
        "quantum": "#1f77b4",
        "dephased": "#ff7f0e",
        "classical": "#2ca02c",
        "no_inheritance": "#7f7f7f",
    }
    x = np.arange(len(SCENARIOS))
    width = 0.18

    final_population = _mean_sd_by_group(summaries, "final_population")
    lineage_depth = _mean_sd_by_group(summaries, "max_lineage_depth")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True)
    for index, model in enumerate(MODELS):
        offsets = x + (index - 1.5) * width
        final_values = [final_population[(scenario, model)] for scenario in SCENARIOS]
        axes[0].bar(
            offsets,
            [value[0] for value in final_values],
            yerr=[value[1] for value in final_values],
            width=width,
            label=model.replace("_", " "),
            color=colors[model],
            capsize=2,
        )
        lineage_values = [lineage_depth[(scenario, model)] for scenario in SCENARIOS]
        axes[1].bar(
            offsets,
            [value[0] for value in lineage_values],
            yerr=[value[1] for value in lineage_values],
            width=width,
            color=colors[model],
            capsize=2,
        )
    axes[0].set_ylabel("final population")
    axes[1].set_ylabel("max lineage depth")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels([scenario.replace("_", "\n") for scenario in SCENARIOS])
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Population outcomes across benchmark scenarios")
    fig.tight_layout()
    fig.savefig(figure_paths[0], dpi=220)
    plt.close(fig)

    lineage_mi = _mean_sd_by_group(summaries, "parent_offspring_mutual_information")
    shuffled_mi = _mean_by_group(summaries, "shuffled_lineage_mutual_information")
    theta_corr = _mean_sd_by_group(summaries, "theta_parent_child_correlation")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for index, model in enumerate(MODELS):
        offsets = x + (index - 1.5) * width
        lineage_values = [lineage_mi[(scenario, model)] for scenario in SCENARIOS]
        axes[0].bar(
            offsets,
            [value[0] for value in lineage_values],
            yerr=[value[1] for value in lineage_values],
            width=width,
            label=model.replace("_", " "),
            color=colors[model],
            capsize=2,
        )
        axes[0].plot(
            offsets,
            [shuffled_mi[(scenario, model)] for scenario in SCENARIOS],
            marker="o",
            linestyle="",
            color="black",
            markersize=3,
        )
        theta_values = [theta_corr[(scenario, model)] for scenario in SCENARIOS]
        axes[1].bar(
            offsets,
            [value[0] for value in theta_values],
            yerr=[value[1] for value in theta_values],
            width=width,
            color=colors[model],
            capsize=2,
        )
    axes[0].set_ylabel("parent-offspring MI (bits)")
    axes[1].set_ylabel("parent-child theta correlation")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels([scenario.replace("_", "\n") for scenario in SCENARIOS])
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[0].set_title("points mark shuffled-lineage nulls")
    axes[1].set_title("continuous genotype-parameter transmission")
    fig.tight_layout()
    fig.savefig(figure_paths[1], dpi=220)
    plt.close(fig)

    resource_rows = [
        row
        for row in trajectories
        if row["scenario"] == "resource_selection" and int(row["step"]) % 4 == 0
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True)
    for model in MODELS:
        model_rows = [row for row in resource_rows if row["model"] == model]
        steps = sorted({int(row["step"]) for row in model_rows})
        population_means = []
        resource_means = []
        for step in steps:
            step_rows = [row for row in model_rows if int(row["step"]) == step]
            population_means.append(float(np.mean([row["alive_population"] for row in step_rows])))
            resource_means.append(float(np.mean([row["mean_resource_score_alive"] for row in step_rows])))
        population_sds = []
        resource_sds = []
        for step in steps:
            step_rows = [row for row in model_rows if int(row["step"]) == step]
            population_values = [float(row["alive_population"]) for row in step_rows]
            resource_values = [float(row["mean_resource_score_alive"]) for row in step_rows]
            population_sds.append(float(np.std(population_values, ddof=1)) if len(population_values) > 1 else 0.0)
            resource_sds.append(float(np.std(resource_values, ddof=1)) if len(resource_values) > 1 else 0.0)
        axes[0].plot(steps, population_means, label=model.replace("_", " "), color=colors[model])
        axes[1].plot(steps, resource_means, label=model.replace("_", " "), color=colors[model])
        axes[0].fill_between(
            steps,
            np.asarray(population_means) - np.asarray(population_sds),
            np.asarray(population_means) + np.asarray(population_sds),
            color=colors[model],
            alpha=0.12,
            linewidth=0,
        )
        axes[1].fill_between(
            steps,
            np.asarray(resource_means) - np.asarray(resource_sds),
            np.asarray(resource_means) + np.asarray(resource_sds),
            color=colors[model],
            alpha=0.12,
            linewidth=0,
        )
    axes[0].set_ylabel("alive population")
    axes[1].set_ylabel("mean resource score")
    for axis in axes:
        axis.set_xlabel("step")
        axis.grid(alpha=0.25)
    axes[0].set_title("resource-coupled population dynamics")
    axes[1].set_title("birth-event score among living individuals")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_paths[2], dpi=220)
    plt.close(fig)

    if paper_figure_dir is not None:
        return figure_paths + _sync_paper_figures(figure_paths, paper_figure_dir)
    return figure_paths


def paper_figure_dir_for_args(args: argparse.Namespace, output_dir: Path) -> Path | None:
    if args.sync_paper_figures is not None:
        return ROOT / "paper" / "figures" if args.sync_paper_figures else None
    if output_dir.resolve() == (ROOT / "results").resolve():
        return ROOT / "paper" / "figures"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--first-seed", type=int, default=11)
    parser.add_argument("--seed-count", type=positive_int_arg, default=100)
    parser.add_argument("--steps", type=positive_int_arg, default=64)
    parser.add_argument("--initial-population", type=population_size_arg, default=24)
    parser.add_argument("--carrying-capacity", type=population_size_arg, default=80)
    parser.add_argument("--base-birth-probability", type=probability_arg, default=0.28)
    parser.add_argument("--base-death-probability", type=probability_arg, default=0.06)
    parser.add_argument("--density-death-probability", type=probability_arg, default=0.08)
    parser.add_argument("--mutation-probability", type=probability_arg, default=0.08)
    parser.add_argument("--mutation-step", type=float, default=0.35)
    parser.add_argument("--local-perturbation-probability", type=probability_arg, default=0.15)
    parser.add_argument("--local-perturbation-angle", type=float, default=np.pi / 3.0)
    parser.add_argument("--damping-probability", type=probability_arg, default=0.10)
    parser.add_argument("--phase-damping", type=probability_arg, default=0.0)
    parser.add_argument("--interaction-strength", type=float, default=0.12)
    parser.add_argument("--lineage-null-permutations", type=positive_int_arg, default=200)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--sync-paper-figures",
        dest="sync_paper_figures",
        action="store_true",
        default=None,
        help=(
            "copy generated figures into paper/figures; default is enabled only "
            "when --output-dir is the repository results directory"
        ),
    )
    parser.add_argument(
        "--no-sync-paper-figures",
        dest="sync_paper_figures",
        action="store_false",
        help="do not copy generated figures into paper/figures",
    )
    parser.add_argument(
        "--sync-package-artifacts",
        action="store_true",
        help="copy generated population artifacts into src/qalbench/data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    summaries, trajectories, individuals, birth_events, attempts = run_population_sweep(args)
    summary_path = output_dir / "qalbench_population.csv"
    trajectory_path = output_dir / "qalbench_population_timeseries.csv"
    individuals_path = output_dir / INDIVIDUALS_FILENAME
    birth_events_path = output_dir / BIRTH_EVENTS_FILENAME
    attempts_path = output_dir / ATTEMPTS_FILENAME
    write_csv(summaries, summary_path, SUMMARY_FIELDS)
    write_csv(trajectories, trajectory_path, TRAJECTORY_FIELDS)
    write_csv(individuals, individuals_path, INDIVIDUAL_FIELDS)
    write_csv(birth_events, birth_events_path, BIRTH_EVENT_FIELDS)
    write_csv(attempts, attempts_path, ATTEMPT_FIELDS)
    paper_figure_dir = None
    figure_paths: list[Path] = []
    if not args.no_figures:
        paper_figure_dir = paper_figure_dir_for_args(args, output_dir)
        figure_paths = write_figures(
            summaries,
            trajectories,
            output_dir,
            paper_figure_dir=paper_figure_dir,
        )
    metadata_path = write_metadata(
        summaries=summaries,
        trajectories=trajectories,
        individuals=individuals,
        birth_events=birth_events,
        attempts=attempts,
        args=args,
        summary_path=summary_path,
        trajectory_path=trajectory_path,
        individuals_path=individuals_path,
        birth_events_path=birth_events_path,
        attempts_path=attempts_path,
        figure_paths=figure_paths,
        output_dir=output_dir,
        paper_figures_synced=paper_figure_dir is not None,
    )
    package_artifact_paths: list[Path] = []
    if args.sync_package_artifacts:
        if output_dir.resolve() != (ROOT / "results").resolve():
            raise SystemExit(
                "--sync-package-artifacts is only allowed with the repository "
                "default results directory"
            )
        package_artifact_paths = _sync_population_package_artifacts(
            summary_path=summary_path,
            trajectory_path=trajectory_path,
            individuals_path=individuals_path,
            birth_events_path=birth_events_path,
            attempts_path=attempts_path,
            metadata_path=metadata_path,
            figure_paths=figure_paths,
        )
    print(f"wrote {len(summaries)} summary rows to {summary_path}")
    print(f"wrote {len(trajectories)} trajectory rows to {trajectory_path}")
    print(f"wrote {len(individuals)} individual rows to {individuals_path}")
    print(f"wrote {len(birth_events)} birth-event rows to {birth_events_path}")
    print(f"wrote {len(attempts)} attempt rows to {attempts_path}")
    print(f"wrote metadata to {metadata_path}")
    if not args.no_figures:
        print(f"wrote figures to {output_dir / 'figures'}")
        if paper_figure_dir is not None:
            print(f"synced paper figures to {paper_figure_dir}")
    if package_artifact_paths:
        print(f"synced population package artifacts to {PACKAGE_ROOT / 'data'}")


if __name__ == "__main__":
    main()
