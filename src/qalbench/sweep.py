#!/usr/bin/env python3
"""Run parameter sweeps for the minimal two-qubit QAL resource kernel.

Outputs are written to ``results/qalbench_sweep.csv`` and PNG figures under
``results/figures`` by default. The default output directory also syncs
paper-ready copies to ``paper/figures``. Installed console scripts use the
current working directory as the artifact root unless ``QALBENCH_ROOT`` is set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "numpy is required to run qalbench; install numpy in your Python "
        "environment, and matplotlib if you want PNG figures"
    ) from exc

ROOT = Path(os.environ.get("QALBENCH_ROOT", Path.cwd())).resolve()
PACKAGE_ROOT = Path(__file__).resolve().parent

from . import ClassicalParams, QuantumParams, compute_metrics
from .classical import BITSTRINGS, run_classical_event
from .quantum import run_quantum_event


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


def canonical_fieldnames() -> list[str]:
    return (
        PARAMETER_FIELDS
        + [f"quantum_{suffix}" for suffix in METRIC_SUFFIXES]
        + [f"dephased_{suffix}" for suffix in METRIC_SUFFIXES]
        + [f"classical_{suffix}" for suffix in METRIC_SUFFIXES]
    )


def probability_arg(value: str) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0, 1], got {probability}")
    return probability


def grid_count_arg(value: str) -> int:
    count = int(value)
    if count < 2:
        raise argparse.ArgumentTypeError(f"must be at least 2, got {count}")
    return count


def _float_grid(start: float, stop: float, count: int) -> np.ndarray:
    return np.linspace(start, stop, count)


def run_sweep(args: argparse.Namespace) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    theta_values = _float_grid(0.0, np.pi, args.theta_count)
    perturbation_values = _float_grid(
        0.0, args.max_local_perturbation_probability, args.local_perturbation_count
    )
    damping_values = _float_grid(0.0, args.max_damping, args.damping_count)

    for theta in theta_values:
        for local_perturbation_probability in perturbation_values:
            for damping_probability in damping_values:
                qparams = QuantumParams(
                    theta=float(theta),
                    phi=args.phi,
                    local_perturbation_probability=float(local_perturbation_probability),
                    local_perturbation_angle=args.local_perturbation_angle,
                    interaction_angle=args.interaction_angle,
                    damping_probability=float(damping_probability),
                    dephase_probability=args.phase_damping,
                )
                cparams = ClassicalParams(
                    theta=float(theta),
                    local_perturbation_probability=float(local_perturbation_probability),
                    local_perturbation_angle=args.local_perturbation_angle,
                    damping_probability=float(damping_probability),
                )

                qstates = run_quantum_event(qparams)
                classical = run_classical_event(cparams)
                probabilities = classical["probabilities"]

                row: dict[str, float | str] = {
                    "theta": float(theta),
                    "phi": args.phi,
                    "local_perturbation_probability": float(local_perturbation_probability),
                    "local_perturbation_angle": args.local_perturbation_angle,
                    "interaction_angle": args.interaction_angle,
                    "damping_probability": float(damping_probability),
                    "phase_damping": args.phase_damping,
                }
                row.update(compute_metrics(qstates["rho"], prefix="quantum_"))
                row.update(compute_metrics(qstates["dephased"], prefix="dephased_"))
                row.update(compute_metrics(classical["rho"], prefix="classical_"))
                for bitstring, probability in zip(BITSTRINGS, probabilities):
                    row[f"classical_p_{bitstring}"] = float(probability)
                rows.append(row)
    return rows


def write_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = canonical_fieldnames()
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return value


def _relative_argv() -> list[str]:
    return [_repo_relative(arg) for arg in sys.argv]


def _package_version() -> str | None:
    try:
        return version("qalbench")
    except PackageNotFoundError:
        return None


def _git_output(args: list[str], *, empty_as_none: bool = True) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, ValueError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    if empty_as_none and not output:
        return None
    return output


def _tracked_or_source_paths() -> list[Path]:
    output = _git_output(
        ["ls-files", "--cached", "--others", "--exclude-standard"],
        empty_as_none=True,
    )
    if output is None:
        candidates = [
            "pyproject.toml",
            "requirements.txt",
            "requirements-lock.txt",
            "README.md",
            "CITATION.cff",
            "ARCHIVE_RELEASE.md",
            "paper/main.tex",
            "paper/references.bib",
        ]
        for directory in ("src", "scripts", "tests"):
            root = ROOT / directory
            if root.exists():
                candidates.extend(
                    str(path.relative_to(ROOT))
                    for path in root.rglob("*")
                    if path.is_file()
                )
    else:
        candidates = output.splitlines()

    excluded_prefixes = (
        "build/",
        "paper/build/",
        "paper/figures/",
        "results/",
        "src/qalbench/data/",
    )
    paths = []
    for candidate in candidates:
        if candidate.startswith(excluded_prefixes):
            continue
        path = ROOT / candidate
        if path.is_file():
            paths.append(path)
    return sorted(set(paths), key=lambda path: str(path.relative_to(ROOT)))


def _source_file_manifest() -> dict[str, object]:
    files = {
        _repo_relative(str(path)): _sha256(path)
        for path in _tracked_or_source_paths()
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "source_files_sha256": files,
        "source_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _source_metadata() -> dict[str, object]:
    status = _git_status()
    metadata: dict[str, str | bool | None | object] = {
        "package_version": _package_version(),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "git_tag": _git_output(["describe", "--tags", "--exact-match"]),
        "git_describe": _git_output(["describe", "--tags", "--always", "--dirty"]),
        "git_dirty": None if status is None else bool(status),
    }
    metadata.update(_source_file_manifest())
    return metadata


def _git_status() -> str | None:
    return _git_output(["status", "--porcelain"], empty_as_none=False)


def _require_clean_tree_for_package_artifact_sync() -> None:
    status = _git_status()
    if status is None:
        raise SystemExit(
            "--sync-package-artifacts requires a Git checkout so artifact "
            "metadata can record release provenance"
        )
    if status:
        raise SystemExit(
            "--sync-package-artifacts requires a clean Git tree; commit or "
            "stash source changes before refreshing packaged canonical artifacts"
        )


def write_metadata(
    *,
    rows: list[dict[str, float | str]],
    args: argparse.Namespace,
    csv_path: Path,
    figure_paths: list[Path],
    output_dir: Path,
    paper_figures_synced: bool,
) -> Path:
    argv = _relative_argv()
    figure_hashes = {
        _repo_relative(str(path.resolve())): _sha256(path)
        for path in figure_paths
        if path.exists()
    }
    metadata = {
        "artifact": "qalbench two-qubit resource-kernel sweep",
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(arg) for arg in argv),
        "argv": argv,
        "row_count": len(rows),
        "csv_sha256": _sha256(csv_path),
        "figures_sha256": figure_hashes,
        "paper_figures_synced": paper_figures_synced,
        "parameters": {
            "theta_count": args.theta_count,
            "local_perturbation_count": args.local_perturbation_count,
            "damping_count": args.damping_count,
            "max_local_perturbation_probability": args.max_local_perturbation_probability,
            "local_perturbation_angle": args.local_perturbation_angle,
            "max_damping": args.max_damping,
            "phi": args.phi,
            "interaction_angle": args.interaction_angle,
            "phase_damping": args.phase_damping,
        },
        "source": _source_metadata(),
        "software": {
            "python": sys.version.split()[0],
            "python_full": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    try:
        import matplotlib

        metadata["software"]["matplotlib"] = matplotlib.__version__
        metadata["software"]["matplotlib_backend"] = matplotlib.get_backend()
        try:
            from matplotlib import ft2font

            metadata["software"]["freetype"] = ft2font.__freetype_version__
        except (ImportError, AttributeError):
            metadata["software"]["freetype"] = None
        try:
            from matplotlib import font_manager, rcParams

            metadata["software"]["font_family"] = rcParams.get("font.family")
            metadata["software"]["font_sans_serif"] = rcParams.get("font.sans-serif")
            default_font = font_manager.findfont(
                font_manager.FontProperties(family=rcParams.get("font.family")),
                fallback_to_default=True,
            )
            metadata["software"]["default_font"] = Path(default_font).name
        except Exception:
            metadata["software"]["font_family"] = None
            metadata["software"]["font_sans_serif"] = None
            metadata["software"]["default_font"] = None
    except ModuleNotFoundError:
        metadata["software"]["matplotlib"] = None
        metadata["software"]["matplotlib_backend"] = None
        metadata["software"]["freetype"] = None
        metadata["software"]["font_family"] = None
        metadata["software"]["font_sans_serif"] = None
        metadata["software"]["default_font"] = None

    metadata_path = output_dir / "qalbench_sweep_metadata.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata_path


def _nearest(values: list[float], target: float) -> float:
    return min(values, key=lambda value: abs(value - target))


def _rows_at(
    rows: list[dict[str, float | str]],
    *,
    theta: float | None = None,
    local_perturbation_probability: float | None = None,
    damping_probability: float | None = None,
) -> list[dict[str, float | str]]:
    out = rows
    if theta is not None:
        out = [row for row in out if np.isclose(float(row["theta"]), theta)]
    if local_perturbation_probability is not None:
        out = [
            row
            for row in out
            if np.isclose(
                float(row["local_perturbation_probability"]),
                local_perturbation_probability,
            )
        ]
    if damping_probability is not None:
        out = [
            row
            for row in out
            if np.isclose(float(row["damping_probability"]), damping_probability)
        ]
    return out


def _p_one_from_z(z_value: float | str) -> float:
    return 0.5 * (1.0 - float(z_value))


def _sync_paper_figures(figure_paths: list[Path], paper_figure_dir: Path) -> list[Path]:
    paper_figure_dir.mkdir(parents=True, exist_ok=True)
    paper_paths = []
    for path in figure_paths:
        paper_path = paper_figure_dir / path.name
        shutil.copy2(path, paper_path)
        paper_paths.append(paper_path)
    return paper_paths


def _is_source_checkout_root() -> bool:
    return (ROOT / "src" / "qalbench").resolve() == PACKAGE_ROOT.resolve()


def _sync_package_artifacts(
    *,
    csv_path: Path,
    metadata_path: Path,
    figure_paths: list[Path],
) -> list[Path]:
    """Mirror canonical artifacts into package data for wheel-installed verify."""

    if not _is_source_checkout_root():
        return []
    data_root = PACKAGE_ROOT / "data"
    copied: list[Path] = []
    for path in [csv_path, metadata_path, *figure_paths]:
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


def write_figures(
    rows: list[dict[str, float | str]],
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
        figure_dir / "z_population.png",
        figure_dir / "quantum_diagnostics.png",
        figure_dir / "phase_diagram.png",
    ]

    theta_values = sorted({float(row["theta"]) for row in rows})
    perturbation_values = sorted(
        {float(row["local_perturbation_probability"]) for row in rows}
    )
    damping_values = sorted({float(row["damping_probability"]) for row in rows})
    theta_mid = _nearest(theta_values, np.pi / 2.0)
    perturbation_zero = _nearest(perturbation_values, 0.0)
    if not np.isclose(theta_mid, np.pi / 2.0):
        print(
            f"warning: figure slice uses theta={theta_mid:.8g}, not pi/2",
            file=sys.stderr,
        )
    if not np.isclose(perturbation_zero, 0.0):
        print(
            "warning: figure slice uses "
            f"local_perturbation_probability={perturbation_zero:.8g}, not 0",
            file=sys.stderr,
        )

    z_rows = sorted(
        _rows_at(
            rows,
            theta=theta_mid,
            local_perturbation_probability=perturbation_zero,
        ),
        key=lambda row: float(row["damping_probability"]),
    )
    damping = np.array([float(row["damping_probability"]) for row in z_rows])
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].plot(
        damping,
        [_p_one_from_z(row["quantum_z_offspring"]) for row in z_rows],
        label="full quantum",
        linewidth=2.2,
    )
    axes[0].plot(
        damping,
        [_p_one_from_z(row["dephased_z_offspring"]) for row in z_rows],
        label="dephased quantum",
        linestyle="--",
        linewidth=2.0,
    )
    axes[0].plot(
        damping,
        [_p_one_from_z(row["classical_z_offspring"]) for row in z_rows],
        label="classical Markov",
        linestyle=":",
        linewidth=2.5,
    )
    quantum_residual = np.array(
        [
            abs(
                _p_one_from_z(row["quantum_z_offspring"])
                - _p_one_from_z(row["classical_z_offspring"])
            )
            for row in z_rows
        ]
    )
    dephased_residual = np.array(
        [
            abs(
                _p_one_from_z(row["dephased_z_offspring"])
                - _p_one_from_z(row["classical_z_offspring"])
            )
            for row in z_rows
        ]
    )
    axes[1].plot(damping, quantum_residual, label="|quantum - Markov|", linewidth=1.8)
    axes[1].plot(
        damping,
        dephased_residual,
        label="|dephased - Markov|",
        linewidth=1.8,
        linestyle="--",
    )
    axes[0].set_ylabel("offspring P(1) in Z basis")
    axes[0].set_title("Z-basis kernel metric is preserved by classical baselines")
    axes[0].legend()
    axes[1].set_xlabel("amplitude damping probability (loss analogue)")
    axes[1].set_ylabel("abs. residual")
    axes[1].set_yscale("symlog", linthresh=1e-16)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_paths[0], dpi=220)
    plt.close()

    diagnostic_rows = z_rows
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(
        damping,
        [row["quantum_negativity"] for row in diagnostic_rows],
        label="negativity",
        linewidth=2.2,
    )
    axes[0].plot(
        damping,
        [row["quantum_concurrence"] for row in diagnostic_rows],
        label="concurrence",
        linewidth=2.2,
    )
    axes[0].plot(
        damping,
        [row["dephased_negativity"] for row in diagnostic_rows],
        label="dephased negativity",
        linestyle="--",
        linewidth=2.0,
    )
    axes[0].plot(
        damping,
        [row["classical_negativity"] for row in diagnostic_rows],
        label="Markov negativity",
        linestyle=":",
        linewidth=2.2,
    )
    axes[0].plot(
        damping,
        [row["classical_concurrence"] for row in diagnostic_rows],
        label="Markov concurrence",
        linestyle="-.",
        linewidth=1.7,
    )
    axes[0].set_xlabel("amplitude damping probability (loss analogue)")
    axes[0].set_ylabel("entanglement measure")
    axes[0].set_title("Entanglement dies while Z basis probabilities remain")
    axes[0].legend()

    axes[1].plot(
        damping,
        [row["quantum_chsh_max"] for row in diagnostic_rows],
        label="full quantum CHSH max",
        linewidth=2.2,
    )
    axes[1].plot(
        damping,
        [row["dephased_chsh_max"] for row in diagnostic_rows],
        label="dephased CHSH max",
        linestyle="--",
        linewidth=2.0,
    )
    axes[1].plot(
        damping,
        [row["classical_chsh_max"] for row in diagnostic_rows],
        label="Markov CHSH max",
        linestyle=":",
        linewidth=2.2,
    )
    axes[1].axhline(2.0, color="0.35", linewidth=1.0, linestyle=":", label="local bound")
    axes[1].set_xlabel("amplitude damping probability (loss analogue)")
    axes[1].set_ylabel("CHSH maximum")
    axes[1].set_title("Noncommuting diagnostic separates models")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figure_paths[1], dpi=220)
    plt.close()

    heat = np.full((len(damping_values), len(perturbation_values)), np.nan)
    for row in _rows_at(rows, theta=theta_mid):
        damping_idx = damping_values.index(float(row["damping_probability"]))
        perturbation_idx = perturbation_values.index(
            float(row["local_perturbation_probability"])
        )
        heat[damping_idx, perturbation_idx] = float(row["quantum_negativity"])

    plt.figure(figsize=(7.2, 5.2))
    image = plt.imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=[
            min(perturbation_values),
            max(perturbation_values),
            min(damping_values),
            max(damping_values),
        ],
        cmap="viridis",
    )
    plt.colorbar(image, label="negativity")
    plt.xlabel("local-perturbation probability")
    plt.ylabel("amplitude damping probability (loss analogue)")
    plt.title("Negativity under stochastic local perturbation and damping")
    plt.tight_layout()
    plt.savefig(figure_paths[2], dpi=220)
    plt.close()
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
    parser.add_argument("--theta-count", type=grid_count_arg, default=41)
    parser.add_argument("--local-perturbation-count", type=grid_count_arg, default=21)
    parser.add_argument("--damping-count", type=grid_count_arg, default=21)
    parser.add_argument(
        "--max-local-perturbation-probability",
        type=probability_arg,
        default=1.0,
    )
    parser.add_argument("--local-perturbation-angle", type=float, default=np.pi / 3.0)
    parser.add_argument("--max-damping", type=probability_arg, default=1.0)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--interaction-angle", type=float, default=0.0)
    parser.add_argument("--phase-damping", type=probability_arg, default=0.0)
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
        help=(
            "copy generated canonical artifacts into src/qalbench/data; intended "
            "only when deliberately refreshing package reference data"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sync_package_artifacts:
        _require_clean_tree_for_package_artifact_sync()
    output_dir = Path(args.output_dir)
    rows = run_sweep(args)
    csv_path = output_dir / "qalbench_sweep.csv"
    write_csv(rows, csv_path)
    figure_paths: list[Path] = []
    paper_figure_dir = None
    if not args.no_figures:
        paper_figure_dir = paper_figure_dir_for_args(args, output_dir)
        figure_paths = write_figures(
            rows,
            output_dir,
            paper_figure_dir=paper_figure_dir,
        )
    metadata_path = write_metadata(
        rows=rows,
        args=args,
        csv_path=csv_path,
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
        package_artifact_paths = _sync_package_artifacts(
            csv_path=csv_path,
            metadata_path=metadata_path,
            figure_paths=figure_paths,
        )
    print(f"wrote {len(rows)} rows to {csv_path}")
    print(f"wrote metadata to {metadata_path}")
    if not args.no_figures:
        print(f"wrote figures to {output_dir / 'figures'}")
        if paper_figure_dir is not None:
            print(f"synced paper figures to {paper_figure_dir}")
        else:
            print("did not sync paper figures")
    if package_artifact_paths:
        print(f"synced package artifacts to {PACKAGE_ROOT / 'data'}")


if __name__ == "__main__":
    main()
