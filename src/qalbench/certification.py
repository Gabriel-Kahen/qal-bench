"""Finite-shot certification utilities for QALBench submissions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log, log2, sqrt
from statistics import NormalDist
from typing import Mapping


@dataclass(frozen=True)
class ConfidenceInterval:
    """Closed confidence interval for one scalar estimate."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    shots: int
    method: str

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class CHSHCertificate:
    """Finite-shot CHSH certificate using four two-outcome setting counts."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    local_bound: float
    certified: bool
    setting_estimates: dict[str, ConfidenceInterval]

    def as_dict(self) -> dict[str, object]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "local_bound": self.local_bound,
            "certified": self.certified,
            "setting_estimates": {
                key: interval.as_dict()
                for key, interval in self.setting_estimates.items()
            },
        }


def _validate_confidence(confidence: float) -> float:
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return confidence


def validate_counts(counts: Mapping[str, int]) -> int:
    """Return total shots after checking that counts are nonnegative integers."""

    total = 0
    for outcome, value in counts.items():
        if int(value) != value or value < 0:
            raise ValueError(f"count for outcome {outcome!r} must be a nonnegative integer")
        total += int(value)
    if total <= 0:
        raise ValueError("at least one shot is required")
    return total


def wilson_interval(
    successes: int,
    shots: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Wilson score interval for a binomial proportion."""

    confidence = _validate_confidence(confidence)
    if shots <= 0:
        raise ValueError("shots must be positive")
    if successes < 0 or successes > shots:
        raise ValueError("successes must be in [0, shots]")

    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    phat = successes / shots
    denom = 1.0 + z * z / shots
    center = (phat + z * z / (2.0 * shots)) / denom
    radius = z * sqrt((phat * (1.0 - phat) + z * z / (4.0 * shots)) / shots) / denom
    return ConfidenceInterval(
        estimate=float(phat),
        lower=max(0.0, float(center - radius)),
        upper=min(1.0, float(center + radius)),
        confidence=confidence,
        shots=shots,
        method="wilson",
    )


def hoeffding_interval(
    estimate: float,
    shots: int,
    confidence: float = 0.95,
    *,
    value_range: float = 2.0,
) -> ConfidenceInterval:
    """Distribution-free Hoeffding interval for a bounded sample mean."""

    confidence = _validate_confidence(confidence)
    if shots <= 0:
        raise ValueError("shots must be positive")
    alpha = 1.0 - confidence
    radius = value_range * sqrt(log(2.0 / alpha) / (2.0 * shots))
    return ConfidenceInterval(
        estimate=float(estimate),
        lower=float(estimate - radius),
        upper=float(estimate + radius),
        confidence=confidence,
        shots=shots,
        method="hoeffding",
    )


def copy_agreement_certificate(
    counts: Mapping[str, int],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Estimate two-bit copy agreement from computational-basis counts."""

    shots = validate_counts(counts)
    successes = 0
    for outcome, count in counts.items():
        if len(outcome) < 2:
            raise ValueError(f"copy-agreement outcome {outcome!r} is not a bitstring")
        successes += int(count) if outcome[0] == outcome[1] else 0
    return wilson_interval(successes, shots, confidence=confidence)


def binary_lineage_mutual_information(counts: Mapping[str, int]) -> float:
    """Return plug-in mutual information in bits for parent/child bit counts."""

    shots = validate_counts(counts)
    joint = [[0.0, 0.0], [0.0, 0.0]]
    for outcome, count in counts.items():
        if len(outcome) < 2 or outcome[0] not in "01" or outcome[1] not in "01":
            raise ValueError(f"lineage outcome {outcome!r} must start with parent/child bits")
        joint[int(outcome[0])][int(outcome[1])] += int(count) / shots

    parent = [joint[0][0] + joint[0][1], joint[1][0] + joint[1][1]]
    child = [joint[0][0] + joint[1][0], joint[0][1] + joint[1][1]]
    mutual_information = 0.0
    for parent_bit in (0, 1):
        for child_bit in (0, 1):
            probability = joint[parent_bit][child_bit]
            if probability > 0.0 and parent[parent_bit] > 0.0 and child[child_bit] > 0.0:
                mutual_information += probability * log2(
                    probability / (parent[parent_bit] * child[child_bit])
                )
    return float(max(0.0, min(1.0, mutual_information)))


def _bounded_unit_interval(
    estimate: float,
    shots: int,
    confidence: float,
    *,
    method: str,
) -> ConfidenceInterval:
    interval = hoeffding_interval(
        estimate,
        shots,
        confidence=confidence,
        value_range=1.0,
    )
    return ConfidenceInterval(
        estimate=interval.estimate,
        lower=max(0.0, interval.lower),
        upper=min(1.0, interval.upper),
        confidence=interval.confidence,
        shots=interval.shots,
        method=method,
    )


def lineage_count_certificate(
    counts: Mapping[str, int],
    confidence: float = 0.95,
) -> dict[str, object]:
    """Finite-shot parent/child bit-count summary for a lineage group.

    The copy-agreement, mutation-rate, parent-one, and child-one intervals are
    Wilson binomial intervals. The lineage mutual-information interval is a
    conservative bounded plug-in interval over the binary range [0, 1] bits; it
    is intended for audit triage, while certification should rely on explicit
    control comparisons.
    """

    shots = validate_counts(counts)
    agreements = 0
    parent_ones = 0
    child_ones = 0
    for outcome, count in counts.items():
        if len(outcome) < 2 or outcome[0] not in "01" or outcome[1] not in "01":
            raise ValueError(f"lineage outcome {outcome!r} must start with parent/child bits")
        value = int(count)
        agreements += value if outcome[0] == outcome[1] else 0
        parent_ones += value if outcome[0] == "1" else 0
        child_ones += value if outcome[1] == "1" else 0

    mutations = shots - agreements
    mutual_information = binary_lineage_mutual_information(counts)
    return {
        "shots": shots,
        "copy_agreement_interval": wilson_interval(
            agreements,
            shots,
            confidence=confidence,
        ).as_dict(),
        "mutation_rate_interval": wilson_interval(
            mutations,
            shots,
            confidence=confidence,
        ).as_dict(),
        "parent_one_interval": wilson_interval(
            parent_ones,
            shots,
            confidence=confidence,
        ).as_dict(),
        "child_one_interval": wilson_interval(
            child_ones,
            shots,
            confidence=confidence,
        ).as_dict(),
        "lineage_mi_interval": _bounded_unit_interval(
            mutual_information,
            shots,
            confidence,
            method="bounded_plugin_hoeffding_binary_mi",
        ).as_dict(),
    }


def finite_shot_lineage_certificate(
    lineage_counts: Mapping[str, Mapping[str, int]] | Mapping[str, int],
    confidence: float = 0.95,
) -> dict[str, object]:
    """Certificate payload for observed lineage counts and optional controls.

    ``lineage_counts`` may be a direct parent/child count dictionary or a group
    mapping with keys such as ``observed``, ``no_inheritance``, and
    ``shuffled_lineage``. When control groups are present, the returned
    ``certified_inheritance`` flag requires the observed copy-agreement lower
    bound to exceed every control copy-agreement upper bound.
    """

    confidence = _validate_confidence(confidence)
    if not isinstance(lineage_counts, Mapping) or not lineage_counts:
        raise ValueError("lineage_counts must be a nonempty object")
    if all(isinstance(value, int | float) for value in lineage_counts.values()):
        groups: Mapping[str, Mapping[str, int]] = {"observed": lineage_counts}  # type: ignore[assignment]
    else:
        groups = lineage_counts  # type: ignore[assignment]

    group_certificates: dict[str, object] = {}
    for group, counts in groups.items():
        if not isinstance(counts, Mapping):
            raise ValueError(f"lineage_counts.{group} must be an object")
        group_certificates[str(group)] = lineage_count_certificate(
            counts,
            confidence=confidence,
        )

    observed = group_certificates.get("observed")
    observed_lower: float | None = None
    control_upper: float | None = None
    certified = False
    if isinstance(observed, Mapping):
        copy_interval = observed.get("copy_agreement_interval")
        if isinstance(copy_interval, Mapping) and isinstance(copy_interval.get("lower"), int | float):
            observed_lower = float(copy_interval["lower"])
    control_uppers: list[float] = []
    for group, certificate in group_certificates.items():
        if group == "observed" or not isinstance(certificate, Mapping):
            continue
        copy_interval = certificate.get("copy_agreement_interval")
        if isinstance(copy_interval, Mapping) and isinstance(copy_interval.get("upper"), int | float):
            control_uppers.append(float(copy_interval["upper"]))
    if observed_lower is not None and control_uppers:
        control_upper = max(control_uppers)
        certified = observed_lower > control_upper

    return {
        "confidence": confidence,
        "groups": group_certificates,
        "observed_copy_agreement_lower": observed_lower,
        "control_copy_agreement_upper": control_upper,
        "certified_inheritance": certified,
        "method": (
            "Wilson intervals for binary rates; inheritance flag compares "
            "observed lower bound against finite-shot control upper bounds"
        ),
    }


def expectation_from_counts(
    counts: Mapping[str, int],
    values: Mapping[str, float],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Estimate a bounded expectation from outcome counts and outcome values."""

    shots = validate_counts(counts)
    missing = set(counts) - set(values)
    if missing:
        raise ValueError(f"missing values for outcomes: {sorted(missing)}")
    estimate = sum(int(count) * float(values[outcome]) for outcome, count in counts.items()) / shots
    value_span = max(values.values()) - min(values.values())
    return hoeffding_interval(estimate, shots, confidence=confidence, value_range=value_span)


def pauli_correlation_certificate(
    counts: Mapping[str, int],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Estimate a two-qubit Pauli correlation from eigenbit counts.

    Outcomes are bitstrings where equal bits contribute +1 and unequal bits
    contribute -1.
    """

    shots = validate_counts(counts)
    values: dict[str, float] = {}
    for outcome in counts:
        if len(outcome) < 2:
            raise ValueError(f"correlation outcome {outcome!r} is not a bitstring")
        values[outcome] = 1.0 if outcome[0] == outcome[1] else -1.0
    estimate = sum(int(count) * values[outcome] for outcome, count in counts.items()) / shots
    return hoeffding_interval(estimate, shots, confidence=confidence, value_range=2.0)


def chsh_certificate(
    setting_counts: Mapping[str, Mapping[str, int]],
    confidence: float = 0.95,
    *,
    local_bound: float = 2.0,
) -> CHSHCertificate:
    """Finite-shot CHSH certificate from four setting-count dictionaries.

    Required setting keys are ``ab``, ``ab_prime``, ``a_prime_b``, and
    ``a_prime_b_prime``. The reported lower bound uses a union bound across the
    four setting correlations.
    """

    confidence = _validate_confidence(confidence)
    required = ("ab", "ab_prime", "a_prime_b", "a_prime_b_prime")
    missing = [key for key in required if key not in setting_counts]
    if missing:
        raise ValueError(f"missing CHSH setting counts: {missing}")

    alpha = 1.0 - confidence
    per_setting_confidence = 1.0 - alpha / len(required)
    intervals = {
        key: pauli_correlation_certificate(
            setting_counts[key],
            confidence=per_setting_confidence,
        )
        for key in required
    }
    estimate = (
        intervals["ab"].estimate
        + intervals["ab_prime"].estimate
        + intervals["a_prime_b"].estimate
        - intervals["a_prime_b_prime"].estimate
    )
    lower = (
        intervals["ab"].lower
        + intervals["ab_prime"].lower
        + intervals["a_prime_b"].lower
        - intervals["a_prime_b_prime"].upper
    )
    upper = (
        intervals["ab"].upper
        + intervals["ab_prime"].upper
        + intervals["a_prime_b"].upper
        - intervals["a_prime_b_prime"].lower
    )
    return CHSHCertificate(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        local_bound=float(local_bound),
        certified=bool(lower > local_bound),
        setting_estimates=intervals,
    )


def histogram_probabilities(
    counts: Mapping[str, int],
    confidence: float = 0.95,
) -> dict[str, ConfidenceInterval]:
    """Return Wilson intervals for every observed outcome probability."""

    shots = validate_counts(counts)
    return {
        outcome: wilson_interval(int(count), shots, confidence=confidence)
        for outcome, count in sorted(counts.items())
    }


def readout_mitigated_probabilities(
    counts: Mapping[str, int],
    inverse_confusion_matrix: list[list[float]],
) -> dict[str, float]:
    """Apply a precomputed inverse readout-confusion matrix to count data.

    The function intentionally accepts the inverse matrix, not calibration
    counts, so submissions must state their calibration assumptions explicitly.
    Negative quasi-probabilities are clipped and the result is renormalized for
    downstream scorecard use.
    """

    import numpy as np

    outcomes = sorted(counts)
    shots = validate_counts(counts)
    matrix = np.asarray(inverse_confusion_matrix, dtype=float)
    if matrix.shape != (len(outcomes), len(outcomes)):
        raise ValueError(
            "inverse_confusion_matrix shape must match the number of observed outcomes"
        )
    raw = np.asarray([counts[outcome] / shots for outcome in outcomes], dtype=float)
    mitigated = matrix @ raw
    mitigated = np.clip(mitigated, 0.0, None)
    total = float(np.sum(mitigated))
    if total <= 0.0:
        raise ValueError("mitigated probabilities have zero mass after clipping")
    mitigated = mitigated / total
    return {outcome: float(value) for outcome, value in zip(outcomes, mitigated)}
