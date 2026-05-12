from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qalbench import population_sweep  # noqa: E402


class PopulationSweepContractTests(unittest.TestCase):
    def test_positive_integer_arguments_are_bounded(self) -> None:
        for value in ("0", "-2"):
            with self.assertRaises(argparse.ArgumentTypeError):
                population_sweep.positive_int_arg(value)
        self.assertEqual(population_sweep.positive_int_arg("3"), 3)

    def test_population_size_arguments_require_two_or_more(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            population_sweep.population_size_arg("1")
        self.assertEqual(population_sweep.population_size_arg("2"), 2)

    def test_small_population_sweep_has_expected_shape(self) -> None:
        args = SimpleNamespace(
            first_seed=101,
            seed_count=1,
            steps=4,
            initial_population=6,
            carrying_capacity=14,
            base_birth_probability=0.28,
            base_death_probability=0.06,
            density_death_probability=0.08,
            mutation_probability=0.08,
            mutation_step=0.35,
            local_perturbation_probability=0.15,
            local_perturbation_angle=1.0471975511965976,
            damping_probability=0.10,
            phase_damping=0.0,
            interaction_strength=0.12,
            lineage_null_permutations=10,
        )
        summaries, trajectories, individuals, birth_events, attempts = population_sweep.run_population_sweep(args)
        expected_summaries = len(population_sweep.SCENARIOS) * len(population_sweep.MODELS)
        self.assertEqual(len(summaries), expected_summaries)
        self.assertEqual(len(trajectories), expected_summaries * (args.steps + 1))
        self.assertGreater(len(individuals), 0)
        self.assertGreater(len(birth_events), 0)
        self.assertGreater(len(attempts), 0)
        self.assertEqual(set(summaries[0]), set(population_sweep.SUMMARY_FIELDS))
        self.assertEqual(set(trajectories[0]), set(population_sweep.TRAJECTORY_FIELDS))
        self.assertEqual(set(individuals[0]), set(population_sweep.INDIVIDUAL_FIELDS))
        self.assertEqual(set(birth_events[0]), set(population_sweep.BIRTH_EVENT_FIELDS))
        self.assertEqual(set(attempts[0]), set(population_sweep.ATTEMPT_FIELDS))


if __name__ == "__main__":
    unittest.main()
